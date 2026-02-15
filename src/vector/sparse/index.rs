//! Inverted index for exact sparse vector search.
//!
//! Uses a HashMap of posting lists keyed by dimension ID. SPLADE vectors
//! typically use only ~5K of 30K possible dimensions per collection,
//! so a sparse map is more memory-efficient than a dense Vec.

use super::SparseVector;
use anyhow::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasher;

/// Single entry in a posting list.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostingElement {
    /// Internal slot ID.
    id: u32,
    /// Dimension weight for this document.
    weight: f32,
}

/// Posting list for one dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostingList {
    /// Elements (unsorted; order doesn't matter for exact search).
    elements: Vec<PostingElement>,
    /// Upper bound on weights (for future WAND optimization).
    max_weight: f32,
}

impl PostingList {
    fn new() -> Self {
        Self {
            elements: Vec::new(),
            max_weight: f32::NEG_INFINITY,
        }
    }

    fn push(&mut self, id: u32, weight: f32) {
        self.elements.push(PostingElement { id, weight });
        if weight > self.max_weight {
            self.max_weight = weight;
        }
    }

    fn remove(&mut self, id: u32) {
        self.elements.retain(|e| e.id != id);
        // Recompute max_weight
        self.max_weight = self
            .elements
            .iter()
            .map(|e| e.weight)
            .fold(f32::NEG_INFINITY, f32::max);
    }
}

/// Inverted index for sparse vector search.
///
/// Supports exact top-k dot product search via score accumulation.
/// At OmenDB's target scale (1K-100K), exact search completes in <2ms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndex {
    /// dim_id -> posting list
    postings: HashMap<u32, PostingList>,
    /// slot -> original sparse vector (for delete/update)
    vectors: HashMap<u32, SparseVector>,
    /// Legacy field kept for serialization compatibility. Derived from vectors.len().
    #[serde(default)]
    len: usize,
}

impl SparseIndex {
    /// Create an empty sparse index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            vectors: HashMap::new(),
            len: 0,
        }
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get the sparse vector for a given slot, if it exists.
    #[must_use]
    pub fn get(&self, slot: u32) -> Option<&SparseVector> {
        self.vectors.get(&slot)
    }

    /// Iterate over all (slot, sparse_vector) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &SparseVector)> {
        self.vectors.iter().map(|(&slot, vec)| (slot, vec))
    }

    /// Insert a sparse vector at the given slot.
    ///
    /// If the slot already exists, the old vector is removed first (upsert).
    pub fn insert(&mut self, slot: u32, vector: &SparseVector) {
        // Remove old vector if updating
        if self.vectors.contains_key(&slot) {
            self.remove(slot);
        }

        // Add to posting lists
        for (&dim, &weight) in vector.indices().iter().zip(vector.values().iter()) {
            self.postings
                .entry(dim)
                .or_insert_with(PostingList::new)
                .push(slot, weight);
        }

        // Store original vector for future removal
        self.vectors.insert(slot, vector.clone());
        self.len = self.vectors.len();
    }

    /// Remove a vector by slot ID.
    ///
    /// Returns true if the vector was found and removed.
    pub fn remove(&mut self, slot: u32) -> bool {
        let vector = match self.vectors.remove(&slot) {
            Some(v) => v,
            None => return false,
        };

        // Collect empty dims to remove in a second pass
        let mut empty_dims: Vec<u32> = Vec::new();
        for &dim in vector.indices() {
            if let Some(posting) = self.postings.get_mut(&dim) {
                posting.remove(slot);
                if posting.elements.is_empty() {
                    empty_dims.push(dim);
                }
            }
        }
        for dim in empty_dims {
            self.postings.remove(&dim);
        }

        self.len = self.vectors.len();
        true
    }

    /// Exact top-k search by dot product.
    ///
    /// Accumulates scores across posting lists for each query dimension,
    /// then returns top-k results sorted by score (descending).
    ///
    /// Returns `(slot, score)` pairs where score is the dot product (higher = better).
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        // Accumulate scores per document
        let mut scores: HashMap<u32, f32> = HashMap::new();

        for (&dim, &q_weight) in query.indices().iter().zip(query.values().iter()) {
            if let Some(posting) = self.postings.get(&dim) {
                for entry in &posting.elements {
                    *scores.entry(entry.id).or_default() += q_weight * entry.weight;
                }
            }
        }

        // Extract top-k using min-heap (Reverse so BinaryHeap acts as min-heap)
        top_k_from_scores(scores, k)
    }

    /// Filtered top-k search using a RoaringBitmap of allowed slots.
    pub fn search_with_bitmap(
        &self,
        query: &SparseVector,
        k: usize,
        allowed: &RoaringBitmap,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();

        for (&dim, &q_weight) in query.indices().iter().zip(query.values().iter()) {
            if let Some(posting) = self.postings.get(&dim) {
                for entry in &posting.elements {
                    if allowed.contains(entry.id) {
                        *scores.entry(entry.id).or_default() += q_weight * entry.weight;
                    }
                }
            }
        }

        top_k_from_scores(scores, k)
    }

    /// Move a sparse vector entry from one slot to another.
    ///
    /// Used when `RecordStore::upsert` creates a new slot for an existing ID.
    /// Returns true if the entry was found and remapped.
    pub fn remap_slot(&mut self, old_slot: u32, new_slot: u32) -> bool {
        let vector = match self.vectors.remove(&old_slot) {
            Some(v) => v,
            None => return false,
        };

        // Update slot IDs in posting lists
        for &dim in vector.indices() {
            if let Some(posting) = self.postings.get_mut(&dim) {
                for entry in &mut posting.elements {
                    if entry.id == old_slot {
                        entry.id = new_slot;
                        break;
                    }
                }
            }
        }

        self.vectors.insert(new_slot, vector);
        true
    }

    /// Remap slot IDs after compaction.
    ///
    /// Used when RecordStore compacts and old slots are renumbered.
    pub fn compact<S: BuildHasher>(&mut self, old_to_new: &HashMap<u32, u32, S>) {
        // Rebuild postings with new slot IDs, consuming old data
        let old_postings = std::mem::take(&mut self.postings);
        let mut new_postings: HashMap<u32, PostingList> = HashMap::new();
        for (dim, posting) in old_postings {
            let mut new_posting = PostingList::new();
            for entry in &posting.elements {
                if let Some(&new_id) = old_to_new.get(&entry.id) {
                    new_posting.push(new_id, entry.weight);
                }
            }
            if !new_posting.elements.is_empty() {
                new_postings.insert(dim, new_posting);
            }
        }
        self.postings = new_postings;

        // Rebuild vectors map with new slot IDs, consuming old data
        let old_vectors = std::mem::take(&mut self.vectors);
        let mut new_vectors: HashMap<u32, SparseVector> = HashMap::new();
        for (old_slot, vector) in old_vectors {
            if let Some(&new_slot) = old_to_new.get(&old_slot) {
                new_vectors.insert(new_slot, vector);
            }
        }
        self.vectors = new_vectors;
        self.len = self.vectors.len();
    }

    /// Serialize to bytes (postcard format).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        postcard::to_allocvec(self).map_err(|e| anyhow::anyhow!("SparseIndex serialize: {e}"))
    }

    /// Deserialize from bytes (postcard format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut index: Self = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseIndex deserialize: {e}"))?;

        // Validate embedded vectors
        for (slot, sv) in &index.vectors {
            if sv.indices().len() != sv.values().len() {
                anyhow::bail!(
                    "Corrupted SparseIndex: vector at slot {slot} has mismatched lengths"
                );
            }
            for window in sv.indices().windows(2) {
                if window[0] >= window[1] {
                    anyhow::bail!(
                        "Corrupted SparseIndex: vector at slot {slot} has unsorted indices"
                    );
                }
            }
            for &v in sv.values() {
                if !v.is_finite() {
                    anyhow::bail!(
                        "Corrupted SparseIndex: vector at slot {slot} has non-finite weight"
                    );
                }
            }
        }

        // Sync len from vectors (len field is legacy)
        index.len = index.vectors.len();

        Ok(index)
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract top-k entries by score from a HashMap using a min-heap.
fn top_k_from_scores(scores: HashMap<u32, f32>, k: usize) -> Vec<(u32, f32)> {
    use std::cmp::Reverse;

    // Use a min-heap bounded at size k
    let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat<f32>, u32)>> =
        BinaryHeap::with_capacity(k + 1);

    for (id, score) in scores {
        heap.push(Reverse((ordered_float::OrderedFloat(score), id)));
        if heap.len() > k {
            heap.pop();
        }
    }

    // Drain and sort descending by score
    let mut results: Vec<(u32, f32)> = heap
        .into_iter()
        .map(|Reverse((score, id))| (id, score.0))
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}
