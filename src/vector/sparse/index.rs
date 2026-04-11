//! Inverted index for exact sparse vector search.
//!
//! Uses a HashMap of posting lists keyed by dimension ID. SPLADE vectors
//! typically use only ~5K of 30K possible dimensions per collection,
//! so a sparse map is more memory-efficient than a dense Vec.

use super::SparseVector;
use anyhow::Result;
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
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
}

impl PostingList {
    fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    fn push(&mut self, id: u32, weight: f32) {
        self.elements.push(PostingElement { id, weight });
    }

    fn remove(&mut self, id: u32) {
        self.elements.retain(|e| e.id != id);
    }
}

/// Inverted index for sparse vector search.
///
/// Supports exact top-k dot product search via score accumulation.
/// At OmenDB's target scale (1K-100K), exact search completes in <2ms.
#[derive(Debug, Clone)]
pub struct SparseIndex {
    /// dim_id -> posting list
    postings: FxHashMap<u32, PostingList>,
    /// Indexed slots.
    slots: RoaringBitmap,
}

impl SparseIndex {
    /// Create an empty sparse index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            postings: FxHashMap::default(),
            slots: RoaringBitmap::new(),
        }
    }

    /// Number of indexed vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len() as usize
    }

    /// Check if index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Insert a sparse vector at the given slot.
    ///
    /// If the slot already exists, the old vector is removed first (upsert).
    pub fn insert(&mut self, slot: u32, vector: &SparseVector) {
        // Remove old postings if updating
        if self.slots.contains(slot) {
            self.remove(slot);
        }

        debug_assert_eq!(
            vector.indices().len(),
            vector.values().len(),
            "SparseVector indices and values length mismatch"
        );

        // Add to posting lists
        for (&dim, &weight) in vector.indices().iter().zip(vector.values().iter()) {
            self.postings
                .entry(dim)
                .or_insert_with(PostingList::new)
                .push(slot, weight);
        }

        self.slots.insert(slot);
    }

    /// Remove a vector by slot ID.
    ///
    /// Returns true if the vector was found and removed.
    pub fn remove(&mut self, slot: u32) -> bool {
        if !self.slots.contains(slot) {
            return false;
        }

        self.slots.remove(slot);
        let mut empty_dims: Vec<u32> = Vec::new();
        for (&dim, posting) in &mut self.postings {
            posting.remove(slot);
            if posting.elements.is_empty() {
                empty_dims.push(dim);
            }
        }
        for dim in empty_dims {
            self.postings.remove(&dim);
        }
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
        let estimated_candidates = query
            .indices()
            .iter()
            .filter_map(|dim| self.postings.get(dim).map(|posting| posting.elements.len()))
            .sum::<usize>();
        let mut scores: FxHashMap<u32, f32> = FxHashMap::default();
        scores.reserve(estimated_candidates);

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

        let estimated_candidates = query
            .indices()
            .iter()
            .filter_map(|dim| self.postings.get(dim).map(|posting| posting.elements.len()))
            .sum::<usize>();
        let mut scores: FxHashMap<u32, f32> = FxHashMap::default();
        scores.reserve(estimated_candidates);

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
        if !self.slots.contains(old_slot) {
            return false;
        }

        // Update slot IDs in posting lists
        for posting in self.postings.values_mut() {
            for entry in &mut posting.elements {
                if entry.id == old_slot {
                    entry.id = new_slot;
                }
            }
        }

        self.slots.remove(old_slot);
        self.slots.insert(new_slot);
        true
    }

    /// Remap slot IDs after compaction.
    ///
    /// Used when RecordStore compacts and old slots are renumbered.
    pub fn compact<S: BuildHasher>(&mut self, old_to_new: &HashMap<u32, u32, S>) {
        // Rebuild postings with new slot IDs, consuming old data
        let old_postings = std::mem::take(&mut self.postings);
        let mut new_postings: FxHashMap<u32, PostingList> = FxHashMap::default();
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

        self.slots = old_to_new.values().copied().collect();
    }

    /// Serialize to bytes (postcard format).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let persisted = PersistedSparseIndex {
            postings: self.postings.clone().into_iter().collect(),
            payloads: HashMap::new(),
            len: self.len(),
        };
        postcard::to_allocvec(&persisted).map_err(|e| anyhow::anyhow!("SparseIndex serialize: {e}"))
    }

    /// Serialize to bytes with sparse payloads for recovery.
    pub fn to_bytes_with_payloads(
        &self,
        payloads: impl IntoIterator<Item = (u32, SparseVector)>,
    ) -> Result<Vec<u8>> {
        let persisted = PersistedSparseIndex {
            postings: self.postings.clone().into_iter().collect(),
            payloads: payloads.into_iter().collect(),
            len: self.len(),
        };
        postcard::to_allocvec(&persisted).map_err(|e| anyhow::anyhow!("SparseIndex serialize: {e}"))
    }

    /// Deserialize from bytes (postcard format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let persisted: PersistedSparseIndex = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseIndex deserialize: {e}"))?;
        Ok(Self {
            slots: derive_slots(&persisted.postings, &persisted.payloads, persisted.len),
            postings: persisted.postings.into_iter().collect(),
        })
    }

    /// Deserialize from bytes and return sparse payloads for RecordStore recovery.
    pub fn from_bytes_with_payloads(bytes: &[u8]) -> Result<(Self, HashMap<u32, SparseVector>)> {
        let persisted: PersistedSparseIndex = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseIndex deserialize: {e}"))?;

        for (slot, sv) in &persisted.payloads {
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

        let slots = derive_slots(&persisted.postings, &persisted.payloads, persisted.len);
        Ok((
            Self {
                postings: persisted.postings.into_iter().collect(),
                slots,
            },
            persisted.payloads,
        ))
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract top-k entries by score from a HashMap using a min-heap.
fn top_k_from_scores(scores: impl IntoIterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSparseIndex {
    postings: HashMap<u32, PostingList>,
    #[serde(default)]
    payloads: HashMap<u32, SparseVector>,
    #[serde(default)]
    len: usize,
}

fn derive_slots(
    postings: &HashMap<u32, PostingList>,
    payloads: &HashMap<u32, SparseVector>,
    legacy_len: usize,
) -> RoaringBitmap {
    if !payloads.is_empty() {
        return payloads.keys().copied().collect();
    }

    let mut slots = RoaringBitmap::new();
    for posting in postings.values() {
        for entry in &posting.elements {
            slots.insert(entry.id);
        }
    }

    debug_assert!(legacy_len == 0 || slots.len() as usize == legacy_len);
    slots
}
