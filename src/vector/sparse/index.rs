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
    pub fn insert(&mut self, slot: u32, vector: &SparseVector) {
        if self.slots.contains(slot) {
            self.remove(slot);
        }

        for (&dim, &weight) in vector.indices().iter().zip(vector.values().iter()) {
            self.postings
                .entry(dim)
                .or_insert_with(PostingList::new)
                .push(slot, weight);
        }
        self.slots.insert(slot);
    }

    /// Remove a slot from the index.
    pub fn remove(&mut self, slot: u32) {
        if !self.slots.contains(slot) {
            return;
        }
        for list in self.postings.values_mut() {
            list.remove(slot);
        }
        self.slots.remove(slot);
    }

    /// Search for top-k nearest neighbors via dot product.
    #[must_use]
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(u32, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scores: HashMap<u32, f32> = HashMap::with_capacity(k * 4);
        for (&dim, &query_weight) in query.indices().iter().zip(query.values().iter()) {
            if let Some(list) = self.postings.get(&dim) {
                for element in &list.elements {
                    *scores.entry(element.id).or_insert(0.0) += query_weight * element.weight;
                }
            }
        }
        top_k_from_scores(scores, k)
    }

    /// Search with metadata filter.
    #[must_use]
    pub fn search_with_filter<F>(
        &self,
        query: &SparseVector,
        k: usize,
        filter: F,
    ) -> Vec<(u32, f32)>
    where
        F: Fn(u32) -> bool,
    {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scores: HashMap<u32, f32> = HashMap::with_capacity(k * 4);
        for (&dim, &query_weight) in query.indices().iter().zip(query.values().iter()) {
            if let Some(list) = self.postings.get(&dim) {
                for element in &list.elements {
                    if filter(element.id) {
                        *scores.entry(element.id).or_insert(0.0) += query_weight * element.weight;
                    }
                }
            }
        }
        top_k_from_scores(scores, k)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let persisted = PersistedSparseIndex {
            postings: self.postings.clone().into_iter().collect(),
            len: self.len(),
        };
        postcard::to_allocvec(&persisted).map_err(|e| anyhow::anyhow!("SparseIndex serialize: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let persisted: PersistedSparseIndex = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseIndex deserialize: {e}"))?;
        Ok(Self {
            slots: derive_slots(&persisted.postings, persisted.len),
            postings: persisted.postings.into_iter().collect(),
        })
    }

    pub fn from_bytes_with_reconstructed_payloads(
        bytes: &[u8],
    ) -> Result<(Self, HashMap<u32, SparseVector>)> {
        let persisted: PersistedSparseIndex = postcard::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("SparseIndex deserialize: {e}"))?;
        let mut payloads: HashMap<u32, (Vec<u32>, Vec<f32>)> = HashMap::new();
        for (&dim, list) in &persisted.postings {
            for element in &list.elements {
                let entry = payloads
                    .entry(element.id)
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                entry.0.push(dim);
                entry.1.push(element.weight);
            }
        }
        let mut final_payloads = HashMap::with_capacity(payloads.len());
        for (slot, (indices, values)) in payloads {
            let mut combined: Vec<_> = indices.into_iter().zip(values.into_iter()).collect();
            combined.sort_by_key(|a| a.0);
            let (sorted_indices, sorted_values) = combined.into_iter().unzip();
            final_payloads.insert(slot, SparseVector::new(sorted_indices, sorted_values)?);
        }
        let slots = derive_slots(&persisted.postings, persisted.len);
        Ok((
            Self {
                postings: persisted.postings.into_iter().collect(),
                slots,
            },
            final_payloads,
        ))
    }

    pub fn remap_slot(&mut self, old_slot: u32, new_slot: u32) {
        if !self.slots.contains(old_slot) {
            return;
        }
        for list in self.postings.values_mut() {
            for element in &mut list.elements {
                if element.id == old_slot {
                    element.id = new_slot;
                }
            }
        }
        self.slots.remove(old_slot);
        self.slots.insert(new_slot);
    }

    pub fn compact(&mut self, mapping: &[u32]) {
        let mut new_slots = RoaringBitmap::new();
        for list in self.postings.values_mut() {
            list.elements.retain_mut(|element| {
                let old_id = element.id as usize;
                if old_id < mapping.len() && mapping[old_id] != u32::MAX {
                    let new_id = mapping[old_id];
                    element.id = new_id;
                    new_slots.insert(new_id);
                    true
                } else {
                    false
                }
            });
        }
        // Cleanup empty posting lists
        self.postings.retain(|_, list| !list.elements.is_empty());
        self.slots = new_slots;
    }

    pub fn to_bytes_with_payloads(
        &self,
        _payloads: impl IntoIterator<Item = (u32, SparseVector)>,
    ) -> Result<Vec<u8>> {
        self.to_bytes()
    }

    pub fn search_with_bitmap(
        &self,
        query: &SparseVector,
        k: usize,
        filter: &RoaringBitmap,
    ) -> Vec<(u32, f32)> {
        self.search_with_filter(query, k, |id| filter.contains(id))
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn top_k_from_scores(scores: impl IntoIterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    use std::cmp::Reverse;
    let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat<f32>, u32)>> =
        BinaryHeap::with_capacity(k + 1);
    for (id, score) in scores {
        heap.push(Reverse((ordered_float::OrderedFloat(score), id)));
        if heap.len() > k {
            heap.pop();
        }
    }
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
    len: usize,
}

fn derive_slots(postings: &HashMap<u32, PostingList>, legacy_len: usize) -> RoaringBitmap {
    let mut slots = RoaringBitmap::new();
    for posting in postings.values() {
        for entry in &posting.elements {
            slots.insert(entry.id);
        }
    }
    debug_assert!(legacy_len == 0 || slots.len() as usize == legacy_len);
    slots
}
