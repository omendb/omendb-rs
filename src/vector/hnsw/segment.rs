//! Segment-based storage for HNSW
//!
//! Segments enable incremental persistence and lock-free reads:
//!
//! - **MutableSegment**: Write-optimized, uses atomic storage for concurrent construction
//! - **FrozenSegment**: Read-optimized, uses unified colocated storage for fast search
//!
//! The segment pattern (similar to Qdrant/Milvus) allows:
//! - Writes to mutable segment without blocking reads
//! - Frozen segments can be mmap'd for memory efficiency
//! - Incremental persistence (only save new segments)

use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::types::{HNSWParams, Metric};

/// Search result from a segment
#[derive(Debug, Clone)]
pub struct SegmentSearchResult {
    /// Internal node ID within segment
    pub id: u32,
    /// Distance to query
    pub distance: f32,
    /// Original slot ID (for mapping back to RecordStore)
    pub slot: u32,
}

impl SegmentSearchResult {
    /// Create new search result
    pub fn new(id: u32, distance: f32, slot: u32) -> Self {
        Self { id, distance, slot }
    }
}

/// Mutable segment for writes
///
/// Wraps HNSWIndex with atomic neighbor storage for concurrent construction.
/// When full, can be frozen into a FrozenSegment for optimized reads.
pub struct MutableSegment {
    /// Underlying HNSW index
    index: HNSWIndex,
    /// Segment ID
    id: u64,
    /// Max capacity before freeze
    capacity: usize,
    /// Global slots for each local node ID (local_id → RecordStore slot)
    slots: Vec<u32>,
}

impl MutableSegment {
    /// Create new mutable segment
    pub fn new(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, false)?,
            id: 0,
            capacity: crate::vector::hnsw::SegmentConfig::DEFAULT_CAPACITY,
            slots: Vec::new(),
        })
    }

    /// Create with specific capacity
    pub fn with_capacity(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
        capacity: usize,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, false)?,
            id: 0,
            capacity,
            slots: Vec::with_capacity(capacity),
        })
    }

    /// Create with SQ8 quantization
    pub fn new_quantized(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: Metric,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, true)?,
            id: 0,
            capacity: crate::vector::hnsw::SegmentConfig::DEFAULT_CAPACITY,
            slots: Vec::new(),
        })
    }

    /// Create from an existing HNSWIndex with slot mapping
    ///
    /// Used for integrating parallel-built indexes into segment system.
    /// The slots slice must have one entry per vector in the index.
    pub fn from_index(index: HNSWIndex, slots: &[u32]) -> Self {
        debug_assert_eq!(
            index.len(),
            slots.len(),
            "Slot count must match vector count"
        );
        Self {
            id: 0,
            capacity: index
                .len()
                .max(crate::vector::hnsw::SegmentConfig::DEFAULT_CAPACITY),
            slots: slots.to_vec(),
            index,
        }
    }

    /// Create from an existing HNSWIndex using sequential slots starting at 0
    pub fn from_index_sequential(index: HNSWIndex) -> Self {
        let len = index.len();
        let slots: Vec<u32> = (0..len as u32).collect();
        Self {
            id: 0,
            capacity: len.max(crate::vector::hnsw::SegmentConfig::DEFAULT_CAPACITY),
            slots,
            index,
        }
    }

    /// Get segment ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Set segment ID
    pub fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    /// Number of vectors in segment
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Check if at capacity
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Get dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.index.dimensions()
    }

    /// Get HNSW parameters
    #[inline]
    pub fn params(&self) -> &HNSWParams {
        self.index.params()
    }

    /// Get distance function
    #[inline]
    pub fn distance_function(&self) -> Metric {
        self.index.distance_function()
    }

    /// Insert vector with global slot, returns internal ID
    ///
    /// The slot is the global RecordStore slot that will be returned in search results.
    pub fn insert_with_slot(
        &mut self,
        vector: &[f32],
        slot: u32,
    ) -> crate::vector::hnsw::error::Result<u32> {
        let local_id = self.index.insert(vector)?;
        debug_assert_eq!(
            local_id as usize,
            self.slots.len(),
            "Slot tracking out of sync: local_id={}, slots.len()={}",
            local_id,
            self.slots.len()
        );
        self.slots.push(slot);
        Ok(local_id)
    }

    /// Insert vector, returns internal ID (slot == local_id for backward compatibility)
    pub fn insert(&mut self, vector: &[f32]) -> crate::vector::hnsw::error::Result<u32> {
        let slot = self.slots.len() as u32;
        self.insert_with_slot(vector, slot)
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> crate::vector::hnsw::error::Result<Vec<SegmentSearchResult>> {
        let results = self.index.search(query, k, ef)?;
        Ok(results
            .into_iter()
            .map(|r| {
                // Map local_id to global slot
                let slot = self.slots.get(r.id as usize).copied().unwrap_or(r.id);
                SegmentSearchResult::new(r.id, r.distance, slot)
            })
            .collect())
    }

    /// Search for k nearest neighbors that match a filter predicate
    ///
    /// Delegates to HNSWIndex::search_with_filter which uses ACORN-1.
    /// The filter predicate receives global slots, not internal IDs.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> crate::vector::hnsw::error::Result<Vec<SegmentSearchResult>>
    where
        F: Fn(u32) -> bool,
    {
        // HNSWIndex::search_with_filter returns SearchResult with id=slot
        // For standard usage (no optimize), id == internal_id
        // We need to translate internal IDs to MutableSegment slots
        let results = self.index.search_with_filter(query, k, ef, |id| {
            // Map internal id to MutableSegment slot for filter
            let slot = self.slots.get(id as usize).copied().unwrap_or(id);
            filter_fn(slot)
        })?;

        Ok(results
            .into_iter()
            .map(|r| {
                // r.id from HNSWIndex is storage.slot(internal_id), which for standard
                // usage equals internal_id. Map to MutableSegment's slot.
                let slot = self.slots.get(r.id as usize).copied().unwrap_or(r.id);
                SegmentSearchResult::new(r.id, r.distance, slot)
            })
            .collect())
    }

    /// Get global slot for a local node ID
    #[inline]
    pub fn get_slot(&self, local_id: u32) -> Option<u32> {
        self.slots.get(local_id as usize).copied()
    }

    /// Get entry point
    #[inline]
    pub fn entry_point(&self) -> Option<u32> {
        self.index.entry_point()
    }

    /// Access underlying index (for advanced operations)
    pub fn index(&self) -> &HNSWIndex {
        &self.index
    }

    /// Mutable access to underlying index (for merging)
    pub fn index_mut(&mut self) -> &mut HNSWIndex {
        &mut self.index
    }

    /// Freeze into read-optimized segment
    ///
    /// This consumes the mutable segment and creates a frozen segment
    /// with colocated vector+neighbor storage for faster reads.
    pub fn freeze(self) -> FrozenSegment {
        FrozenSegment::from_mutable(self)
    }
}

/// Frozen segment for reads
///
/// Delegates search to an internal HNSWIndex for a single implementation
/// of the HNSW search algorithm. Cannot be modified after creation.
pub struct FrozenSegment {
    /// Underlying HNSW index (owns the colocated NodeStorage)
    index: HNSWIndex,
    /// Segment ID
    id: u64,
}

impl FrozenSegment {
    /// Construct from individual parts (for loading from disk)
    pub(crate) fn from_parts(
        id: u64,
        entry_point: Option<u32>,
        params: HNSWParams,
        distance_fn: Metric,
        storage: NodeStorage,
    ) -> Self {
        Self {
            index: HNSWIndex {
                storage,
                entry_point,
                params,
                distance_fn,
                rng_state: params.seed,
            },
            id,
        }
    }

    /// Create from mutable segment (zero-copy — moves the index directly)
    fn from_mutable(mutable: MutableSegment) -> Self {
        let MutableSegment {
            mut index,
            id,
            slots,
            ..
        } = mutable;

        // Force SQ8 training if still buffering (< 256 vectors)
        // so that node data contains actual quantized vectors
        index.storage.flush_sq8_training();

        // MutableSegment tracks slots separately in Vec<u32>;
        // write them into NodeStorage so FrozenSegment search returns correct slots
        index.remap_slots(&slots);

        // Reorder nodes into BFS traversal order so frozen segments are laid out
        // for cache-friendly reads. If optimization fails, keep the insertion-order
        // layout rather than failing freeze/persistence.
        if !index.is_empty() {
            if let Err(error) = index.optimize_cache_locality() {
                tracing::warn!("Failed to reorder frozen segment for cache locality: {error}");
            }
        }

        Self::from_parts(
            id,
            index.entry_point,
            *index.params(),
            index.distance_fn,
            index.storage,
        )
    }

    /// Get segment ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of vectors
    #[inline]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get entry point
    #[inline]
    pub fn entry_point(&self) -> Option<u32> {
        self.index.entry_point()
    }

    /// Get HNSW parameters
    #[inline]
    pub fn params(&self) -> &HNSWParams {
        self.index.params()
    }

    /// Get distance function
    #[inline]
    pub fn distance_function(&self) -> Metric {
        self.index.distance_function()
    }

    /// Search for k nearest neighbors
    ///
    /// Delegates to HNSWIndex::search which uses the colocated layout,
    /// SQ8 fast path, prefetch, and NeighborCollector for optimal search.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        match self.index.search(query, k, ef) {
            Ok(results) => results
                .into_iter()
                .map(|r| SegmentSearchResult::new(r.id, r.distance, r.id))
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Access underlying index (for diagnostics)
    pub fn index(&self) -> &HNSWIndex {
        &self.index
    }

    /// Access underlying storage (for persistence)
    pub fn storage(&self) -> &NodeStorage {
        &self.index.storage
    }

    /// Search for k nearest neighbors that match a filter predicate
    ///
    /// Delegates to HNSWIndex::search_with_filter which uses ACORN-1
    /// (arXiv:2403.04871) with adaptive 2-hop expansion for selective filters
    /// and post-filtering fallback for broad filters.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search expansion factor (higher = better recall, slower)
    /// * `filter_fn` - Predicate that takes a slot and returns true if it matches
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> Vec<SegmentSearchResult>
    where
        F: Fn(u32) -> bool,
    {
        match self.index.search_with_filter(query, k, ef, filter_fn) {
            Ok(results) => results
                .into_iter()
                .map(|r| SegmentSearchResult::new(r.id, r.distance, r.id))
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> HNSWParams {
        HNSWParams {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        }
    }

    #[test]
    fn test_mutable_segment_insert_and_search() {
        let mut segment = MutableSegment::new(4, default_params(), Metric::L2).unwrap();

        // Insert some vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        let id1 = segment.insert(&v1).unwrap();
        let id2 = segment.insert(&v2).unwrap();
        let id3 = segment.insert(&v3).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(segment.len(), 3);

        // Search for v1
        let results = segment.search(&v1, 3, 100).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0); // Should find v1 first
        assert!(results[0].distance < 0.001); // Should be very close
    }

    #[test]
    fn test_frozen_segment_search() {
        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();

        // Insert vectors
        mutable.insert(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 0.0, 1.0, 0.0]).unwrap();
        mutable.insert(&[0.0, 0.0, 0.0, 1.0]).unwrap();

        // Freeze
        let frozen = mutable.freeze();
        assert_eq!(frozen.len(), 4);

        // Search should work on frozen segment
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = frozen.search(&query, 2, 100);

        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0); // Should find first vector
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_frozen_segment_preserves_graph() {
        let mut mutable =
            MutableSegment::with_capacity(128, default_params(), Metric::L2, 1000).unwrap();

        // Insert 100 random-ish vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) % 256) as f32 / 256.0)
                .collect();
            mutable.insert(&vector).unwrap();
        }

        // Freeze
        let frozen = mutable.freeze();
        assert_eq!(frozen.len(), 100);

        // Search should return reasonable results
        let query: Vec<f32> = (0..128)
            .map(|j| (50 * 128 + j) as f32 / 256.0 % 1.0)
            .collect();
        let results = frozen.search(&query, 10, 100);

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_segment_capacity() {
        let segment = MutableSegment::with_capacity(4, default_params(), Metric::L2, 5).unwrap();
        assert!(!segment.is_full());
    }

    #[test]
    fn test_frozen_segment_filtered_search() {
        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();

        // Insert vectors with slots 0-9
        for i in 0..10 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            mutable.insert(&vector).unwrap();
        }

        let frozen = mutable.freeze();

        // Filter: only even slots
        let results =
            frozen.search_with_filter(&[4.0, 0.0, 0.0, 0.0], 3, 100, |slot| slot % 2 == 0);

        // Should find even-numbered results
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.slot % 2 == 0, "slot {} should be even", r.slot);
        }
        // Closest even should be slot 4
        assert_eq!(results[0].slot, 4);
    }

    #[test]
    fn test_frozen_segment_filtered_search_no_matches() {
        let mut mutable = MutableSegment::new(4, default_params(), Metric::L2).unwrap();

        for i in 0..10 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            mutable.insert(&vector).unwrap();
        }

        let frozen = mutable.freeze();

        // Filter: no matches
        let results = frozen.search_with_filter(&[5.0, 0.0, 0.0, 0.0], 3, 100, |_| false);

        assert!(results.is_empty());
    }

    #[test]
    fn test_freeze_applies_bfs_reorder_and_preserves_slots() {
        let mut chosen_seed = None;

        for seed in 1..=32 {
            let mut params = default_params();
            params.seed = seed;

            let mut mutable = MutableSegment::new(4, params, Metric::L2).unwrap();
            for i in 0..64 {
                mutable.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
            }

            if mutable.entry_point().is_some_and(|entry| entry != 0) {
                chosen_seed = Some(seed);
                break;
            }
        }

        let seed = chosen_seed.expect("expected a deterministic seed with a nonzero entry point");
        let mut params = default_params();
        params.seed = seed;

        let mut mutable = MutableSegment::new(4, params, Metric::L2).unwrap();
        for i in 0..64 {
            mutable.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        let pre_freeze_entry = mutable.entry_point().unwrap();
        assert_ne!(pre_freeze_entry, 0, "test setup requires a nonzero entry point");

        let frozen = mutable.freeze();

        // BFS reorder always places the entry point first in storage.
        assert_eq!(frozen.entry_point(), Some(0));

        let results = frozen.search(&[31.0, 0.0, 0.0, 0.0], 5, 100);
        assert!(!results.is_empty());
        assert_eq!(results[0].slot, 31);
    }

    #[test]
    fn test_mutable_segment_filtered_search() {
        let mut segment = MutableSegment::new(4, default_params(), Metric::L2).unwrap();

        // Insert with custom slots
        for i in 0..20 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            segment.insert_with_slot(&vector, i * 10).unwrap();
        }

        // Filter: slots divisible by 30 (0, 30, 60, 90, ...)
        let results =
            segment.search_with_filter(&[5.0, 0.0, 0.0, 0.0], 3, 100, |slot| slot % 30 == 0);

        assert!(results.is_ok());
        let results = results.unwrap();

        for r in &results {
            assert!(
                r.slot % 30 == 0,
                "slot {} should be divisible by 30",
                r.slot
            );
        }
    }
}
