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
use crate::vector::hnsw::query_buffers::VisitedList;
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};

use ordered_float::OrderedFloat;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

thread_local! {
    /// Thread-local visited list for FrozenSegment search
    /// Avoids Vec<bool> allocation per search, uses O(1) clear via generation counter
    static FROZEN_VISITED: RefCell<VisitedList> = RefCell::new(VisitedList::new());
}

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
        distance_fn: DistanceFunction,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, false)?,
            id: 0,
            capacity: 100_000, // Default capacity
            slots: Vec::new(),
        })
    }

    /// Create with specific capacity
    pub fn with_capacity(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        capacity: usize,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, false)?,
            id: 0,
            capacity,
            slots: Vec::with_capacity(capacity),
        })
    }

    /// Create with quantization
    pub fn new_quantized(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
    ) -> crate::vector::hnsw::error::Result<Self> {
        Ok(Self {
            index: HNSWIndex::new(dimensions, params, distance_fn, true)?,
            id: 0,
            capacity: 100_000,
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
            capacity: index.len().max(100_000),
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
            capacity: len.max(100_000),
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
    pub fn distance_function(&self) -> DistanceFunction {
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

    /// Access slots for freeze operation
    #[allow(dead_code)]
    pub(crate) fn slots(&self) -> &[u32] {
        &self.slots
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
/// Uses unified colocated storage for cache-efficient search.
/// Cannot be modified after creation.
pub struct FrozenSegment {
    /// Unified storage (colocated vectors + neighbors)
    storage: NodeStorage,
    /// Segment ID
    id: u64,
    /// Entry point for search
    entry_point: Option<u32>,
    /// HNSW parameters
    params: HNSWParams,
    /// Distance function
    distance_fn: DistanceFunction,
}

impl FrozenSegment {
    /// Construct from individual parts (for loading from disk)
    pub(crate) fn from_parts(
        id: u64,
        entry_point: Option<u32>,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        storage: NodeStorage,
    ) -> Self {
        Self {
            storage,
            id,
            entry_point,
            params,
            distance_fn,
        }
    }

    /// Create from mutable segment
    fn from_mutable(mutable: MutableSegment) -> Self {
        let dimensions = mutable.index.dimensions();
        let params = *mutable.index.params();
        let distance_fn = mutable.index.distance_function();
        let m = params.m;

        // Create unified storage
        let mut storage = NodeStorage::new(dimensions, m, params.max_level as usize);

        // Copy all nodes from mutable to frozen
        for id in 0..mutable.index.len() as u32 {
            storage.allocate_node();

            // Copy vector
            if let Some(vector) = mutable.index.get_vector(id) {
                storage.set_vector(id, vector);
            }

            // Copy level 0 neighbors (main graph layer)
            let neighbors = mutable.index.get_neighbors_level0(id);
            storage.set_neighbors(id, &neighbors);

            // Copy metadata and upper level neighbors
            if let Some(level) = mutable.index.node_level(id) {
                storage.set_level(id, level);

                // Copy upper level neighbors (levels 1+) for HNSW hierarchy
                for l in 1..=level {
                    let upper_neighbors = mutable.index.get_neighbors(id, l);
                    if !upper_neighbors.is_empty() {
                        storage.set_neighbors_at_level(id, l, upper_neighbors);
                    }
                }
            }
            // Use slot from mutable segment's slot tracking
            let slot = mutable.get_slot(id).unwrap_or(id);
            storage.set_slot(id, slot);
        }

        Self {
            storage,
            id: mutable.id,
            entry_point: mutable.index.entry_point(),
            params,
            distance_fn,
        }
    }

    /// Get segment ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of vectors
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get entry point
    #[inline]
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get HNSW parameters
    #[inline]
    pub fn params(&self) -> &HNSWParams {
        &self.params
    }

    /// Get distance function
    #[inline]
    pub fn distance_function(&self) -> DistanceFunction {
        self.distance_fn
    }

    /// Search for k nearest neighbors using unified storage
    ///
    /// This uses the colocated layout for cache-efficient search.
    /// Uses thread-local VisitedList for O(1) clear between searches.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        if self.storage.is_empty() {
            return Vec::new();
        }

        FROZEN_VISITED.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            visited.clear(); // O(1) via generation counter

            // Min-heap for candidates (closest first)
            let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>> = BinaryHeap::new();

            // Max-heap for results (furthest first, for trimming)
            let mut results: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();

            // Start from entry point
            let ep_dist = self.compute_distance(query, self.storage.vector(entry_point));
            visited.insert(entry_point);
            candidates.push(Reverse((OrderedFloat(ep_dist), entry_point)));
            results.push((OrderedFloat(ep_dist), entry_point));

            // Greedy search on level 0
            while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
                // Early termination: if current candidate is worse than worst result
                if results.len() >= ef {
                    if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                        if c_dist > worst_dist {
                            break;
                        }
                    }
                }

                // Get neighbors and prefetch
                let neighbors = self.storage.neighbors(c_id);

                // Prefetch first few neighbors
                for &neighbor in neighbors.iter().take(4) {
                    self.storage.prefetch(neighbor);
                }

                // Explore neighbors
                for &neighbor in neighbors {
                    if visited.contains(neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let n_dist = self.compute_distance(query, self.storage.vector(neighbor));

                    // Only add if better than worst result (or results not full)
                    let dominated = results.len() >= ef && {
                        let &(OrderedFloat(worst), _) = results.peek().unwrap();
                        n_dist > worst
                    };

                    if !dominated {
                        candidates.push(Reverse((OrderedFloat(n_dist), neighbor)));
                        results.push((OrderedFloat(n_dist), neighbor));

                        // Trim results if over ef
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }

            // Convert to output format, sorted by distance
            let mut output: Vec<_> = results
                .into_iter()
                .map(|(OrderedFloat(dist), id)| {
                    SegmentSearchResult::new(id, dist, self.storage.slot(id))
                })
                .collect();

            output.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            output.truncate(k);
            output
        })
    }

    /// Compute distance between query and candidate vector (SIMD-accelerated)
    #[inline]
    fn compute_distance(&self, query: &[f32], candidate: &[f32]) -> f32 {
        // Use SIMD-accelerated distance from DistanceFunction
        // For L2: returns squared distance (skips sqrt for faster comparisons)
        self.distance_fn.distance_for_comparison(query, candidate)
    }

    /// Access underlying storage (for advanced operations)
    pub fn storage(&self) -> &NodeStorage {
        &self.storage
    }

    // ============================================================================
    // Filtered Search (ACORN-1)
    // ============================================================================

    /// Search for k nearest neighbors that match a filter predicate
    ///
    /// Uses ACORN-1 algorithm (arXiv:2403.04871) with adaptive 2-hop expansion
    /// for efficient filtered search. Falls back to post-filtering for high
    /// selectivity (>60% match) or small segments (<1000 vectors).
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Search expansion factor (higher = better recall, slower)
    /// * `filter_fn` - Predicate that takes a slot and returns true if it matches
    ///
    /// # Performance
    /// - Low selectivity (5-20% match): 3-6x faster than post-filtering
    /// - High selectivity (>60% match): Falls back to post-filter
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
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        if self.storage.is_empty() {
            return Vec::new();
        }

        // Wrap filter to work on internal IDs - filter expects slots
        let slot_filter = |id: u32| filter_fn(self.storage.slot(id));

        // Estimate selectivity by sampling
        let selectivity = self.estimate_selectivity(&slot_filter);

        // Adaptive thresholds
        const SELECTIVITY_THRESHOLD: f32 = 0.6;
        const SMALL_SEGMENT_SIZE: usize = 1000;

        if selectivity > SELECTIVITY_THRESHOLD || self.len() <= SMALL_SEGMENT_SIZE {
            // High selectivity or small segment: use post-filter
            return self.search_with_postfilter(query, k, ef, &filter_fn);
        }

        // Low selectivity: use ACORN-1
        FROZEN_VISITED.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            visited.clear();

            // Start from entry point, descend to layer 0
            let entry_level = self.storage.level(entry_point);
            let mut nearest = vec![entry_point];

            // Greedy search at upper layers (find nearest matching node)
            for level in (1..=entry_level).rev() {
                nearest = self.search_layer_filtered(
                    query,
                    &nearest,
                    1,
                    level,
                    &slot_filter,
                    &mut visited,
                );
                if nearest.is_empty() {
                    nearest = vec![entry_point];
                }
            }

            // Beam search at layer 0
            let candidates = self.search_layer_filtered(
                query,
                &nearest,
                ef.max(k),
                0,
                &slot_filter,
                &mut visited,
            );

            // Convert to results and sort
            let mut results: Vec<_> = candidates
                .into_iter()
                .map(|id| {
                    let dist = self.compute_distance(query, self.storage.vector(id));
                    SegmentSearchResult::new(id, dist, self.storage.slot(id))
                })
                .collect();

            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);

            // Fallback if ACORN-1 found too few results
            if results.len() < k {
                return self.search_with_postfilter(query, k, ef, &filter_fn);
            }

            results
        })
    }

    /// Post-filter search: standard search followed by filtering
    fn search_with_postfilter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: &F,
    ) -> Vec<SegmentSearchResult>
    where
        F: Fn(u32) -> bool,
    {
        // Oversample to account for filtered items
        let oversample_k = (k * 10).min(self.len());
        let search_ef = ef.max(oversample_k);

        let mut results = self.search(query, oversample_k, search_ef);
        results.retain(|r| filter_fn(r.slot));
        results.truncate(k);
        results
    }

    /// Estimate filter selectivity by sampling
    fn estimate_selectivity<F>(&self, filter_fn: &F) -> f32
    where
        F: Fn(u32) -> bool,
    {
        const SAMPLE_SIZE: usize = 100;
        let n = self.len();
        if n == 0 {
            return 0.0;
        }

        let sample_count = SAMPLE_SIZE.min(n);
        let step = n / sample_count;
        let mut matches = 0;

        for i in 0..sample_count {
            let id = (i * step) as u32;
            if filter_fn(id) {
                matches += 1;
            }
        }

        matches as f32 / sample_count as f32
    }

    /// ACORN-1 layer search with 2-hop expansion
    fn search_layer_filtered<F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
        visited: &mut VisitedList,
    ) -> Vec<u32>
    where
        F: Fn(u32) -> bool,
    {
        let m = self.params.m;

        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>> = BinaryHeap::new();
        // Max-heap for results (furthest first, for trimming)
        let mut results: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        // Buffer for ACORN-1 neighbor collection
        let mut neighbor_buffer = Vec::with_capacity(m * 2);

        // Initialize with entry points
        for &ep in entry_points {
            if !visited.contains(ep) {
                visited.insert(ep);
                let dist = self.compute_distance(query, self.storage.vector(ep));
                candidates.push(Reverse((OrderedFloat(dist), ep)));
                if filter_fn(ep) {
                    results.push((OrderedFloat(dist), ep));
                }
            }
        }

        // Greedy search with ACORN-1 neighbor expansion
        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            // Early termination
            if results.len() >= ef {
                if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                    if c_dist > worst_dist {
                        break;
                    }
                }
            }

            // ACORN-1: collect matching neighbors with 2-hop expansion (shared impl)
            super::acorn::collect_matching_neighbors(
                &self.storage,
                c_id,
                level,
                visited,
                filter_fn,
                m,
                &mut neighbor_buffer,
            );

            // Process collected neighbors
            for &neighbor_id in &neighbor_buffer {
                if visited.contains(neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let n_dist = self.compute_distance(query, self.storage.vector(neighbor_id));

                let dominated = results.len() >= ef && {
                    let &(OrderedFloat(worst), _) = results.peek().unwrap();
                    n_dist > worst
                };

                if !dominated {
                    candidates.push(Reverse((OrderedFloat(n_dist), neighbor_id)));
                    if filter_fn(neighbor_id) {
                        results.push((OrderedFloat(n_dist), neighbor_id));
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results.into_iter().map(|(_, id)| id).collect()
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
        let mut segment = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();

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
        let mut mutable = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();

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
            MutableSegment::with_capacity(128, default_params(), DistanceFunction::L2, 1000)
                .unwrap();

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
        let segment =
            MutableSegment::with_capacity(4, default_params(), DistanceFunction::L2, 5).unwrap();
        assert!(!segment.is_full());
    }

    #[test]
    fn test_frozen_segment_filtered_search() {
        let mut mutable = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();

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
        let mut mutable = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();

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
    fn test_mutable_segment_filtered_search() {
        let mut segment = MutableSegment::new(4, default_params(), DistanceFunction::L2).unwrap();

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
