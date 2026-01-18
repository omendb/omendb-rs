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
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use crate::vector::hnsw::unified_storage::UnifiedNodeStorage;

use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

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
        })
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

    /// Insert vector, returns internal ID
    pub fn insert(&mut self, vector: &[f32]) -> crate::vector::hnsw::error::Result<u32> {
        self.index.insert(vector)
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
            .map(|r| SegmentSearchResult::new(r.id, r.distance, r.id)) // In mutable, id == slot
            .collect())
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
    storage: UnifiedNodeStorage,
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
    /// Create from mutable segment
    fn from_mutable(mutable: MutableSegment) -> Self {
        let dimensions = mutable.index.dimensions();
        let params = *mutable.index.params();
        let distance_fn = mutable.index.distance_function();
        let m = params.m;

        // Create unified storage
        let mut storage = UnifiedNodeStorage::new(dimensions, m, params.max_level as usize);

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

            // Copy metadata
            if let Some(level) = mutable.index.node_level(id) {
                storage.set_level(id, level);
            }
            storage.set_slot(id, id); // In mutable segment, slot == id
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
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<SegmentSearchResult> {
        let Some(entry_point) = self.entry_point else {
            return Vec::new();
        };

        if self.storage.is_empty() {
            return Vec::new();
        }

        // Visited tracking
        let mut visited = vec![false; self.storage.len()];

        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>> = BinaryHeap::new();

        // Max-heap for results (furthest first, for trimming)
        let mut results: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();

        // Start from entry point
        let ep_dist = self.compute_distance(query, self.storage.vector(entry_point));
        visited[entry_point as usize] = true;
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
                if visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;

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

        output.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        output.truncate(k);
        output
    }

    /// Compute distance between query and candidate vector
    #[inline]
    fn compute_distance(&self, query: &[f32], candidate: &[f32]) -> f32 {
        match self.distance_fn {
            DistanceFunction::L2 => {
                // Squared L2 distance
                query
                    .iter()
                    .zip(candidate.iter())
                    .map(|(a, b)| {
                        let diff = a - b;
                        diff * diff
                    })
                    .sum()
            }
            DistanceFunction::Cosine => {
                // Cosine distance = 1 - cosine_similarity
                let dot: f32 = query.iter().zip(candidate.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_c: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (norm_q * norm_c + 1e-10)
            }
            DistanceFunction::NegativeDotProduct => {
                // Negative dot product (for max inner product search)
                -query
                    .iter()
                    .zip(candidate.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            }
        }
    }

    /// Access underlying storage (for advanced operations)
    pub fn storage(&self) -> &UnifiedNodeStorage {
        &self.storage
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
}
