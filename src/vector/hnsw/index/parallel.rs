//! Parallel HNSW construction
//!
//! Implements parallel graph construction using:
//! - Per-node fine-grained locking (hnswlib pattern)
//! - Ready bitmap for correctness (Qdrant pattern)
//! - Warm start: first N nodes sequential, then parallel
//! - Lock ordering: lower node ID first (prevents deadlocks)
//! - Cached vectors for lock-free distance computation
//!
//! Reference implementations:
//! - hnswlib: https://github.com/nmslib/hnswlib
//! - Qdrant: https://github.com/qdrant/qdrant
//! - pgvector: https://github.com/pgvector/pgvector

use super::HNSWIndex;
use crate::vector::hnsw::atomic_bitvec::AtomicBitVec;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::query_buffers::VisitedList;
use crate::vector::hnsw::types::{Candidate, DistanceFunction, HNSWParams};
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::cell::{RefCell, UnsafeCell};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use tracing::{debug, info};

/// Reusable buffers for parallel build operations
struct BuildBuffers {
    /// Visited nodes during search
    visited: VisitedList,
    /// Candidate queue (min-heap)
    candidates: BinaryHeap<std::cmp::Reverse<Candidate>>,
    /// Working set (max-heap)
    working: BinaryHeap<Candidate>,
    /// Results with distances
    results_with_dist: Vec<(u32, f32)>,
}

impl BuildBuffers {
    fn new() -> Self {
        Self {
            visited: VisitedList::new(),
            candidates: BinaryHeap::new(),
            working: BinaryHeap::new(),
            results_with_dist: Vec::new(),
        }
    }
}

thread_local! {
    /// Thread-local build buffers for parallel construction
    /// Avoids allocations per insert, uses O(1) clear via generation counter
    static BUILD_BUFFERS: RefCell<BuildBuffers> = RefCell::new(BuildBuffers::new());
}

/// Parallel HNSW builder
///
/// Uses cached vectors for lock-free distance computation.
/// Only neighbor lists require synchronization via per-node locks.
///
/// SAFETY: The storage is wrapped in UnsafeCell for interior mutability.
/// We ensure thread safety by:
/// 1. Using per-node locks for neighbor list modifications
/// 2. Vectors are immutable after allocation (stored in separate Vec)
/// 3. Node metadata (level, slot) is immutable after allocation
pub struct ParallelBuilder {
    /// Node storage (for neighbor lists and metadata)
    /// Wrapped in UnsafeCell for interior mutability with manual synchronization
    storage: UnsafeCell<NodeStorage>,
    /// Cached vectors for lock-free distance computation
    vectors: Vec<Vec<f32>>,
    /// Node levels
    levels: Vec<u8>,
    /// Per-node locks for neighbor list modification
    node_locks: Vec<Mutex<()>>,
    /// Bitmap tracking which nodes are fully connected
    ready_bitmap: AtomicBitVec,
    /// Atomic entry point (u32::MAX = None)
    entry_point: AtomicU32,
    /// Construction parameters
    params: HNSWParams,
    /// Distance function
    distance_fn: DistanceFunction,
    /// Random number generator state (atomic for thread-safe level assignment)
    rng_state: AtomicU64,
}

/// Number of nodes to insert sequentially before switching to parallel
const WARM_START_SIZE: usize = 512;

impl ParallelBuilder {
    /// Create a new parallel builder
    pub fn new(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
    ) -> Result<Self> {
        params.validate().map_err(HNSWError::InvalidParams)?;

        let storage = if use_quantization {
            NodeStorage::new_sq8(dimensions, params.m, params.max_level as usize)
        } else {
            NodeStorage::new(dimensions, params.m, params.max_level as usize)
        };

        Ok(Self {
            storage: UnsafeCell::new(storage),
            vectors: Vec::new(),
            levels: Vec::new(),
            node_locks: Vec::new(),
            ready_bitmap: AtomicBitVec::empty(),
            entry_point: AtomicU32::new(u32::MAX),
            rng_state: AtomicU64::new(params.seed),
            params,
            distance_fn,
        })
    }

    /// Get mutable reference to storage
    ///
    /// SAFETY: Caller must ensure exclusive access via appropriate locking
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn storage(&self) -> &mut NodeStorage {
        &mut *self.storage.get()
    }

    /// Build index from vectors using parallel construction
    pub fn build(mut self, vectors: Vec<Vec<f32>>) -> Result<HNSWIndex> {
        if vectors.is_empty() {
            return Ok(self.into_index());
        }

        let batch_size = vectors.len();
        info!(batch_size, "Starting parallel HNSW construction");
        let start = std::time::Instant::now();

        // Validate all vectors for dimensions and NaN/Inf
        // SAFETY: No concurrent access yet - we're still in single-threaded setup
        let dimensions = unsafe { self.storage() }.dimensions();
        for vec in &vectors {
            if vec.len() != dimensions {
                return Err(HNSWError::DimensionMismatch {
                    expected: dimensions,
                    actual: vec.len(),
                });
            }
            if vec.iter().any(|x| !x.is_finite()) {
                return Err(HNSWError::InvalidVector);
            }
        }
        debug!(batch_size, dimensions, "Validated all vectors");

        // Phase 1: Allocate all nodes, assign levels, store vectors
        self.allocate_all_nodes(&vectors);
        debug!(nodes = batch_size, "Allocated all nodes");

        // Phase 2: Initialize concurrency structures
        self.init_concurrency(batch_size);

        // Phase 3: Warm start - sequential insertion of first N nodes
        let warm_size = WARM_START_SIZE.min(batch_size);
        for node_id in 0..warm_size as u32 {
            self.insert_sequential(node_id)?;
        }
        debug!(warm_size, "Warm start complete");

        // Phase 4: Parallel insertion of remaining nodes
        if batch_size > warm_size {
            let parallel_count = batch_size - warm_size;

            (warm_size as u32..batch_size as u32)
                .into_par_iter()
                .try_for_each(|node_id| self.insert_parallel(node_id))?;

            debug!(parallel_count, "Parallel insertion complete");
        }

        let elapsed = start.elapsed();
        let rate = batch_size as f64 / elapsed.as_secs_f64();
        info!(
            batch_size,
            elapsed_secs = elapsed.as_secs_f64(),
            rate_vec_per_sec = rate as u64,
            "Parallel construction complete"
        );

        Ok(self.into_index())
    }

    /// Allocate all nodes and assign levels
    fn allocate_all_nodes(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        self.vectors = vectors.to_vec();
        self.levels = Vec::with_capacity(n);

        // SAFETY: Single-threaded setup phase - no concurrent access
        // Using raw pointer to avoid borrow checker issues with self.levels
        let storage_ptr = self.storage.get();

        for vector in vectors {
            let storage = unsafe { &mut *storage_ptr };
            let node_id = storage.allocate_node();
            let level = self.random_level();

            storage.set_vector(node_id, vector);
            storage.set_slot(node_id, node_id);
            storage.set_level(node_id, level);

            if level > 0 {
                storage.allocate_upper_levels(node_id, level);
            }

            self.levels.push(level);
        }
    }

    /// Initialize concurrency structures
    fn init_concurrency(&mut self, capacity: usize) {
        self.node_locks = (0..capacity).map(|_| Mutex::new(())).collect();
        self.ready_bitmap = AtomicBitVec::new(capacity);
    }

    /// Assign random level using atomic RNG
    fn random_level(&self) -> u8 {
        let mut state = self.rng_state.load(Ordering::Relaxed);
        loop {
            let new_state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            match self.rng_state.compare_exchange_weak(
                state,
                new_state,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let rand_val = (new_state >> 32) as f32 / u32::MAX as f32;
                    let level = (-rand_val.ln() * self.params.ml) as u8;
                    return level.min(self.params.max_level - 1);
                }
                Err(actual) => state = actual,
            }
        }
    }

    /// Sequential insertion (used for warm start)
    fn insert_sequential(&self, node_id: u32) -> Result<()> {
        // First node becomes entry point
        if self
            .entry_point
            .compare_exchange(u32::MAX, node_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.ready_bitmap.set(node_id as usize);
            return Ok(());
        }

        let level = self.levels[node_id as usize];
        self.insert_into_graph(node_id, level, false)?;

        self.ready_bitmap.set(node_id as usize);
        self.maybe_update_entry_point(node_id, level);

        Ok(())
    }

    /// Parallel insertion with fine-grained locking
    fn insert_parallel(&self, node_id: u32) -> Result<()> {
        let level = self.levels[node_id as usize];
        self.insert_into_graph(node_id, level, true)?;

        self.ready_bitmap.set(node_id as usize);
        self.maybe_update_entry_point(node_id, level);

        Ok(())
    }

    /// Insert node into graph
    fn insert_into_graph(&self, node_id: u32, level: u8, use_ready_filter: bool) -> Result<()> {
        let entry_point = self.entry_point.load(Ordering::Acquire);
        if entry_point == u32::MAX {
            return Err(HNSWError::EmptyIndex);
        }

        let entry_level = self.levels[entry_point as usize];
        let vector = &self.vectors[node_id as usize];

        // Search for nearest neighbors
        let mut nearest = vec![entry_point];

        // Descend from top level to target level (just need IDs here)
        for lc in ((level + 1)..=entry_level).rev() {
            nearest = self.search_layer(vector, &nearest, 1, lc, use_ready_filter);
        }

        // Insert at each level from target down to 0
        for lc in (0..=level).rev() {
            // Use distance-aware search to avoid recomputing in heuristic
            let candidates_with_distances = self.search_layer_with_distances(
                vector,
                &nearest,
                self.params.ef_construction,
                lc,
                use_ready_filter,
            );

            let m = self.params.m_for_level(lc);
            // Use pre-computed distances
            let neighbors =
                self.select_neighbors_heuristic_with_distances(&candidates_with_distances, m);

            self.connect_with_locks(node_id, &neighbors, lc);

            // Extract IDs for next level
            nearest = candidates_with_distances
                .iter()
                .map(|(id, _)| *id)
                .collect();
        }

        Ok(())
    }

    /// Search layer with optional ready filter, returning (node_id, distance) pairs
    ///
    /// Uses lock-free distance computation (vectors are cached).
    /// Only neighbor list reads require synchronization.
    /// Uses thread-local buffers to avoid per-search allocations.
    /// Returns distances to avoid recomputation in select_neighbors_heuristic.
    fn search_layer_with_distances(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        use_ready_filter: bool,
    ) -> Vec<(u32, f32)> {
        use std::cmp::Reverse;

        BUILD_BUFFERS.with(|buffers_cell| {
            let mut buffers = buffers_cell.borrow_mut();
            // Clear but keep capacity
            buffers.visited.clear();
            buffers.candidates.clear();
            buffers.working.clear();
            buffers.results_with_dist.clear();

            // Initialize with entry points
            for &ep in entry_points {
                if use_ready_filter && !self.ready_bitmap.is_ready(ep as usize) {
                    continue;
                }
                if buffers.visited.contains(ep) {
                    continue;
                }
                buffers.visited.insert(ep);

                let dist = self.distance_cmp(query, ep);
                let candidate = Candidate::new(ep, dist);
                buffers.candidates.push(Reverse(candidate));
                buffers.working.push(candidate);
            }

            if buffers.candidates.is_empty() {
                return Vec::new();
            }

            // Greedy search
            while let Some(Reverse(current)) = buffers.candidates.pop() {
                if let Some(&farthest) = buffers.working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Get neighbors (requires lock)
                let neighbors = {
                    let _lock = self.node_locks[current.node_id as usize].lock();
                    // SAFETY: Protected by per-node lock
                    unsafe { self.storage() }.neighbors_at_level(current.node_id, level)
                };

                for neighbor_id in neighbors {
                    if buffers.visited.contains(neighbor_id) {
                        continue;
                    }
                    buffers.visited.insert(neighbor_id);

                    if use_ready_filter && !self.ready_bitmap.is_ready(neighbor_id as usize) {
                        continue;
                    }

                    let dist = self.distance_cmp(query, neighbor_id);
                    let neighbor = Candidate::new(neighbor_id, dist);

                    if let Some(&farthest) = buffers.working.peek() {
                        if neighbor.distance < farthest.distance || buffers.working.len() < ef {
                            buffers.candidates.push(Reverse(neighbor));
                            buffers.working.push(neighbor);
                            if buffers.working.len() > ef {
                                buffers.working.pop();
                            }
                        }
                    } else {
                        buffers.candidates.push(Reverse(neighbor));
                        buffers.working.push(neighbor);
                    }
                }
            }

            // Collect results sorted by distance
            let working_len = buffers.working.len();
            buffers.results_with_dist.clear();
            buffers.results_with_dist.reserve(working_len);
            while let Some(c) = buffers.working.pop() {
                buffers
                    .results_with_dist
                    .push((c.node_id, c.distance.into_inner()));
            }
            buffers
                .results_with_dist
                .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Return a clone (buffer stays for next use)
            buffers.results_with_dist.clone()
        })
    }

    /// Search layer returning only node IDs (for upper level traversal)
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        use_ready_filter: bool,
    ) -> Vec<u32> {
        self.search_layer_with_distances(query, entry_points, ef, level, use_ready_filter)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Connect node to neighbors with fine-grained locking
    fn connect_with_locks(&self, node_id: u32, neighbors: &[u32], level: u8) {
        let m = self.params.m_for_level(level);

        // Set forward links (node -> neighbors)
        {
            let _lock = self.node_locks[node_id as usize].lock();
            // SAFETY: Protected by per-node lock
            unsafe { self.storage() }.set_neighbors_at_level(node_id, level, neighbors.to_vec());
        }

        // Add reverse links (neighbors -> node) with proper pruning
        for &neighbor_id in neighbors {
            // Lock ordering: lower ID first
            let (first_id, second_id) = if node_id < neighbor_id {
                (node_id, neighbor_id)
            } else {
                (neighbor_id, node_id)
            };

            let _lock1 = self.node_locks[first_id as usize].lock();
            let _lock2 = if first_id == second_id {
                None
            } else {
                Some(self.node_locks[second_id as usize].lock())
            };

            // SAFETY: Protected by per-node locks (both nodes locked in order)
            let storage = unsafe { self.storage() };

            // Check if already connected
            if storage.contains_neighbor(neighbor_id, level, node_id) {
                continue;
            }

            let mut neighbor_neighbors = storage.neighbors_at_level(neighbor_id, level);
            neighbor_neighbors.push(node_id);

            // Prune if over capacity
            if neighbor_neighbors.len() > m {
                let neighbor_vec = &self.vectors[neighbor_id as usize];
                neighbor_neighbors =
                    self.select_neighbors_heuristic(&neighbor_neighbors, m, neighbor_vec);
            }

            storage.set_neighbors_at_level(neighbor_id, level, neighbor_neighbors);
        }
    }

    /// Select neighbors using diversity heuristic (with pre-computed distances)
    ///
    /// Accepts candidates with distances already computed from search phase
    /// to avoid recomputing them.
    fn select_neighbors_heuristic_with_distances(
        &self,
        candidates_with_distances: &[(u32, f32)],
        m: usize,
    ) -> Vec<u32> {
        if candidates_with_distances.len() <= m {
            return candidates_with_distances
                .iter()
                .map(|(id, _)| *id)
                .collect();
        }

        // Already sorted by distance from search_layer_with_distances
        let mut result = Vec::with_capacity(m);
        let mut remaining = Vec::new();

        for &(candidate_id, candidate_dist) in candidates_with_distances {
            if result.len() >= m {
                remaining.push(candidate_id);
                continue;
            }

            let mut good = true;
            for &result_id in &result {
                let dist_to_result = self.distance_between_cmp(candidate_id, result_id);
                if dist_to_result < candidate_dist {
                    good = false;
                    break;
                }
            }

            if good {
                result.push(candidate_id);
            } else {
                remaining.push(candidate_id);
            }
        }

        // Fill remaining slots
        for id in remaining {
            if result.len() >= m {
                break;
            }
            result.push(id);
        }

        result
    }

    /// Select neighbors using diversity heuristic (computes distances)
    ///
    /// Used for pruning reverse links where we don't have pre-computed distances.
    fn select_neighbors_heuristic(
        &self,
        candidates: &[u32],
        m: usize,
        query_vector: &[f32],
    ) -> Vec<u32> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        // Sort candidates by distance
        let mut sorted: Vec<_> = candidates
            .iter()
            .map(|&id| (id, self.distance_cmp(query_vector, id)))
            .collect();
        sorted.sort_unstable_by_key(|c| OrderedFloat(c.1));

        // Delegate to the distance-aware version
        self.select_neighbors_heuristic_with_distances(&sorted, m)
    }

    /// Update entry point if new node has higher level
    fn maybe_update_entry_point(&self, node_id: u32, level: u8) {
        loop {
            let current = self.entry_point.load(Ordering::Acquire);
            if current == u32::MAX {
                return;
            }

            let current_level = self.levels[current as usize];
            if level <= current_level {
                return;
            }

            if self
                .entry_point
                .compare_exchange(current, node_id, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                debug!(
                    old_entry = current,
                    new_entry = node_id,
                    old_level = current_level,
                    new_level = level,
                    "Updated entry point"
                );
                return;
            }
        }
    }

    /// Lock-free distance from query to node (uses cached vectors)
    #[inline]
    fn distance_cmp(&self, query: &[f32], id: u32) -> f32 {
        let vec = &self.vectors[id as usize];
        self.distance_fn.distance_for_comparison(query, vec)
    }

    /// Lock-free distance between two nodes (uses cached vectors)
    #[inline]
    fn distance_between_cmp(&self, id_a: u32, id_b: u32) -> f32 {
        let vec_a = &self.vectors[id_a as usize];
        let vec_b = &self.vectors[id_b as usize];
        self.distance_fn.distance_for_comparison(vec_a, vec_b)
    }

    /// Convert builder to HNSWIndex
    fn into_index(self) -> HNSWIndex {
        let entry_point = self.entry_point.load(Ordering::Acquire);

        HNSWIndex {
            storage: self.storage.into_inner(),
            entry_point: if entry_point == u32::MAX {
                None
            } else {
                Some(entry_point)
            },
            params: self.params,
            distance_fn: self.distance_fn,
            rng_state: self.rng_state.load(Ordering::Relaxed),
        }
    }
}

// SAFETY: ParallelBuilder is Sync because:
// 1. `storage` (UnsafeCell<NodeStorage>) - access is synchronized via per-node locks
// 2. `vectors` (Vec<Vec<f32>>) - immutable after allocation, safe to share
// 3. `levels` (Vec<u8>) - immutable after allocation, safe to share
// 4. `node_locks` (Vec<Mutex<()>>) - Mutex is Sync
// 5. `ready_bitmap` (AtomicBitVec) - uses atomics, safe to share
// 6. `entry_point` (AtomicU32) - atomic, safe to share
// 7. `params` (HNSWParams) - immutable, safe to share
// 8. `distance_fn` (DistanceFunction) - immutable, safe to share
// 9. `rng_state` (AtomicU64) - atomic, safe to share
unsafe impl Sync for ParallelBuilder {}

impl HNSWIndex {
    /// Build index from vectors using parallel construction
    ///
    /// This is the fastest way to build an index from a batch of vectors.
    /// Uses parallel graph construction with fine-grained locking.
    ///
    /// # Performance
    /// - Warm start: first 512 vectors inserted sequentially
    /// - Remaining vectors inserted in parallel with lock-based synchronization
    /// - Target: 20,000+ vectors/second on 8 cores
    pub fn build_parallel(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Self> {
        let builder = ParallelBuilder::new(dimensions, params, distance_fn, use_quantization)?;
        builder.build(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::hnsw::types::DistanceFunction;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_parallel_build_small() {
        let vectors = random_vectors(100, 32);
        let params = HNSWParams::default();

        let index =
            HNSWIndex::build_parallel(32, params, DistanceFunction::L2, false, vectors).unwrap();

        assert_eq!(index.len(), 100);
        assert!(index.entry_point().is_some());
    }

    #[test]
    fn test_parallel_build_medium() {
        let vectors = random_vectors(1000, 64);
        let params = HNSWParams::default();

        let index =
            HNSWIndex::build_parallel(64, params, DistanceFunction::L2, false, vectors).unwrap();

        assert_eq!(index.len(), 1000);

        // Test search works
        let query: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let results = index.search(&query, 10, 100).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_parallel_vs_sequential_recall() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..64).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        };

        // Build with parallel
        let parallel_index = HNSWIndex::build_parallel(
            64,
            params.clone(),
            DistanceFunction::L2,
            false,
            vectors.clone(),
        )
        .unwrap();

        // Build with sequential
        let mut sequential_index =
            HNSWIndex::new(64, params.clone(), DistanceFunction::L2, false).unwrap();
        for vec in &vectors {
            sequential_index.insert(vec).unwrap();
        }

        // Compare recall
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..64).map(|_| normal.sample(&mut rng)).collect())
            .collect();

        let k = 10;
        let ef = 100;
        let mut parallel_matches = 0;
        let mut total = 0;

        for query in &queries {
            let parallel_results = parallel_index.search(query, k, ef).unwrap();
            let sequential_results = sequential_index.search(query, k, ef).unwrap();

            let parallel_ids: std::collections::HashSet<_> =
                parallel_results.iter().map(|r| r.id).collect();
            let sequential_ids: std::collections::HashSet<_> =
                sequential_results.iter().map(|r| r.id).collect();

            parallel_matches += parallel_ids.intersection(&sequential_ids).count();
            total += k;
        }

        let recall = parallel_matches as f64 / total as f64;
        eprintln!("Parallel vs Sequential recall: {:.1}%", recall * 100.0);

        assert!(
            recall >= 0.85,
            "Parallel recall too low: {:.1}%",
            recall * 100.0
        );
    }

    #[test]
    #[ignore] // Benchmark test - run with: cargo test --release -- --ignored test_parallel_build_benchmark
    fn test_parallel_build_benchmark() {
        let vectors = random_vectors(10_000, 128);
        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let index =
            HNSWIndex::build_parallel(128, params, DistanceFunction::L2, false, vectors).unwrap();
        let elapsed = start.elapsed();

        let rate = 10_000.0 / elapsed.as_secs_f64();
        eprintln!(
            "Parallel build: 10K vectors in {:.2}s ({:.0} vec/s)",
            elapsed.as_secs_f64(),
            rate
        );

        assert_eq!(index.len(), 10_000);

        assert!(
            rate > 5_000.0,
            "Build too slow: {:.0} vec/s (target: >5000 vec/s)",
            rate
        );
    }

    #[test]
    fn test_parallel_graph_connectivity() {
        use std::collections::{HashSet, VecDeque};

        let vectors = random_vectors(1000, 32);
        let params = HNSWParams::default();

        let index =
            HNSWIndex::build_parallel(32, params, DistanceFunction::L2, false, vectors).unwrap();

        let entry_point = index.entry_point().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(entry_point);
        visited.insert(entry_point);

        while let Some(node_id) = queue.pop_front() {
            let neighbors = index.get_neighbors_level0(node_id);
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        let reachable = visited.len();
        let total = index.len();
        let connectivity = reachable as f64 / total as f64;

        eprintln!(
            "Graph connectivity: {}/{} nodes reachable ({:.1}%)",
            reachable,
            total,
            connectivity * 100.0
        );

        assert!(
            connectivity >= 0.99,
            "Graph not fully connected: {:.1}%",
            connectivity * 100.0
        );
    }
}
