//! Sequential HNSW construction
//!
//! Zero-overhead sequential builder using `&mut` access.
//! No atomics, no locks, no ready bitmap - just direct memory access.
//!
//! Use this for:
//! - Single-threaded builds
//! - Small batches (< 1K vectors)
//! - When you want maximum single-threaded performance

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::query_buffers::VisitedList;
use crate::vector::hnsw::types::{Candidate, DistanceFunction, HNSWParams};
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use tracing::{debug, info};

/// Sequential HNSW builder - zero synchronization overhead
pub struct SequentialBuilder {
    /// Node storage (direct mutable access)
    storage: NodeStorage,
    /// Cached vectors for distance computation
    vectors: Vec<Vec<f32>>,
    /// Node levels
    levels: Vec<u8>,
    /// Entry point (None = empty, Some(id) = entry node)
    entry_point: Option<u32>,
    /// Construction parameters
    params: HNSWParams,
    /// Distance function
    distance_fn: DistanceFunction,
    /// RNG state for level assignment
    rng_state: u64,
    /// Reusable visited list
    visited: VisitedList,
    /// Reusable candidate heap
    candidates: BinaryHeap<std::cmp::Reverse<Candidate>>,
    /// Reusable working set
    working: BinaryHeap<Candidate>,
}

impl SequentialBuilder {
    /// Create a new sequential builder
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
            storage,
            vectors: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            rng_state: params.seed,
            params,
            distance_fn,
            visited: VisitedList::new(),
            candidates: BinaryHeap::new(),
            working: BinaryHeap::new(),
        })
    }

    /// Build index from vectors - zero synchronization overhead
    pub fn build(mut self, vectors: Vec<Vec<f32>>) -> Result<HNSWIndex> {
        if vectors.is_empty() {
            return Ok(self.into_index());
        }

        let batch_size = vectors.len();
        info!(batch_size, "Starting sequential HNSW construction");
        let start = std::time::Instant::now();

        // Validate all vectors
        let dimensions = self.storage.dimensions();
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

        // Phase 1: Allocate all nodes
        self.allocate_all_nodes(&vectors);
        debug!(nodes = batch_size, "Allocated all nodes");

        // Phase 2: Insert all nodes sequentially
        for node_id in 0..batch_size as u32 {
            self.insert_node(node_id)?;
        }

        let elapsed = start.elapsed();
        let rate = batch_size as f64 / elapsed.as_secs_f64();
        info!(
            batch_size,
            elapsed_secs = elapsed.as_secs_f64(),
            rate_vec_per_sec = rate as u64,
            "Sequential construction complete"
        );

        Ok(self.into_index())
    }

    /// Allocate all nodes and assign levels
    fn allocate_all_nodes(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        self.vectors = vectors.to_vec();
        self.levels = Vec::with_capacity(n);

        for vector in vectors {
            let node_id = self.storage.allocate_node();
            let level = self.random_level();

            self.storage.set_vector(node_id, vector);
            self.storage.set_slot(node_id, node_id);
            self.storage.set_level(node_id, level);

            if level > 0 {
                self.storage.allocate_upper_levels(node_id, level);
            }

            self.levels.push(level);
        }
    }

    /// Insert a single node into the graph
    fn insert_node(&mut self, node_id: u32) -> Result<()> {
        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(());
        }

        let level = self.levels[node_id as usize];
        let entry_point = self.entry_point.unwrap();
        let entry_level = self.levels[entry_point as usize];

        // Find nearest neighbors starting from entry point
        let mut nearest = vec![entry_point];

        // Descend from top level to target level
        for lc in ((level + 1)..=entry_level).rev() {
            nearest = self.search_layer_by_id(node_id, &nearest, 1, lc);
        }

        // Insert at each level from target down to 0
        for lc in (0..=level).rev() {
            let ef = if lc == 0 {
                self.params.ef_construction
            } else {
                self.params.ef_construction.max(self.params.m)
            };

            // Search for candidates with distances
            let candidates = self.search_layer_with_distances_by_id(node_id, &nearest, ef, lc);

            // Select neighbors using heuristic
            let m = self.params.m_for_level(lc);
            let neighbors = self.select_neighbors_heuristic(&candidates, m);

            // Connect node to neighbors (direct write, no locks)
            self.storage
                .set_neighbors_at_level(node_id, lc, neighbors.clone());

            // Add reverse connections
            for &neighbor_id in &neighbors {
                self.add_reverse_connection(neighbor_id, node_id, lc);
            }

            // Use best neighbor as entry for next level
            if !candidates.is_empty() {
                nearest = vec![candidates[0].0];
            }
        }

        // Update entry point if this node has higher level
        if level > entry_level {
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Add reverse connection with pruning if needed
    fn add_reverse_connection(&mut self, from: u32, to: u32, level: u8) {
        let m_max = self.params.m_for_level(level);
        let mut neighbors = self.storage.neighbors_at_level(from, level);

        if neighbors.len() < m_max {
            neighbors.push(to);
            self.storage.set_neighbors_at_level(from, level, neighbors);
        } else {
            // Need to prune - add new neighbor and select best M
            neighbors.push(to);
            let pruned = self.select_neighbors_heuristic_for_node_by_id(from, &neighbors, m_max);
            self.storage.set_neighbors_at_level(from, level, pruned);
        }
    }

    /// Search layer by query node ID and return node IDs (avoids vector clone)
    fn search_layer_by_id(
        &mut self,
        query_id: u32,
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Vec<u32> {
        self.search_layer_with_distances_by_id(query_id, entry_points, ef, level)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Search layer by query node ID and return (node_id, distance) pairs sorted by distance
    fn search_layer_with_distances_by_id(
        &mut self,
        query_id: u32,
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Vec<(u32, f32)> {
        self.visited.clear();
        self.candidates.clear();
        self.working.clear();

        // Initialize with entry points
        for &ep in entry_points {
            let dist = self.distance_between(query_id, ep);
            self.candidates
                .push(std::cmp::Reverse(Candidate::new(ep, dist)));
            self.working.push(Candidate::new(ep, dist));
            self.visited.insert(ep);
        }

        // Greedy search
        while let Some(std::cmp::Reverse(current)) = self.candidates.pop() {
            // Check if we can stop
            if let Some(worst) = self.working.peek() {
                if current.distance > worst.distance {
                    break;
                }
            }

            // Explore neighbors
            let neighbors = self.storage.neighbors_at_level(current.node_id, level);
            for &neighbor_id in &neighbors {
                if !self.visited.contains(neighbor_id) {
                    self.visited.insert(neighbor_id);
                    let dist = self.distance_between(query_id, neighbor_id);

                    let dominated = self.working.len() >= ef
                        && self.working.peek().map_or(false, |w| dist > w.distance.0);

                    if !dominated {
                        self.candidates
                            .push(std::cmp::Reverse(Candidate::new(neighbor_id, dist)));
                        self.working.push(Candidate::new(neighbor_id, dist));

                        if self.working.len() > ef {
                            self.working.pop();
                        }
                    }
                }
            }
        }

        // Extract results sorted by distance
        let mut results: Vec<_> = self
            .working
            .drain()
            .map(|c| (c.node_id, c.distance.0))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Select neighbors using diversity heuristic (with pre-computed distances)
    fn select_neighbors_heuristic(&self, candidates: &[(u32, f32)], m: usize) -> Vec<u32> {
        if candidates.len() <= m {
            return candidates.iter().map(|(id, _)| *id).collect();
        }

        let mut result = Vec::with_capacity(m);
        let mut remaining = Vec::new();

        for &(candidate_id, candidate_dist) in candidates {
            if result.len() >= m {
                remaining.push(candidate_id);
                continue;
            }

            // Check if candidate is closer to query than to any selected neighbor
            let mut good = true;
            for &result_id in &result {
                let dist_to_result = self.distance_between(candidate_id, result_id);
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

    /// Select neighbors for a specific node by ID (computes distances)
    fn select_neighbors_heuristic_for_node_by_id(
        &self,
        node_id: u32,
        candidates: &[u32],
        m: usize,
    ) -> Vec<u32> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        // Sort candidates by distance to node
        let mut sorted: Vec<_> = candidates
            .iter()
            .map(|&id| (id, self.distance_between(node_id, id)))
            .collect();
        sorted.sort_by_key(|(_, d)| OrderedFloat(*d));

        self.select_neighbors_heuristic(&sorted, m)
    }

    /// Compute distance between two nodes
    #[inline]
    fn distance_between(&self, id_a: u32, id_b: u32) -> f32 {
        let vec_a = &self.vectors[id_a as usize];
        let vec_b = &self.vectors[id_b as usize];
        self.distance_fn.distance_for_comparison(vec_a, vec_b)
    }

    /// Generate random level for a node
    fn random_level(&mut self) -> u8 {
        // Simple xorshift64
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        // Geometric distribution with p = 1/m_l
        let ml = self.params.ml as f64;
        let r = (x as f64) / (u64::MAX as f64);
        let level = (-r.ln() * ml).floor() as u8;
        level.min(self.params.max_level)
    }

    /// Convert builder to HNSWIndex
    fn into_index(self) -> HNSWIndex {
        HNSWIndex {
            storage: self.storage,
            entry_point: self.entry_point,
            rng_state: self.rng_state,
            params: self.params,
            distance_fn: self.distance_fn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_build_small() {
        let params = HNSWParams::default();
        let builder = SequentialBuilder::new(4, params, DistanceFunction::L2, false).unwrap();

        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let index = builder.build(vectors).unwrap();
        assert_eq!(index.len(), 4);
    }

    #[test]
    fn test_sequential_build_empty() {
        let params = HNSWParams::default();
        let builder = SequentialBuilder::new(4, params, DistanceFunction::L2, false).unwrap();
        let index = builder.build(vec![]).unwrap();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_sequential_vs_parallel_10k() {
        use super::super::ParallelBuilder;
        use rand::Rng;
        use std::time::Instant;

        let mut rng = rand::thread_rng();
        let n = 10_000;
        let dim = 128;

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let params = HNSWParams::default();

        // Sequential builder
        let builder =
            SequentialBuilder::new(dim, params.clone(), DistanceFunction::L2, false).unwrap();
        let start = Instant::now();
        let index = builder.build(vectors.clone()).unwrap();
        let seq_elapsed = start.elapsed();
        let seq_rate = n as f64 / seq_elapsed.as_secs_f64();
        assert_eq!(index.len(), n);

        // Parallel builder
        let builder =
            ParallelBuilder::new(dim, params.clone(), DistanceFunction::L2, false).unwrap();
        let start = Instant::now();
        let index = builder.build(vectors).unwrap();
        let par_elapsed = start.elapsed();
        let par_rate = n as f64 / par_elapsed.as_secs_f64();
        assert_eq!(index.len(), n);

        println!("\n=== Build Comparison (10K random, 128D) ===");
        println!("Sequential: {:?} ({:.0} vec/s)", seq_elapsed, seq_rate);
        println!("Parallel:   {:?} ({:.0} vec/s)", par_elapsed, par_rate);
        println!("Ratio: {:.2}x", seq_rate / par_rate);
    }
}
