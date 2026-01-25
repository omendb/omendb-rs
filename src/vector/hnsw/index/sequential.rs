//! Sequential HNSW construction - optimized version
//!
//! Zero-overhead sequential builder using `&mut` access.
//! No atomics, no locks, no ready bitmap - just direct memory access.
//!
//! Optimizations over parallel builder:
//! - Uses storage.vector() directly (no separate Vec<Vec<f32>>)
//! - Uses storage.add_neighbor() for in-place updates (no get-modify-set)
//! - Reusable search result buffers (no allocation per search)
//! - Passes distances from search to heuristic (no redundant computation)

// Experimental builder, not yet integrated
#![allow(dead_code)]
#![allow(clippy::unnecessary_wraps)]

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::node_storage::NodeStorage;
use crate::vector::hnsw::query_buffers::VisitedList;
use crate::vector::hnsw::types::{Candidate, DistanceFunction, HNSWParams};
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use tracing::{debug, info};

/// Sequential HNSW builder - optimized for single-threaded performance
pub struct SequentialBuilder {
    /// Node storage (direct mutable access, holds vectors + neighbors)
    storage: NodeStorage,
    /// Node levels (assigned during allocation)
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
    /// Reusable candidate heap (min-heap via Reverse)
    candidates: BinaryHeap<std::cmp::Reverse<Candidate>>,
    /// Reusable working set (max-heap for ef-limited results)
    working: BinaryHeap<Candidate>,
    /// Reusable search results buffer
    search_results: Vec<(u32, f32)>,
    /// Reusable neighbor selection buffers
    heuristic_result: Vec<u32>,
    heuristic_remaining: Vec<u32>,
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
            levels: Vec::new(),
            entry_point: None,
            rng_state: params.seed,
            params,
            distance_fn,
            visited: VisitedList::new(),
            candidates: BinaryHeap::with_capacity(params.ef_construction * 2),
            working: BinaryHeap::with_capacity(params.ef_construction * 2),
            search_results: Vec::with_capacity(params.ef_construction),
            heuristic_result: Vec::with_capacity(params.m * 2),
            heuristic_remaining: Vec::with_capacity(params.ef_construction),
        })
    }

    /// Build index from vectors - optimized sequential construction
    pub fn build(mut self, vectors: Vec<Vec<f32>>) -> Result<HNSWIndex> {
        if vectors.is_empty() {
            return Ok(self.into_index());
        }

        let batch_size = vectors.len();
        info!(
            batch_size,
            "Starting optimized sequential HNSW construction"
        );
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

        // Phase 1: Allocate all nodes and store vectors in NodeStorage
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
            "Optimized sequential construction complete"
        );

        Ok(self.into_index())
    }

    /// Allocate all nodes and assign levels
    fn allocate_all_nodes(&mut self, vectors: &[Vec<f32>]) {
        let n = vectors.len();
        self.levels = Vec::with_capacity(n);

        for vector in vectors {
            let node_id = self.storage.allocate_node();
            let level = self.random_level();

            // Store vector directly in NodeStorage (no separate Vec<Vec<f32>>)
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
        let mut nearest = entry_point;

        // Descend from top level to target level (greedy, ef=1)
        for lc in ((level + 1)..=entry_level).rev() {
            nearest = self.greedy_search(node_id, nearest, lc);
        }

        // Insert at each level from target down to 0
        for lc in (0..=level).rev() {
            let ef = if lc == 0 {
                self.params.ef_construction
            } else {
                self.params.ef_construction.max(self.params.m)
            };

            // Search for candidates with distances (results in self.search_results)
            self.search_layer(node_id, nearest, ef, lc);

            // Select neighbors using heuristic (uses distances from search)
            let m = self.params.m_for_level(lc);
            self.select_neighbors_heuristic(node_id, m);

            // Connect node to selected neighbors using in-place add
            for i in 0..self.heuristic_result.len() {
                let neighbor_id = self.heuristic_result[i];
                self.storage.add_neighbor(node_id, lc, neighbor_id);
            }

            // Add reverse connections (in-place, with pruning if needed)
            for i in 0..self.heuristic_result.len() {
                let neighbor_id = self.heuristic_result[i];
                self.add_reverse_connection(neighbor_id, node_id, lc);
            }

            // Use best neighbor as entry for next level
            if !self.search_results.is_empty() {
                nearest = self.search_results[0].0;
            }
        }

        // Update entry point if this node has higher level
        if level > entry_level {
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    /// Greedy search for single nearest neighbor (ef=1, for upper level descent)
    #[inline]
    fn greedy_search(&mut self, query_id: u32, mut current: u32, level: u8) -> u32 {
        let mut current_dist = self.distance_between(query_id, current);

        loop {
            let neighbors = self.storage.neighbors_at_level_cow(current, level);
            let mut changed = false;

            for &neighbor in neighbors.iter() {
                let dist = self.distance_between(query_id, neighbor);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search layer and store results in self.search_results (sorted by distance)
    fn search_layer(&mut self, query_id: u32, entry: u32, ef: usize, level: u8) {
        self.visited.clear();
        self.candidates.clear();
        self.working.clear();

        // Initialize with entry point
        let entry_dist = self.distance_between(query_id, entry);
        self.candidates
            .push(std::cmp::Reverse(Candidate::new(entry, entry_dist)));
        self.working.push(Candidate::new(entry, entry_dist));
        self.visited.insert(entry);

        // Greedy search
        while let Some(std::cmp::Reverse(current)) = self.candidates.pop() {
            // Check if we can stop (current is farther than worst in working set)
            if let Some(worst) = self.working.peek() {
                if current.distance > worst.distance {
                    break;
                }
            }

            // Get neighbors (zero-copy for level 0, allocates for upper levels)
            let neighbors = self.storage.neighbors_at_level_cow(current.node_id, level);

            for &neighbor_id in neighbors.iter() {
                if !self.visited.contains(neighbor_id) {
                    self.visited.insert(neighbor_id);
                    let dist = self.distance_between(query_id, neighbor_id);

                    // Check if dominated by worst in working set
                    let dominated = self.working.len() >= ef
                        && self.working.peek().is_some_and(|w| dist > w.distance.0);

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

        // Extract results into reusable buffer (sorted by distance)
        self.search_results.clear();
        while let Some(c) = self.working.pop() {
            self.search_results.push((c.node_id, c.distance.0));
        }
        // Reverse to get ascending order (working is max-heap)
        self.search_results.reverse();
    }

    /// Select neighbors using diversity heuristic
    /// Uses distances from self.search_results, stores result in self.heuristic_result
    fn select_neighbors_heuristic(&mut self, _query_id: u32, m: usize) {
        self.heuristic_result.clear();
        self.heuristic_remaining.clear();

        if self.search_results.len() <= m {
            // Just take all candidates
            for &(id, _) in &self.search_results {
                self.heuristic_result.push(id);
            }
            return;
        }

        // Diversity-based selection
        for &(candidate_id, candidate_dist) in &self.search_results {
            if self.heuristic_result.len() >= m {
                self.heuristic_remaining.push(candidate_id);
                continue;
            }

            // Check if candidate is closer to query than to any selected neighbor
            let mut good = true;
            for &result_id in &self.heuristic_result {
                let dist_to_result = self.distance_between(candidate_id, result_id);
                if dist_to_result < candidate_dist {
                    good = false;
                    break;
                }
            }

            if good {
                self.heuristic_result.push(candidate_id);
            } else {
                self.heuristic_remaining.push(candidate_id);
            }
        }

        // Fill remaining slots from discarded candidates
        for &id in &self.heuristic_remaining {
            if self.heuristic_result.len() >= m {
                break;
            }
            self.heuristic_result.push(id);
        }
    }

    /// Add reverse connection with pruning if needed (uses in-place add when possible)
    fn add_reverse_connection(&mut self, from: u32, to: u32, level: u8) {
        let m_max = self.params.m_for_level(level);
        let current_count = self.storage.neighbor_count_at_level(from, level);

        if current_count < m_max {
            // Fast path: just add the neighbor in-place
            self.storage.add_neighbor(from, level, to);
        } else {
            // Slow path: need to prune - get all neighbors, add new one, select best M
            let mut neighbors = self.storage.neighbors_at_level(from, level);
            neighbors.push(to);

            // Select best neighbors using distances from 'from' node
            let pruned = self.select_neighbors_for_node(from, &neighbors, m_max);
            self.storage.set_neighbors_at_level(from, level, pruned);
        }
    }

    /// Select neighbors for a specific node (for reverse connection pruning)
    fn select_neighbors_for_node(&self, node_id: u32, candidates: &[u32], m: usize) -> Vec<u32> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        // Sort candidates by distance to node
        let mut sorted: Vec<_> = candidates
            .iter()
            .map(|&id| (id, self.distance_between(node_id, id)))
            .collect();
        sorted.sort_by_key(|(_, d)| OrderedFloat(*d));

        // Apply diversity heuristic
        let mut result = Vec::with_capacity(m);
        let mut remaining = Vec::new();

        for (candidate_id, candidate_dist) in sorted {
            if result.len() >= m {
                remaining.push(candidate_id);
                continue;
            }

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

    /// Compute distance between two nodes using storage vectors
    #[inline]
    fn distance_between(&self, id_a: u32, id_b: u32) -> f32 {
        let vec_a = self.storage.vector(id_a);
        let vec_b = self.storage.vector(id_b);
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

        // Optimized sequential builder
        let builder =
            SequentialBuilder::new(dim, params.clone(), DistanceFunction::L2, false).unwrap();
        let start = Instant::now();
        let index = builder.build(vectors.clone()).unwrap();
        let seq_elapsed = start.elapsed();
        let seq_rate = n as f64 / seq_elapsed.as_secs_f64();
        assert_eq!(index.len(), n);

        // Parallel builder (for comparison)
        let builder =
            ParallelBuilder::new(dim, params.clone(), DistanceFunction::L2, false).unwrap();
        let start = Instant::now();
        let index = builder.build(vectors).unwrap();
        let par_elapsed = start.elapsed();
        let par_rate = n as f64 / par_elapsed.as_secs_f64();
        assert_eq!(index.len(), n);

        println!("\n=== Build Comparison (10K random, 128D) ===");
        println!(
            "Sequential (opt): {:?} ({:.0} vec/s)",
            seq_elapsed, seq_rate
        );
        println!(
            "Parallel:         {:?} ({:.0} vec/s)",
            par_elapsed, par_rate
        );
        println!("Ratio: {:.2}x", seq_rate / par_rate);
    }

    #[test]
    fn test_sequential_search_quality() {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let n = 1000;
        let dim = 32;
        let k = 10;

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let params = HNSWParams::default();
        let builder = SequentialBuilder::new(dim, params, DistanceFunction::L2, false).unwrap();
        let index = builder.build(vectors.clone()).unwrap();

        // Query with first vector
        let query = &vectors[0];
        let results = index.search(query, k, 100).unwrap();

        // First result should be the query itself (distance ~0)
        assert_eq!(results.len(), k);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.001);
    }
}
