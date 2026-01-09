//! HNSW insertion operations
//!
//! Implements single insert, batch insert, and neighbor selection heuristic.

use super::HNSWIndex;
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::types::HNSWNode;
use ordered_float::OrderedFloat;
use tracing::{debug, error, info, instrument};

impl HNSWIndex {
    /// Validate vector dimensions and values for insertion
    fn validate_insert_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions() {
            error!(
                expected_dim = self.dimensions(),
                actual_dim = vector.len(),
                "Dimension mismatch during insert"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: vector.len(),
            });
        }

        if vector.iter().any(|x| !x.is_finite()) {
            error!("Invalid vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }

        Ok(())
    }

    /// Store vector and create node, returns (node_id, level)
    fn store_and_create_node(&mut self, vector: &[f32]) -> Result<(u32, u8)> {
        let node_id = self.vectors.insert(vector.to_owned()).map_err(|e| {
            error!(error = ?e, "Failed to store vector");
            HNSWError::Storage(e.clone())
        })?;

        let level = self.random_level();
        let node = HNSWNode::new(node_id, level);
        self.nodes.push(node);

        Ok((node_id, level))
    }

    /// Update entry point if new node has higher level
    fn maybe_update_entry_point(&mut self, node_id: u32, level: u8) -> Result<()> {
        let entry_point_id = self
            .entry_point
            .ok_or_else(|| HNSWError::internal("Entry point should exist after first insert"))?;
        let entry_level = self.nodes[entry_point_id as usize].level;

        if level > entry_level {
            self.entry_point = Some(node_id);
            debug!(
                old_entry = entry_point_id,
                new_entry = node_id,
                old_level = entry_level,
                new_level = level,
                "Updated entry point to higher level node"
            );
        }

        Ok(())
    }

    /// Insert a vector into the index
    ///
    /// Returns the node ID assigned to this vector.
    #[instrument(skip(self, vector), fields(dimensions = vector.len(), index_size = self.len()))]
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        self.validate_insert_vector(vector)?;
        let (node_id, level) = self.store_and_create_node(vector)?;

        // If this is the first node, set as entry point and return
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(node_id);
        }

        self.insert_into_graph(node_id, vector, level)?;
        self.maybe_update_entry_point(node_id, level)?;

        debug!(
            node_id = node_id,
            level = level,
            index_size = self.len(),
            "Successfully inserted vector"
        );

        Ok(node_id)
    }

    /// Insert a vector with entry point hints for faster insertion
    ///
    /// Used by graph merging to speed up insertion when we already know
    /// good starting points (neighbors from the source graph).
    ///
    /// # Arguments
    /// * `vector` - Vector to insert
    /// * `entry_hints` - Node IDs to use as starting points (must exist in index)
    /// * `ef` - Expansion factor for search (lower = faster, may reduce quality)
    ///
    /// # Performance
    /// ~5x faster than standard insert when hints are good neighbors
    #[instrument(skip(self, vector, entry_hints), fields(dimensions = vector.len(), hints = entry_hints.len()))]
    pub fn insert_with_hints(
        &mut self,
        vector: &[f32],
        entry_hints: &[u32],
        ef: usize,
    ) -> Result<u32> {
        self.validate_insert_vector(vector)?;
        let (node_id, level) = self.store_and_create_node(vector)?;

        // If this is the first node, set as entry point and return
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(node_id);
        }

        // Filter hints to valid node IDs that exist in the index
        let valid_hints: Vec<u32> = entry_hints
            .iter()
            .filter(|&&id| (id as usize) < self.nodes.len())
            .copied()
            .collect();

        // If no valid hints, fall back to standard insertion
        if valid_hints.is_empty() {
            return self
                .insert_into_graph(node_id, vector, level)
                .map(|()| node_id);
        }

        // Use hints as starting points for graph insertion
        self.insert_into_graph_with_hints(node_id, vector, level, &valid_hints, ef)?;
        self.maybe_update_entry_point(node_id, level)?;

        Ok(node_id)
    }

    /// Insert node into graph using entry hints instead of global entry point
    pub(super) fn insert_into_graph_with_hints(
        &mut self,
        node_id: u32,
        vector: &[f32],
        level: u8,
        entry_hints: &[u32],
        ef: usize,
    ) -> Result<()> {
        // Start search from hints (skip upper layer traversal)
        let mut nearest = entry_hints.to_vec();

        // Insert at levels 0..=level (iterate from top to bottom)
        // Use full precision distances during graph construction for better quality
        for lc in (0..=level).rev() {
            // Find ef nearest neighbors at this level using reduced ef
            let candidates = self.search_layer_full_precision(vector, &nearest, ef, lc)?;

            // Select M best neighbors using heuristic
            let m = if lc == 0 {
                self.params.m * 2
            } else {
                self.params.m
            };

            let neighbors = self.select_neighbors_heuristic(node_id, &candidates, m, lc, vector)?;

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                self.neighbors
                    .add_bidirectional_link(node_id, neighbor_id, lc);
            }

            // Update neighbor counts
            self.nodes[node_id as usize].set_neighbor_count(lc, neighbors.len());

            // Prune overloaded neighbors
            for &neighbor_id in &neighbors {
                let neighbor_neighbors = self.neighbors.get_neighbors(neighbor_id, lc);
                if neighbor_neighbors.len() > m {
                    let neighbor_vec = self
                        .vectors
                        .get_dequantized(neighbor_id)
                        .ok_or(HNSWError::VectorNotFound(neighbor_id))?;
                    let pruned = self.select_neighbors_heuristic(
                        neighbor_id,
                        &neighbor_neighbors,
                        m,
                        lc,
                        &neighbor_vec,
                    )?;
                    self.neighbors
                        .set_neighbors(neighbor_id, lc, pruned.clone());
                    self.nodes[neighbor_id as usize].set_neighbor_count(lc, pruned.len());
                }
            }

            // Update nearest for next level
            nearest = candidates;
        }

        Ok(())
    }

    /// Batch insert multiple vectors with parallel graph construction
    ///
    /// This method achieves 10-50x speedup over incremental insertion by:
    /// 1. Storing all vectors first (no graph construction)
    /// 2. Building the HNSW graph in parallel using RwLock-protected neighbor lists
    ///
    /// # Performance
    /// - Small batches (<100): Use `insert()` for simplicity
    /// - Medium batches (100-10K): 8-12x speedup expected
    /// - Large batches (10K+): 20-50x speedup expected
    ///
    /// # Algorithm
    /// - Pre-allocate all nodes and levels (deterministic)
    /// - Parallel graph construction with thread-safe neighbor updates
    /// - Lock ordering prevents deadlocks
    ///
    /// # Arguments
    /// * `vectors` - Batch of vectors to insert
    ///
    /// # Returns
    /// Vector of node IDs corresponding to inserted vectors
    #[instrument(skip(self, vectors), fields(batch_size = vectors.len()))]
    pub fn batch_insert(&mut self, vectors: Vec<Vec<f32>>) -> Result<Vec<u32>> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU32, Ordering};

        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = vectors.len();
        info!(batch_size, "Starting parallel batch insertion");

        // Parallel validation (fast, no graph modifications)
        let dimensions = self.dimensions();
        let validation_start = std::time::Instant::now();

        vectors.par_iter().try_for_each(|vec| -> Result<()> {
            if vec.len() != dimensions {
                return Err(HNSWError::DimensionMismatch {
                    expected: dimensions,
                    actual: vec.len(),
                });
            }
            if vec.iter().any(|x| !x.is_finite()) {
                return Err(HNSWError::InvalidVector);
            }
            Ok(())
        })?;

        debug!(
            duration_ms = validation_start.elapsed().as_millis(),
            "Parallel validation complete"
        );

        // Phase 1: Store all vectors and create nodes (fast, sequential)
        let storage_start = std::time::Instant::now();
        let mut node_ids = Vec::with_capacity(batch_size);
        let mut new_nodes = Vec::with_capacity(batch_size);

        // Track highest level node in this batch for entry point update AFTER graph construction
        let mut highest_level_node: Option<(u32, u8)> = None;

        for vector in vectors {
            // Store vector
            let node_id = self.vectors.insert(vector).map_err(|e| {
                error!(error = ?e, "Failed to store vector");
                HNSWError::Storage(e.clone())
            })?;

            // Assign level (deterministic from RNG state)
            let level = self.random_level();

            // Create node
            let node = HNSWNode::new(node_id, level);
            new_nodes.push(node);
            node_ids.push(node_id);

            // Track highest level node (entry point update deferred until after graph construction)
            if self.entry_point.is_none() {
                // First node ever - set entry point immediately
                self.entry_point = Some(node_id);
                highest_level_node = Some((node_id, level));
            } else {
                // Track highest level for later update
                match highest_level_node {
                    None => highest_level_node = Some((node_id, level)),
                    Some((_, prev_level)) if level > prev_level => {
                        highest_level_node = Some((node_id, level));
                    }
                    _ => {}
                }
            }
        }

        // Add new nodes to index
        self.nodes.extend(new_nodes);

        debug!(
            duration_ms = storage_start.elapsed().as_millis(),
            nodes_added = node_ids.len(),
            "Vector storage complete"
        );

        // Pre-allocate neighbor storage for all new nodes (required for parallel access)
        for &node_id in &node_ids {
            // Pre-allocate empty neighbor lists for all levels
            for level in 0..self.params.max_level {
                self.neighbors.set_neighbors(node_id, level, Vec::new());
            }
        }

        // Phase 2: Build graph in parallel (the key optimization!)
        let graph_start = std::time::Instant::now();

        // If this is the only node, no graph to build
        if self.nodes.len() == 1 {
            info!("Single node, no graph construction needed");
            return Ok(node_ids);
        }

        // Parallel graph construction
        // Note: We need to handle the case where we're building incrementally
        // (adding to existing graph) vs building from scratch
        let nodes_to_insert: Vec<(u32, u8)> = node_ids
            .iter()
            .map(|&id| {
                let level = self.nodes[id as usize].level;
                (id, level)
            })
            .collect();

        // Use atomic counter for progress tracking
        let progress_counter = AtomicU32::new(0);
        let progress_interval = if batch_size >= 1000 {
            batch_size / 10
        } else {
            batch_size
        };

        // Parallel insertion into graph
        let result: Result<()> = nodes_to_insert.par_iter().try_for_each(|(node_id, level)| {
            // Get vector for this node
            // Use get_dequantized for SQ8 support (get() returns None for trained SQ8)
            let vector = self
                .vectors
                .get_dequantized(*node_id)
                .ok_or(HNSWError::VectorNotFound(*node_id))?;

            // Build graph connections for all nodes (including node_id=0)
            // During batch insertion into empty index, search_layer may return limited
            // results since the graph is sparse, but connections will still be made
            let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
            let entry_level = self.nodes[entry_point as usize].level;

            // Search for nearest neighbors at each level above target level
            // Use full precision distances during graph construction for better quality
            let mut nearest = vec![entry_point];
            for lc in ((*level + 1)..=entry_level).rev() {
                nearest = self.search_layer_full_precision(&vector, &nearest, 1, lc)?;
            }

            // Insert at levels 0..=level
            for lc in (0..=*level).rev() {
                // Find ef_construction nearest neighbors at this level
                let candidates = self.search_layer_full_precision(
                    &vector,
                    &nearest,
                    self.params.ef_construction,
                    lc,
                )?;

                // Select M best neighbors using heuristic
                let m = if lc == 0 {
                    self.params.m * 2
                } else {
                    self.params.m
                };

                let neighbors =
                    self.select_neighbors_heuristic(*node_id, &candidates, m, lc, &vector)?;

                // Add bidirectional links (thread-safe via RwLock parallel methods)
                for &neighbor_id in &neighbors {
                    self.neighbors
                        .add_bidirectional_link_parallel(*node_id, neighbor_id, lc);
                }

                // NOTE: Pruning is deferred to after parallel loop for performance
                // This allows the parallel phase to only add links (fast, less contention)
                // Pruning happens in a single pass after all insertions complete

                // Update nearest for next level
                nearest = candidates;
            }

            // Progress tracking
            let count = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(progress_interval as u32) {
                let elapsed = graph_start.elapsed().as_secs_f64();
                let rate = count as f64 / elapsed;
                info!(
                    progress = count,
                    total = batch_size,
                    percent = (count as usize * 100) / batch_size,
                    rate_vec_per_sec = rate as u64,
                    "Parallel graph construction progress"
                );
            }

            Ok(())
        });

        result?;

        // Update entry point AFTER graph construction (critical for incremental inserts)
        // Only update if a new node has a higher level than current entry point
        if let Some((new_entry, new_level)) = highest_level_node {
            if let Some(current_entry) = self.entry_point {
                let current_level = self.nodes[current_entry as usize].level;
                if new_level > current_level {
                    self.entry_point = Some(new_entry);
                }
            }
        }

        // Phase 3: Prune over-connected nodes to restore search performance
        // During parallel insertion, nodes accumulate many neighbors (unbounded).
        // Without pruning, search degrades from O(M) to O(N) distance calcs per hop.
        // See: HNSW paper (Malkov 2018) SELECT-NEIGHBORS-HEURISTIC, Qdrant PR #2869
        let prune_start = std::time::Instant::now();
        let mut pruned_count = 0u32;

        // Prune all nodes in the graph (not just newly inserted ones)
        // because bidirectional links may have over-connected existing nodes
        let max_node_id = self.nodes.len() as u32;
        for node_id in 0..max_node_id {
            let level = self.nodes[node_id as usize].level;
            for lc in 0..=level {
                let m = if lc == 0 {
                    self.params.m * 2
                } else {
                    self.params.m
                };

                let neighbors = self.neighbors.get_neighbors(node_id, lc);

                if neighbors.len() > m {
                    let vector = match self.vectors.get(node_id) {
                        Some(v) => v,
                        None => continue,
                    };

                    let pruned = match self
                        .select_neighbors_heuristic(node_id, &neighbors, m, lc, &vector)
                    {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    // Update neighbor list (mutable borrow is safe here - not parallel)
                    self.neighbors.set_neighbors(node_id, lc, pruned.clone());
                    self.nodes[node_id as usize].set_neighbor_count(lc, pruned.len());
                    pruned_count += 1;
                }
            }
        }

        let prune_time = prune_start.elapsed().as_secs_f64();
        let total_time = graph_start.elapsed().as_secs_f64();
        let final_rate = batch_size as f64 / total_time;

        info!(
            inserted = node_ids.len(),
            pruned = pruned_count,
            prune_secs = prune_time,
            duration_secs = total_time,
            rate_vec_per_sec = final_rate as u64,
            "Parallel batch insertion complete"
        );

        Ok(node_ids)
    }

    /// Insert node into graph structure
    ///
    /// Implements HNSW insertion algorithm (Malkov & Yashunin 2018)
    pub(super) fn insert_into_graph(
        &mut self,
        node_id: u32,
        vector: &[f32],
        level: u8,
    ) -> Result<()> {
        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Search for nearest neighbors at each level above target level
        // Use full precision distances during graph construction for better quality
        let mut nearest = vec![entry_point];
        for lc in ((level + 1)..=entry_level).rev() {
            nearest = self.search_layer_full_precision(vector, &nearest, 1, lc)?;
        }

        // Insert at levels 0..=level (iterate from top to bottom)
        for lc in (0..=level).rev() {
            // Find ef_construction nearest neighbors at this level
            let candidates = self.search_layer_full_precision(
                vector,
                &nearest,
                self.params.ef_construction,
                lc,
            )?;

            // Select M best neighbors using heuristic
            let m = if lc == 0 {
                self.params.m * 2 // Level 0 has more connections
            } else {
                self.params.m
            };

            let neighbors = self.select_neighbors_heuristic(node_id, &candidates, m, lc, vector)?;

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                self.neighbors
                    .add_bidirectional_link(node_id, neighbor_id, lc);
            }

            // Update neighbor counts
            self.nodes[node_id as usize].set_neighbor_count(lc, neighbors.len());

            // Prune neighbors' connections if they exceed M
            for &neighbor_id in &neighbors {
                let neighbor_neighbors = self.neighbors.get_neighbors(neighbor_id, lc);
                if neighbor_neighbors.len() > m {
                    let neighbor_vec = self
                        .vectors
                        .get_dequantized(neighbor_id)
                        .ok_or(HNSWError::VectorNotFound(neighbor_id))?;
                    let pruned = self.select_neighbors_heuristic(
                        neighbor_id,
                        &neighbor_neighbors,
                        m,
                        lc,
                        &neighbor_vec,
                    )?;

                    // Clear and reset neighbors
                    self.neighbors
                        .set_neighbors(neighbor_id, lc, pruned.clone());
                    self.nodes[neighbor_id as usize].set_neighbor_count(lc, pruned.len());
                }
            }

            // Update nearest for next level
            nearest = candidates;
        }

        Ok(())
    }

    /// Select neighbors using heuristic (diverse neighbors, better recall)
    ///
    /// Algorithm from Malkov 2018, Section 4
    pub(super) fn select_neighbors_heuristic(
        &self,
        _node_id: u32,
        candidates: &[u32],
        m: usize,
        _level: u8,
        query_vector: &[f32],
    ) -> Result<Vec<u32>> {
        if candidates.len() <= m {
            return Ok(candidates.to_vec());
        }
        // Sort candidates by distance to query
        let mut sorted_candidates: Vec<_> = candidates
            .iter()
            .map(|&id| {
                let dist = self.distance_cmp(query_vector, id)?;
                Ok((id, dist))
            })
            .collect::<Result<Vec<_>>>()?;
        sorted_candidates.sort_unstable_by_key(|c| OrderedFloat(c.1));

        let mut result = Vec::with_capacity(m);
        let mut remaining = Vec::new();

        // Heuristic: Select diverse neighbors
        for (candidate_id, candidate_dist) in &sorted_candidates {
            if result.len() >= m {
                remaining.push(*candidate_id);
                continue;
            }

            // Check if candidate is closer to query than to existing neighbors
            let mut good = true;
            for &result_id in &result {
                let dist_to_result = self.distance_between_cmp(*candidate_id, result_id)?;
                if dist_to_result < *candidate_dist {
                    good = false;
                    break;
                }
            }

            if good {
                result.push(*candidate_id);
            } else {
                remaining.push(*candidate_id);
            }
        }

        // Fill remaining slots with closest candidates if needed
        for candidate_id in remaining {
            if result.len() >= m {
                break;
            }
            result.push(candidate_id);
        }

        Ok(result)
    }
}
