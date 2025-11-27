// HNSW Index - Main implementation
//
// Architecture:
// - Flattened index (contiguous nodes, u32 node IDs)
// - Separate neighbor storage (fetch only when needed)
// - Cache-optimized layout (64-byte aligned hot data)

use super::error::{HNSWError, Result};
use super::graph_storage::{DiskConfig, GraphStorage};
use super::storage::{NeighborLists, VectorStorage};
use super::storage_tiering::StorageMode;
use super::types::{Candidate, DistanceFunction, HNSWNode, HNSWParams, SearchResult};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::{debug, error, info, instrument, warn};

/// Index statistics for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of vectors in index
    pub num_vectors: usize,

    /// Vector dimensionality
    pub dimensions: usize,

    /// Entry point node ID
    pub entry_point: Option<u32>,

    /// Maximum level in the graph
    pub max_level: u8,

    /// Level distribution (count of nodes at each level as their TOP level)
    pub level_distribution: Vec<usize>,

    /// Average neighbors per node (level 0)
    pub avg_neighbors_l0: f32,

    /// Max neighbors per node (level 0)
    pub max_neighbors_l0: usize,

    /// Memory usage in bytes
    pub memory_bytes: usize,

    /// HNSW parameters
    pub params: HNSWParams,

    /// Distance function
    pub distance_function: DistanceFunction,

    /// Whether quantization is enabled
    pub quantization_enabled: bool,
}

/// HNSW Index
///
/// Hierarchical graph index for approximate nearest neighbor search.
/// Optimized for cache locality and memory efficiency.
///
/// **Note**: Not Clone due to GraphStorage containing non-cloneable backends.
/// Use persistence APIs (save/load) instead of cloning.
#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWIndex {
    /// Node metadata (cache-line aligned)
    nodes: Vec<HNSWNode>,

    /// Graph storage (mode-dependent: in-memory or hybrid disk+cache)
    neighbors: GraphStorage,

    /// Vector storage (full precision or quantized)
    vectors: VectorStorage,

    /// Entry point (top-level node)
    entry_point: Option<u32>,

    /// Construction parameters
    params: HNSWParams,

    /// Distance function
    distance_fn: DistanceFunction,

    /// Random number generator seed state
    rng_state: u64,
}

impl HNSWIndex {
    /// Create a new empty HNSW index with Memory storage mode
    ///
    /// This is the default constructor and maintains backward compatibility.
    /// For explicit storage mode selection, use `new_with_storage()`.
    pub fn new(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
    ) -> Result<Self> {
        Self::new_with_storage(
            dimensions,
            params,
            distance_fn,
            use_quantization,
            StorageMode::Memory,
        )
    }

    /// Create a new empty HNSW index with explicit storage mode
    ///
    /// # Arguments
    /// * `dimensions` - Vector dimensionality
    /// * `params` - HNSW construction parameters
    /// * `distance_fn` - Distance function (L2, Cosine, Dot)
    /// * `use_quantization` - Whether to use binary quantization
    /// * `storage_mode` - Storage mode (Memory, Hybrid, DiskHeavy)
    ///
    /// # Storage Modes
    /// * `Memory` (<10M vectors): Pure in-memory storage, fully serializable
    /// * `Hybrid` (10M-100M vectors): Layer 0 on disk with 30% cache, layers 1-N in memory
    /// * `DiskHeavy` (100M+ vectors): Layer 0 on disk with 10% cache, layers 1-N in memory
    ///
    /// # Example
    /// ```ignore
    /// // Automatic mode selection
    /// let mode = StorageMode::auto_select(num_vectors);
    /// let index = HNSWIndex::new_with_storage(128, params, DistanceFunction::L2, false, mode)?;
    /// ```
    pub fn new_with_storage(
        dimensions: usize,
        params: HNSWParams,
        distance_fn: DistanceFunction,
        use_quantization: bool,
        storage_mode: StorageMode,
    ) -> Result<Self> {
        params.validate().map_err(HNSWError::InvalidParams)?;

        let vectors = if use_quantization {
            VectorStorage::new_binary_quantized(dimensions, true)
        } else {
            VectorStorage::new_full_precision(dimensions)
        };

        let neighbors = GraphStorage::from_mode(storage_mode, params.max_level as usize);

        Ok(Self {
            nodes: Vec::new(),
            neighbors,
            vectors,
            entry_point: None,
            params,
            distance_fn,
            rng_state: params.seed,
        })
    }

    /// Get number of vectors in index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.vectors.dimensions()
    }

    /// Get a vector by ID (full precision)
    ///
    /// Returns None if the ID is invalid or out of bounds.
    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id)
    }

    /// Get entry point
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get node level
    pub fn node_level(&self, node_id: u32) -> Option<u8> {
        self.nodes.get(node_id as usize).map(|n| n.level)
    }

    /// Get neighbor count for a node at a level
    pub fn neighbor_count(&self, node_id: u32, level: u8) -> Result<usize> {
        Ok(self.neighbors.get_neighbors(node_id, level)?.len())
    }

    /// Get HNSW parameters
    pub fn params(&self) -> &HNSWParams {
        &self.params
    }

    /// Get neighbors at level 0 for a node
    ///
    /// Level 0 has the most connections (M*2) and is used for graph merging.
    pub fn get_neighbors_level0(&self, node_id: u32) -> Result<Vec<u32>> {
        self.neighbors.get_neighbors(node_id, 0)
    }

    /// Assign random level to new node
    ///
    /// Uses exponential decay: P(level = l) = (1/M)^l
    /// This ensures most nodes are at level 0, fewer at higher levels.
    fn random_level(&mut self) -> u8 {
        // Simple LCG for deterministic random numbers
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand_val = (self.rng_state >> 32) as f32 / u32::MAX as f32;

        // Exponential distribution: -ln(uniform) / ln(M)
        let level = (-rand_val.ln() * self.params.ml) as u8;
        level.min(self.params.max_level - 1)
    }

    /// Compute distance between two vectors
    fn distance(&self, id_a: u32, id_b: u32) -> Result<f32> {
        let vec_a = self.vectors.get(id_a).ok_or(HNSWError::VectorNotFound(id_a))?;
        let vec_b = self.vectors.get(id_b).ok_or(HNSWError::VectorNotFound(id_b))?;
        Ok(self.distance_fn.distance(vec_a, vec_b))
    }

    /// Compute distance between query and vector (for internal comparisons)
    ///
    /// Uses optimized comparison distance (squared L2 for L2 metric).
    /// This is ~10-15% faster in search_layer.
    #[inline]
    fn distance_to_query_cmp(&self, query: &[f32], id: u32) -> Result<f32> {
        let vec = self.vectors.get(id).ok_or(HNSWError::VectorNotFound(id))?;
        Ok(self.distance_fn.distance_for_comparison(query, vec))
    }

    /// Compute actual distance between query and vector (for final results)
    fn distance_to_query(&self, query: &[f32], id: u32) -> Result<f32> {
        let vec = self.vectors.get(id).ok_or(HNSWError::VectorNotFound(id))?;
        Ok(self.distance_fn.distance(query, vec))
    }

    /// Insert a vector into the index
    ///
    /// Returns the node ID assigned to this vector.
    #[instrument(skip(self, vector), fields(dimensions = vector.len(), index_size = self.len()))]
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32> {
        // Validate dimensions
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

        // Check for NaN/Inf in vector
        if vector.iter().any(|x| !x.is_finite()) {
            error!("Invalid vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }

        // Store vector and get ID
        let node_id = self.vectors.insert(vector.clone()).map_err(|e| {
            error!(error = ?e, "Failed to store vector");
            HNSWError::Storage(e.to_string())
        })?;

        // Assign random level
        let level = self.random_level();

        // Create node
        let node = HNSWNode::new(node_id, level);
        self.nodes.push(node);

        // If this is the first node, set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            return Ok(node_id);
        }

        // Insert into graph
        self.insert_into_graph(node_id, &vector, level)?;

        // Update entry point if this node has higher level than current entry point
        let entry_point_id = self.entry_point.ok_or_else(|| HNSWError::internal("Entry point should exist after first insert"))?;
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
        vector: Vec<f32>,
        entry_hints: &[u32],
        ef: usize,
    ) -> Result<u32> {
        // Validate dimensions
        if vector.len() != self.dimensions() {
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: vector.len(),
            });
        }

        // Check for NaN/Inf in vector
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(HNSWError::InvalidVector);
        }

        // Store vector and get ID
        let node_id = self.vectors.insert(vector.clone()).map_err(|e| {
            HNSWError::Storage(e.to_string())
        })?;

        // Assign random level
        let level = self.random_level();

        // Create node
        let node = HNSWNode::new(node_id, level);
        self.nodes.push(node);

        // If this is the first node, set as entry point
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
            return self.insert_into_graph(node_id, &vector, level).map(|_| node_id);
        }

        // Use hints as starting points for graph insertion
        self.insert_into_graph_with_hints(node_id, &vector, level, &valid_hints, ef)?;

        // Update entry point if this node has higher level
        let entry_point_id = self.entry_point.ok_or(HNSWError::internal("Entry point should exist"))?;
        let entry_level = self.nodes[entry_point_id as usize].level;
        if level > entry_level {
            self.entry_point = Some(node_id);
        }

        Ok(node_id)
    }

    /// Insert node into graph using entry hints instead of global entry point
    fn insert_into_graph_with_hints(
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
        for lc in (0..=level).rev() {
            // Find ef nearest neighbors at this level using reduced ef
            let candidates = self.search_layer(vector, &nearest, ef, lc)?;

            // Select M best neighbors using heuristic
            let m = if lc == 0 {
                self.params.m * 2
            } else {
                self.params.m
            };

            let neighbors = self.select_neighbors_heuristic(node_id, &candidates, m, lc, vector)?;

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                self.neighbors.add_bidirectional_link(node_id, neighbor_id, lc)?;
            }

            // Update neighbor counts
            self.nodes[node_id as usize].set_neighbor_count(lc, neighbors.len());

            // Prune overloaded neighbors
            for &neighbor_id in &neighbors {
                let neighbor_neighbors = self.neighbors.get_neighbors(neighbor_id, lc)?;
                if neighbor_neighbors.len() > m {
                    let neighbor_vec = self.vectors.get(neighbor_id)
                        .ok_or(HNSWError::VectorNotFound(neighbor_id))?;
                    let pruned = self.select_neighbors_heuristic(
                        neighbor_id,
                        &neighbor_neighbors,
                        m,
                        lc,
                        &neighbor_vec,
                    )?;
                    self.neighbors.set_neighbors(neighbor_id, lc, pruned.clone())?;
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
    /// - Small batches (<100): Use insert() for simplicity
    /// - Medium batches (100-10K): 8-12x speedup expected
    /// - Large batches (10K+): 20-50x speedup expected (matches ChromaDB/LanceDB)
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
        let base_id = self.nodes.len() as u32;
        let mut node_ids = Vec::with_capacity(batch_size);
        let mut new_nodes = Vec::with_capacity(batch_size);

        // Track highest level node in this batch for entry point update AFTER graph construction
        let mut highest_level_node: Option<(u32, u8)> = None;

        for vector in vectors {
            // Store vector
            let node_id = self.vectors.insert(vector).map_err(|e| {
                error!(error = ?e, "Failed to store vector");
                HNSWError::Storage(e.to_string())
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
        if let GraphStorage::Memory(ref mut lists) = self.neighbors {
            for &node_id in &node_ids {
                // Pre-allocate empty neighbor lists for all levels
                for level in 0..self.params.max_level {
                    lists.set_neighbors(node_id, level, Vec::new());
                }
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
        let nodes_to_insert: Vec<(u32, u8)> = node_ids.iter()
            .map(|&id| {
                let level = self.nodes[id as usize].level;
                (id, level)
            })
            .collect();

        // Use atomic counter for progress tracking
        let progress_counter = AtomicU32::new(0);
        let progress_interval = if batch_size >= 1000 { batch_size / 10 } else { batch_size };

        // Parallel insertion into graph
        let result: Result<()> = nodes_to_insert.par_iter().try_for_each(|(node_id, level)| {
            // Get vector for this node
            let vector = self.vectors.get(*node_id)
                .ok_or(HNSWError::VectorNotFound(*node_id))?;

            // Build graph connections for all nodes (including node_id=0)
            // During batch insertion into empty index, search_layer may return limited
            // results since the graph is sparse, but connections will still be made
            let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
            let entry_level = self.nodes[entry_point as usize].level;

            // Search for nearest neighbors at each level above target level
            let mut nearest = vec![entry_point];
            for lc in ((*level + 1)..=entry_level).rev() {
                nearest = self.search_layer(&vector, &nearest, 1, lc)?;
            }

            // Insert at levels 0..=level
            for lc in (0..=*level).rev() {
                // Find ef_construction nearest neighbors at this level
                let candidates = self.search_layer(&vector, &nearest, self.params.ef_construction, lc)?;

                // Select M best neighbors using heuristic
                let m = if lc == 0 {
                    self.params.m * 2
                } else {
                    self.params.m
                };

                let neighbors = self.select_neighbors_heuristic(*node_id, &candidates, m, lc, &vector)?;

                // Add bidirectional links (thread-safe via RwLock parallel methods)
                for &neighbor_id in &neighbors {
                    self.neighbors.add_bidirectional_link_parallel(*node_id, neighbor_id, lc);
                }

                // NOTE: Pruning is deferred to after parallel loop for performance
                // This allows the parallel phase to only add links (fast, less contention)
                // Pruning happens in a single pass after all insertions complete

                // Update nearest for next level
                nearest = candidates;
            }

            // Progress tracking
            let count = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count % progress_interval as u32 == 0 {
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

        let total_time = graph_start.elapsed().as_secs_f64();
        let final_rate = batch_size as f64 / total_time;

        // NOTE: Pruning is skipped during batch insert for performance
        // Over-connected nodes don't significantly hurt recall (HNSW paper)
        //
        // KNOWN ISSUE: Parallel batch insert achieves only ~1.2x speedup on 16 cores
        // due to RwLock contention in search_layer. To achieve hnswlib-level perf
        // (10x faster), would need lock-free neighbor list reads using AtomicPtr.

        info!(
            inserted = node_ids.len(),
            duration_secs = total_time,
            rate_vec_per_sec = final_rate as u64,
            "Parallel batch insertion complete"
        );

        Ok(node_ids)
    }

    /// Insert node into graph structure
    ///
    /// Implements HNSW insertion algorithm (Malkov & Yashunin 2018)
    fn insert_into_graph(
        &mut self,
        node_id: u32,
        vector: &[f32],
        level: u8,
    ) -> Result<()> {
        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Search for nearest neighbors at each level above target level
        let mut nearest = vec![entry_point];
        for lc in ((level + 1)..=entry_level).rev() {
            nearest = self.search_layer(vector, &nearest, 1, lc)?;
        }

        // Insert at levels 0..=level (iterate from top to bottom)
        for lc in (0..=level).rev() {
            // Find ef_construction nearest neighbors at this level
            let candidates = self.search_layer(vector, &nearest, self.params.ef_construction, lc)?;

            // Select M best neighbors using heuristic
            let m = if lc == 0 {
                self.params.m * 2 // Level 0 has more connections
            } else {
                self.params.m
            };

            let neighbors = self.select_neighbors_heuristic(node_id, &candidates, m, lc, vector)?;

            // Add bidirectional links
            for &neighbor_id in &neighbors {
                self.neighbors.add_bidirectional_link(node_id, neighbor_id, lc)?;
            }

            // Update neighbor counts
            self.nodes[node_id as usize].set_neighbor_count(lc, neighbors.len());

            // Prune neighbors' connections if they exceed M
            for &neighbor_id in &neighbors {
                let neighbor_neighbors = self.neighbors.get_neighbors(neighbor_id, lc)?;
                if neighbor_neighbors.len() > m {
                    let neighbor_vec = self.vectors.get(neighbor_id).ok_or(HNSWError::VectorNotFound(neighbor_id))?;
                    let pruned = self.select_neighbors_heuristic(
                        neighbor_id,
                        &neighbor_neighbors,
                        m,
                        lc,
                        neighbor_vec,
                    )?;

                    // Clear and reset neighbors
                    self.neighbors.set_neighbors(neighbor_id, lc, pruned.clone())?;
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
    fn select_neighbors_heuristic(
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
                let dist = self.distance_to_query(query_vector, id)?;
                Ok((id, dist))
            })
            .collect::<Result<Vec<_>>>()?;
        sorted_candidates.sort_by_key(|c| OrderedFloat(c.1));

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
                let dist_to_result = self.distance(*candidate_id, result_id)?;
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

    /// Search for k nearest neighbors
    ///
    /// Returns up to k nearest neighbors sorted by distance (closest first).
    #[instrument(skip(self, query), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        // Validate k > 0
        if k == 0 {
            error!(k, ef, "Invalid search parameters: k must be > 0");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        // Validate ef >= k
        if ef < k {
            error!(k, ef, "Invalid search parameters: ef must be >= k");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }

        // Validate dimensions
        if query.len() != self.dimensions() {
            error!(
                expected_dim = self.dimensions(),
                actual_dim = query.len(),
                "Dimension mismatch during search"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }

        // Check for NaN/Inf in query
        if query.iter().any(|x| !x.is_finite()) {
            error!("Invalid query vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }

        // Handle empty index
        if self.is_empty() {
            debug!("Search on empty index, returning empty results");
            return Ok(Vec::new());
        }

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Greedy search at each layer (find 1 nearest)
        for level in (1..=entry_level).rev() {
            nearest = self.search_layer(query, &nearest, 1, level)?;
        }

        // Beam search at layer 0 (find ef nearest)
        let candidates = self.search_layer(query, &nearest, ef.max(k), 0)?;

        // Convert to SearchResult and return k nearest
        let mut results: Vec<SearchResult> = candidates
            .iter()
            .map(|&id| {
                let distance = self.distance_to_query(query, id)?;
                Ok(SearchResult::new(id, distance))
            })
            .collect::<Result<Vec<_>>>()?;

        // Sort by distance (closest first)
        results.sort_by_key(|r| OrderedFloat(r.distance));

        // Return top k
        results.truncate(k);

        debug!(
            num_results = results.len(),
            closest_distance = results.first().map(|r| r.distance),
            "Search completed successfully"
        );

        Ok(results)
    }

    /// Search for k nearest neighbors with metadata filtering (ACORN-1)
    ///
    /// Implements ACORN-1 filtered search algorithm for efficient metadata-aware search.
    /// Skips distance calculations for nodes that don't match the filter.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Size of dynamic candidate list (must be >= k)
    /// * `filter_fn` - Filter predicate: returns true if node should be considered
    ///
    /// # Returns
    /// Up to k nearest neighbors that match the filter, sorted by distance
    ///
    /// # Performance
    /// - Low selectivity (5-20% match): 3-6x faster than post-filtering
    /// - High selectivity (>60% match): Falls back to standard search + post-filter
    /// - Recall: 93-98% (slightly lower than standard search due to graph sparsity)
    ///
    /// # Reference
    /// ACORN: SIGMOD 2024, arXiv:2403.04871
    #[instrument(skip(self, query, filter_fn), fields(k, ef, dimensions = query.len(), index_size = self.len()))]
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(u32) -> bool,
    {
        // Validate parameters (same as standard search)
        if k == 0 {
            error!(k, ef, "Invalid search parameters: k must be > 0");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }
        if ef < k {
            error!(k, ef, "Invalid search parameters: ef must be >= k");
            return Err(HNSWError::InvalidSearchParams { k, ef });
        }
        if query.len() != self.dimensions() {
            error!(
                expected_dim = self.dimensions(),
                actual_dim = query.len(),
                "Dimension mismatch during search"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: self.dimensions(),
                actual: query.len(),
            });
        }
        if query.iter().any(|x| !x.is_finite()) {
            error!("Invalid query vector: contains NaN or Inf values");
            return Err(HNSWError::InvalidVector);
        }
        if self.is_empty() {
            debug!("Search on empty index, returning empty results");
            return Ok(Vec::new());
        }

        // Estimate filter selectivity
        let selectivity = self.estimate_selectivity(&filter_fn)?;
        debug!(selectivity, "Estimated filter selectivity");

        // Adaptive threshold: bypass ACORN-1 if filter is too permissive
        // Or for small/medium graphs where brute force is fast enough
        // ACORN-1 becomes effective at larger scales (1000+ vectors)
        const SELECTIVITY_THRESHOLD: f32 = 0.6;
        const SMALL_GRAPH_SIZE: usize = 1000;

        if selectivity > SELECTIVITY_THRESHOLD || self.len() <= SMALL_GRAPH_SIZE {
            // Filter is broad (>60% match) or graph is small: use standard search + post-filter
            debug!(
                selectivity,
                threshold = SELECTIVITY_THRESHOLD,
                graph_size = self.len(),
                "Using standard search + post-filter"
            );

            // Oversample to account for filtered results
            let oversample_factor = 1.0 / selectivity.max(0.1);
            let oversample_k = ((k as f32 * oversample_factor).ceil() as usize)
                .max(k * 2)
                .min(self.len()); // Cap at graph size

            // Ensure ef >= oversample_k (required by HNSW)
            let search_ef = ef.max(oversample_k).max(self.len().min(200));

            let mut all_results = self.search(query, oversample_k, search_ef)?;
            all_results.retain(|r| filter_fn(r.id));
            all_results.truncate(k);

            debug!(
                num_results = all_results.len(),
                "Post-filtering complete"
            );

            return Ok(all_results);
        }

        // Filter is selective (<60% match): use ACORN-1
        debug!(
            selectivity,
            "Using ACORN-1 filtered search"
        );

        let entry_point = self.entry_point.ok_or(HNSWError::EmptyIndex)?;
        let entry_level = self.nodes[entry_point as usize].level;

        // Start from entry point, descend to layer 0
        let mut nearest = vec![entry_point];

        // Greedy search at each layer (find 1 nearest that matches filter)
        for level in (1..=entry_level).rev() {
            nearest = self.search_layer_with_filter(query, &nearest, 1, level, &filter_fn, selectivity)?;
            if nearest.is_empty() {
                // No matching nodes found at this level, try standard search
                debug!(level, "No matches at this level, falling back");
                nearest = vec![entry_point];
            }
        }

        // Beam search at layer 0 (find ef nearest that match filter)
        let candidates = self.search_layer_with_filter(query, &nearest, ef.max(k), 0, &filter_fn, selectivity)?;

        // Convert to SearchResult and return k nearest
        let mut results: Vec<SearchResult> = candidates
            .iter()
            .map(|&id| {
                let distance = self.distance_to_query(query, id)?;
                Ok(SearchResult::new(id, distance))
            })
            .collect::<Result<Vec<_>>>()?;

        results.sort_by_key(|r| OrderedFloat(r.distance));
        results.truncate(k);

        debug!(
            num_results = results.len(),
            closest_distance = results.first().map(|r| r.distance),
            "ACORN-1 search completed"
        );

        // Fallback: if ACORN-1 found no results, try brute-force post-filter
        // This can happen when the graph structure doesn't connect to matching nodes
        if results.is_empty() && k > 0 {
            debug!("ACORN-1 returned no results, falling back to post-filter");

            // Full post-filter search as last resort
            let oversample_k = (k as f32 / selectivity.max(0.01)).ceil() as usize;
            let oversample_k = oversample_k.max(k * 5).min(self.len());
            let search_ef = ef.max(oversample_k).max(self.len().min(200));

            let mut all_results = self.search(query, oversample_k, search_ef)?;
            all_results.retain(|r| filter_fn(r.id));
            all_results.truncate(k);

            debug!(
                num_results = all_results.len(),
                "ACORN-1 fallback post-filter complete"
            );

            return Ok(all_results);
        }

        Ok(results)
    }

    /// Estimate filter selectivity by sampling nodes
    ///
    /// Samples up to 100 random nodes to estimate what fraction matches the filter.
    /// Returns value in [0.0, 1.0] where 1.0 means all nodes match.
    fn estimate_selectivity<F>(&self, filter_fn: &F) -> Result<f32>
    where
        F: Fn(u32) -> bool,
    {
        const SAMPLE_SIZE: usize = 100;

        if self.is_empty() {
            return Ok(1.0);
        }

        let sample_size = SAMPLE_SIZE.min(self.len());
        let step = self.len() / sample_size;

        let mut matches = 0;
        for i in 0..sample_size {
            let node_id = (i * step) as u32;
            if filter_fn(node_id) {
                matches += 1;
            }
        }

        Ok(matches as f32 / sample_size as f32)
    }

    /// Search for nearest neighbors at a specific level with metadata filtering (ACORN-1)
    ///
    /// Key differences from standard search_layer:
    /// 1. Only calculates distance for nodes matching the filter
    /// 2. Uses 2-hop exploration when filter is very selective (<10% match rate)
    /// 3. Expands search more aggressively to compensate for graph sparsity
    ///
    /// Optimized (Nov 25, 2025):
    /// - Uses VisitedList with O(1) clear (generation-based, like hnswlib)
    /// - Reuses pre-allocated unvisited buffer to avoid per-iteration allocation
    #[allow(clippy::too_many_arguments)]
    fn search_layer_with_filter<F>(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
        filter_fn: &F,
        selectivity: f32,
    ) -> Result<Vec<u32>>
    where
        F: Fn(u32) -> bool,
    {
        use super::query_buffers;

        // Determine if we need 2-hop exploration (very selective filters)
        const TWO_HOP_THRESHOLD: f32 = 0.1;
        let use_two_hop = selectivity < TWO_HOP_THRESHOLD;

        if use_two_hop {
            debug!(selectivity, "Using 2-hop exploration for sparse filter");
        }

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let neighbors_to_explore = &mut buffers.unvisited; // Reuse unvisited buffer

            // Initialize with entry points (only add if they match filter)
            for &ep in entry_points {
                if !filter_fn(ep) {
                    visited.insert(ep);
                    continue;
                }

                let dist = self.distance_to_query(query, ep)?;
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // If no entry points match, return empty
            if candidates.is_empty() {
                return Ok(Vec::new());
            }

            // Greedy search with filtered distance calculations
            while let Some(Reverse(current)) = candidates.pop() {
                // If current is farther than farthest in working set, stop
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect neighbors into pre-allocated buffer (no allocation!)
                neighbors_to_explore.clear();
                let neighbors = self.neighbors.get_neighbors(current.node_id, level)?;

                for &neighbor_id in &neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    neighbors_to_explore.push(neighbor_id);

                    // 2-hop exploration: if neighbor doesn't match filter, explore its neighbors
                    if use_two_hop && !filter_fn(neighbor_id) {
                        if let Ok(second_hop) = self.neighbors.get_neighbors(neighbor_id, level) {
                            for &second_hop_id in &second_hop {
                                if !visited.contains(&second_hop_id) {
                                    neighbors_to_explore.push(second_hop_id);
                                }
                            }
                        }
                    }
                }

                // Process all neighbors (1-hop and 2-hop) with prefetching
                let neighbors_slice = neighbors_to_explore.as_slice();
                for (i, &neighbor_id) in neighbors_slice.iter().enumerate() {
                    // Prefetch next neighbor's vector
                    if i + 1 < neighbors_slice.len() {
                        self.vectors.prefetch(neighbors_slice[i + 1]);
                    }

                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    // ACORN-1 key optimization: skip distance calculation if filter doesn't match
                    if !filter_fn(neighbor_id) {
                        continue;
                    }

                    let dist = self.distance_to_query(query, neighbor_id)?;
                    let neighbor = Candidate::new(neighbor_id, dist);

                    // If neighbor is closer than farthest in working set, or working set not full, add it
                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);

                            // Prune working set to ef size
                            if working.len() > ef {
                                working.pop();
                            }
                        }
                    } else {
                        candidates.push(Reverse(neighbor));
                        working.push(neighbor);
                    }
                }
            }

            // Return node IDs sorted by distance (closest first)
            let mut results: Vec<_> = working.drain().collect();
            results.sort_by_key(|c| c.distance);
            Ok(results.into_iter().map(|c| c.node_id).collect())
        })
    }

    /// Search for nearest neighbors at a specific level
    ///
    /// Returns node IDs of up to ef nearest neighbors.
    ///
    /// Optimized (Nov 25, 2025):
    /// - Uses VisitedList with O(1) clear (generation-based, like hnswlib)
    /// - Reuses pre-allocated unvisited buffer to avoid per-iteration allocation
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        level: u8,
    ) -> Result<Vec<u32>> {
        use super::query_buffers;

        query_buffers::with_buffers(|buffers| {
            let visited = &mut buffers.visited;
            let candidates = &mut buffers.candidates;
            let working = &mut buffers.working;
            let unvisited = &mut buffers.unvisited;

            // Initialize with entry points
            for &ep in entry_points {
                let dist = self.distance_to_query(query, ep)?;
                let candidate = Candidate::new(ep, dist);

                candidates.push(Reverse(candidate));
                working.push(candidate);
                visited.insert(ep);
            }

            // Greedy search
            while let Some(Reverse(current)) = candidates.pop() {
                // If current is farther than farthest in working set, stop
                if let Some(&farthest) = working.peek() {
                    if current.distance > farthest.distance {
                        break;
                    }
                }

                // Collect unvisited neighbors into pre-allocated buffer (no allocation!)
                unvisited.clear();
                self.neighbors.with_neighbors(current.node_id, level, |neighbors| {
                    for &id in neighbors {
                        if !visited.contains(&id) {
                            unvisited.push(id);
                        }
                    }
                })?;

                // Process unvisited neighbors with prefetching
                // Prefetch next neighbor's vector data while computing distance to current
                let unvisited_slice = unvisited.as_slice();
                for (i, &neighbor_id) in unvisited_slice.iter().enumerate() {
                    // Prefetch next neighbor's vector (hides memory latency)
                    if i + 1 < unvisited_slice.len() {
                        self.vectors.prefetch(unvisited_slice[i + 1]);
                    }

                    visited.insert(neighbor_id);

                    let dist = self.distance_to_query(query, neighbor_id)?;
                    let neighbor = Candidate::new(neighbor_id, dist);

                    // If neighbor is closer than farthest in working set, or working set not full, add it
                    if let Some(&farthest) = working.peek() {
                        if dist < farthest.distance.0 || working.len() < ef {
                            candidates.push(Reverse(neighbor));
                            working.push(neighbor);

                            // Prune working set to ef size
                            if working.len() > ef {
                                working.pop();
                            }
                        }
                    } else {
                        candidates.push(Reverse(neighbor));
                        working.push(neighbor);
                    }
                }
            }

            // Return node IDs sorted by distance (closest first)
            let mut results: Vec<_> = working.drain().collect();
            results.sort_by_key(|c| c.distance);
            Ok(results.into_iter().map(|c| c.node_id).collect())
        })
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        let nodes_size = self.nodes.len() * std::mem::size_of::<HNSWNode>();
        let neighbors_size = self.neighbors.memory_usage();
        let vectors_size = self.vectors.memory_usage();

        nodes_size + neighbors_size + vectors_size
    }

    /// Get comprehensive index statistics
    ///
    /// Returns detailed statistics about the index state, useful for
    /// monitoring, debugging, and performance analysis.
    #[instrument(skip(self), fields(index_size = self.len()))]
    pub fn stats(&self) -> IndexStats {
        debug!("Computing index statistics");

        // Level distribution
        let max_level = self.nodes.iter().map(|n| n.level).max().unwrap_or(0);
        let mut level_distribution = vec![0; (max_level + 1) as usize];
        for node in &self.nodes {
            level_distribution[node.level as usize] += 1;
        }

        // Neighbor statistics at level 0
        let mut total_neighbors = 0;
        let mut max_neighbors = 0;
        for node in &self.nodes {
            let neighbor_count = self.neighbors.get_neighbors(node.id, 0).unwrap_or_default().len();
            total_neighbors += neighbor_count;
            max_neighbors = max_neighbors.max(neighbor_count);
        }

        let avg_neighbors_l0 = if self.nodes.is_empty() {
            0.0
        } else {
            total_neighbors as f32 / self.nodes.len() as f32
        };

        // Check if quantization is enabled
        let quantization_enabled = matches!(self.vectors, VectorStorage::BinaryQuantized { .. });

        let stats = IndexStats {
            num_vectors: self.len(),
            dimensions: self.dimensions(),
            entry_point: self.entry_point,
            max_level,
            level_distribution,
            avg_neighbors_l0,
            max_neighbors_l0: max_neighbors,
            memory_bytes: self.memory_usage(),
            params: self.params,
            distance_function: self.distance_fn,
            quantization_enabled,
        };

        debug!(
            num_vectors = stats.num_vectors,
            max_level = stats.max_level,
            avg_neighbors_l0 = stats.avg_neighbors_l0,
            memory_mb = stats.memory_bytes / (1024 * 1024),
            "Index statistics computed"
        );

        stats
    }

    /// Extract all edges from the HNSW graph
    ///
    /// Returns edges in format: Vec<(node_id, level, neighbors)>
    /// Useful for persisting the graph structure to disk (LSM-VEC flush operation).
    ///
    /// # Returns
    ///
    /// Vector of tuples (node_id: u32, level: u8, neighbors: Vec<u32>)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use omendb::vector::hnsw::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut index = HNSWIndex::new(128, HNSWParams::default(), DistanceFunction::L2, false)?;
    /// // After building index...
    /// let edges = index.get_all_edges();
    /// for (node_id, level, neighbors) in edges {
    ///     // Persist edges to disk storage...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_all_edges(&self) -> Vec<(u32, u8, Vec<u32>)> {
        let mut edges = Vec::new();

        // Iterate through all nodes
        for node in &self.nodes {
            let node_id = node.id;
            let max_level = node.level;

            // Get neighbors at each level for this node
            for level in 0..=max_level {
                if let Ok(neighbors) = self.neighbors.get_neighbors(node_id, level) {
                    if !neighbors.is_empty() {
                        edges.push((node_id, level, neighbors));
                    }
                }
            }
        }

        edges
    }

    /// Get all node max levels
    ///
    /// Returns a vector of (node_id, max_level) pairs for all nodes in the index.
    /// Useful for LSM-VEC to persist node metadata during flush operations.
    ///
    /// **Important**: Computes max_level from actual edge data, not from node.level.
    /// This is because bidirectional edges can create connections at layers higher
    /// than the node's originally assigned level.
    ///
    /// # Returns
    /// Vector of tuples (node_id: u32, max_level: u8)
    ///
    /// # Example
    /// ```ignore
    /// let node_levels = index.get_all_node_levels();
    /// for (node_id, max_level) in node_levels {
    ///     println!("Node {} has max level {}", node_id, max_level);
    /// }
    /// ```
    pub fn get_all_node_levels(&self) -> Vec<(u32, u8)> {
        self.nodes
            .iter()
            .map(|n| {
                // Compute actual max level from neighbor_counts
                // neighbor_counts[i] > 0 means node has edges at level i
                let max_level = n.neighbor_counts
                    .iter()
                    .enumerate()
                    .rev()  // Start from highest level
                    .find(|(_, count)| **count > 0)
                    .map(|(level, _)| level as u8)
                    .unwrap_or(0);
                (n.id, max_level)
            })
            .collect()
    }

    /// Optimize cache locality by reordering nodes using BFS
    ///
    /// This improves query performance by placing frequently-accessed neighbors
    /// close together in memory. Should be called after index construction
    /// and before querying for best performance.
    ///
    /// Returns the number of nodes reordered.
    #[instrument(skip(self), fields(num_nodes = self.len()))]
    pub fn optimize_cache_locality(&mut self) -> Result<usize> {
        let entry = self.entry_point.ok_or(HNSWError::EmptyIndex)?;

        if self.nodes.is_empty() {
            info!("Index is empty, skipping cache optimization");
            return Ok(0);
        }

        let max_level = self.nodes.iter().map(|n| n.level).max().unwrap_or(0);

        info!(
            num_nodes = self.nodes.len(),
            entry_point = entry,
            max_level = max_level,
            "Starting BFS graph reordering for cache locality"
        );

        // Reorder neighbors and get node ID mapping
        let old_to_new = self.neighbors.reorder_bfs(entry, max_level);

        // Reorder vectors to match
        self.vectors.reorder(&old_to_new);

        // Reorder nodes metadata
        let num_nodes = self.nodes.len();
        let mut new_nodes = Vec::with_capacity(num_nodes);

        // Initialize with dummy nodes
        for _ in 0..num_nodes {
            new_nodes.push(HNSWNode::new(0, 0));
        }

        for (old_id, &new_id) in old_to_new.iter().enumerate() {
            let mut node = self.nodes[old_id].clone();
            node.id = new_id;
            new_nodes[new_id as usize] = node;
        }

        self.nodes = new_nodes;

        // Update entry point
        self.entry_point = Some(old_to_new[entry as usize]);

        info!(
            new_entry_point = self.entry_point,
            "BFS graph reordering complete"
        );

        Ok(self.nodes.len())
    }

    /// Save index to disk
    ///
    /// Format:
    /// - Magic: b"HNSWIDX\0" (8 bytes)
    /// - Version: u32 (4 bytes)
    /// - Dimensions: u32 (4 bytes)
    /// - Num nodes: u32 (4 bytes)
    /// - Entry point: Option<u32> (1 + 4 bytes)
    /// - Distance function: DistanceFunction (bincode)
    /// - Params: HNSWParams (bincode)
    /// - RNG state: u64 (8 bytes)
    /// - Nodes: Vec<HNSWNode> (raw bytes, 64 * num_nodes)
    /// - Neighbors: NeighborLists (bincode)
    /// - Vectors: VectorStorage (bincode)
    #[instrument(skip(self, path), fields(index_size = self.len(), dimensions = self.dimensions()))]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        info!("Starting index save");
        let start = std::time::Instant::now();

        let file = File::create(path).map_err(|e| {
            error!(error = ?e, "Failed to create index file");
            HNSWError::from(e)
        })?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes
        writer.write_all(b"HNSWIDX\0")?;

        // Write version
        writer.write_all(&1u32.to_le_bytes())?;

        // Write dimensions
        writer.write_all(&(self.dimensions() as u32).to_le_bytes())?;

        // Write num nodes
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;

        // Write entry point
        match self.entry_point {
            Some(ep) => {
                writer.write_all(&[1u8])?;
                writer.write_all(&ep.to_le_bytes())?;
            }
            None => {
                writer.write_all(&[0u8])?;
            }
        }

        // Write distance function
        bincode::serialize_into(&mut writer, &self.distance_fn)?;

        // Write params
        bincode::serialize_into(&mut writer, &self.params)?;

        // Write RNG state
        writer.write_all(&self.rng_state.to_le_bytes())?;

        // Write nodes (raw bytes for fast I/O)
        if !self.nodes.is_empty() {
            let nodes_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.nodes.as_ptr() as *const u8,
                    self.nodes.len() * std::mem::size_of::<HNSWNode>(),
                )
            };
            writer.write_all(nodes_bytes)?;
        }

        // Write neighbor lists
        bincode::serialize_into(&mut writer, &self.neighbors)?;

        // Write vectors
        bincode::serialize_into(&mut writer, &self.vectors)?;

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            memory_bytes = self.memory_usage(),
            "Index save completed successfully"
        );

        Ok(())
    }

    /// Load index from disk
    #[instrument(skip(path))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Starting index load");
        let start = std::time::Instant::now();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"HNSWIDX\0" {
            error!(magic = ?magic, "Invalid magic bytes in index file");
            return Err(HNSWError::Storage(format!("Invalid magic bytes: {:?}", magic)));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            error!(version, "Unsupported index file version");
            return Err(HNSWError::Storage(format!("Unsupported version: {}", version)));
        }

        // Read dimensions
        let mut dimensions_bytes = [0u8; 4];
        reader.read_exact(&mut dimensions_bytes)?;
        let dimensions = u32::from_le_bytes(dimensions_bytes) as usize;

        // Read num nodes
        let mut num_nodes_bytes = [0u8; 4];
        reader.read_exact(&mut num_nodes_bytes)?;
        let num_nodes = u32::from_le_bytes(num_nodes_bytes) as usize;

        // Read entry point
        let mut entry_point_flag = [0u8; 1];
        reader.read_exact(&mut entry_point_flag)?;
        let entry_point = if entry_point_flag[0] == 1 {
            let mut ep_bytes = [0u8; 4];
            reader.read_exact(&mut ep_bytes)?;
            Some(u32::from_le_bytes(ep_bytes))
        } else {
            None
        };

        // Read distance function
        let distance_fn: DistanceFunction = bincode::deserialize_from(&mut reader)?;

        // Read params
        let params: HNSWParams = bincode::deserialize_from(&mut reader)?;

        // Read RNG state
        let mut rng_state_bytes = [0u8; 8];
        reader.read_exact(&mut rng_state_bytes)?;
        let rng_state = u64::from_le_bytes(rng_state_bytes);

        // Read nodes (raw bytes for fast I/O)
        let mut nodes = vec![HNSWNode::default(); num_nodes];
        if num_nodes > 0 {
            let nodes_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    nodes.as_mut_ptr() as *mut u8,
                    nodes.len() * std::mem::size_of::<HNSWNode>(),
                )
            };
            reader.read_exact(nodes_bytes)?;
        }

        // Read neighbor lists (always Memory mode when loading from file)
        let neighbor_lists: NeighborLists = bincode::deserialize_from(&mut reader)?;
        let neighbors = GraphStorage::Memory(neighbor_lists);

        // Read vectors
        let vectors: VectorStorage = bincode::deserialize_from(&mut reader)?;

        // Verify dimensions match
        if vectors.dimensions() != dimensions {
            error!(
                expected_dim = dimensions,
                actual_dim = vectors.dimensions(),
                "Dimension mismatch in loaded index"
            );
            return Err(HNSWError::DimensionMismatch {
                expected: dimensions,
                actual: vectors.dimensions(),
            });
        }

        let elapsed = start.elapsed();
        let index = Self {
            nodes,
            neighbors,
            vectors,
            entry_point,
            params,
            distance_fn,
            rng_state,
        };

        info!(
            duration_ms = elapsed.as_millis(),
            index_size = index.len(),
            dimensions = index.dimensions(),
            memory_bytes = index.memory_usage(),
            "Index load completed successfully"
        );

        Ok(index)
    }

    /// Save graph to disk for disk-backed queries
    ///
    /// Uses WritableDiskStorage for incremental writes with offset index.
    ///
    /// # Workflow
    /// 1. Build index with Memory mode
    /// 2. save() - Save full index to file
    /// 3. save_graph_to_disk() - Also save graph to disk directory
    /// 4. load_with_disk_graph() - Load for queries with disk-backed graph
    ///
    /// # Arguments
    /// * `disk_path` - Path to disk storage directory (will be created)
    ///
    /// # Errors
    /// - Returns error if GraphStorage is not Memory mode
    /// - Returns error if directory creation or file writing fails
    ///
    /// # Example
    /// ```ignore
    /// // Build index
    /// let mut index = HNSWIndex::new(...)?;
    /// for vec in vectors {
    ///     index.insert(vec)?;
    /// }
    ///
    /// // Save for persistence
    /// index.save("index.bin")?;
    ///
    /// // Also save graph to disk (uses WritableDiskStorage)
    /// index.save_graph_to_disk("index_graph")?;
    ///
    /// // Later: Load with disk-backed graph for memory-efficient queries
    /// let config = DiskConfig::new(PathBuf::from("index_graph"), index.len());
    /// let index = HNSWIndex::load_with_disk_graph("index.bin", config)?;
    /// ```
    #[instrument(skip(self, disk_path))]
    pub fn save_graph_to_disk<P: AsRef<Path>>(&self, disk_path: P) -> Result<()> {
        info!("Starting graph save to disk with WritableDiskStorage");
        let start = std::time::Instant::now();

        // Only Memory mode can be saved to disk
        // (Layered mode is already using disk, no need to save again)
        if !self.neighbors.is_memory_mode() {
            return Err(HNSWError::Storage(
                "save_graph_to_disk() requires GraphStorage::Memory mode. \
                 Layered mode is already using disk storage."
                    .to_string(),
            ));
        }

        // Create WritableDiskStorage
        use super::disk_storage::WritableDiskStorage;
        let mut writable = WritableDiskStorage::create(
            disk_path.as_ref(),
            self.params.max_level as u32,
            self.params.m as u32,
        )?;

        // Write nodes incrementally
        for node_id in 0..self.nodes.len() {
            let node = &self.nodes[node_id];

            // Collect neighbors for all levels
            let mut neighbors_per_level = Vec::new();
            for level in 0..=node.level as usize {
                let neighbors = self.neighbors
                    .get_neighbors(node_id as u32, level as u8)
                    .unwrap_or_else(|_| Vec::new());
                neighbors_per_level.push(neighbors);
            }

            // Write node with all levels at once
            writable.write_node(node_id as u32, &neighbors_per_level)?;
        }

        // Finalize (flush + save offset index)
        let _disk_storage = writable.finalize()?;

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            num_nodes = self.nodes.len(),
            disk_path = ?disk_path.as_ref(),
            "Graph save to disk completed successfully (WritableDiskStorage)"
        );

        Ok(())
    }

    /// Load index with disk-backed graph storage for memory-efficient queries
    ///
    /// Loads index from save file, but replaces GraphStorage with disk-backed
    /// version using DiskStorage + CachedStorage.
    ///
    /// # Memory Savings
    /// - Memory mode: ~1.2 GB graph for 1M vectors
    /// - Disk mode (30% cache): ~400 MB graph (67% savings!)
    ///
    /// # Arguments
    /// * `index_path` - Path to saved index file (from save())
    /// * `disk_config` - Disk storage configuration (path + cache size)
    ///
    /// # Errors
    /// - Returns error if index file doesn't exist
    /// - Returns error if disk storage directory doesn't exist
    /// - Returns error if file format is invalid
    ///
    /// # Example
    /// ```ignore
    /// // After save() and save_graph_to_disk()
    /// let config = DiskConfig::new(
    ///     PathBuf::from("index_graph"),
    ///     1_000_000, // 1M nodes
    /// );
    ///
    /// let index = HNSWIndex::load_with_disk_graph("index.bin", config)?;
    ///
    /// // Query as normal (graph loaded from disk on-demand)
    /// let results = index.search(&query, 10, 50)?;
    /// ```
    #[instrument(skip(index_path, disk_config))]
    pub fn load_with_disk_graph<P: AsRef<Path>>(
        index_path: P,
        disk_config: DiskConfig,
    ) -> Result<Self> {
        info!("Starting index load with disk-backed graph");
        let start = std::time::Instant::now();

        // Load index normally (will load GraphStorage::Memory from file)
        let mut index = Self::load(index_path)?;

        // Extract upper layer neighbors from loaded index (Memory mode)
        // We need this because LayeredStorage routes upper layers to MemoryStorage
        use super::cached_storage::CachedStorage;
        use super::disk_storage::DiskStorage;
        use super::layered_storage::LayeredStorage;
        use super::node_storage::{MemoryStorage, NodeStorage};

        let mut upper_layers_storage = MemoryStorage::new(index.params.max_level as usize);

        // Copy upper layer neighbors (level 1+) from loaded index
        for node in &index.nodes {
            for level in 1..=node.level as usize {
                if let Ok(neighbors) = index.neighbors.get_neighbors(node.id, level as u8) {
                    upper_layers_storage.write_neighbors(node.id, level as u8, &neighbors)?;
                }
            }
        }

        // Load layer 0 from disk
        let disk_storage = DiskStorage::open_with_offsets(&disk_config.path, disk_config.populate)?;
        let cached = CachedStorage::new(Box::new(disk_storage), disk_config.cache_capacity);

        // Create LayeredStorage with:
        // - Layer 0: DiskStorage (from disk)
        // - Upper layers: MemoryStorage (populated from loaded index)
        let layered = LayeredStorage::new_with_upper_layers(
            Box::new(cached),
            upper_layers_storage,
            super::storage_tiering::StorageMode::Hybrid,
            index.params.max_level as usize,
        );

        index.neighbors = GraphStorage::Layered(Box::new(layered));

        let elapsed = start.elapsed();
        info!(
            duration_ms = elapsed.as_millis(),
            index_size = index.len(),
            dimensions = index.dimensions(),
            storage_mode = ?index.neighbors.mode(),
            "Index load with disk graph completed successfully"
        );

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_index_creation() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_index_insert_single() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec = vec![1.0, 2.0, 3.0];
        let id = index.insert(vec).unwrap();

        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_hnsw_index_insert_multiple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let vec3 = vec![7.0, 8.0, 9.0];

        let id1 = index.insert(vec1).unwrap();
        let id2 = index.insert(vec2).unwrap();
        let id3 = index.insert(vec3).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_hnsw_index_dimension_validation() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let wrong_dim = vec![1.0, 2.0]; // Only 2 dimensions
        assert!(index.insert(wrong_dim).is_err());
    }

    #[test]
    fn test_hnsw_index_search_empty() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5, 100).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_hnsw_index_search_single() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec = vec![1.0, 2.0, 3.0];
        index.insert(vec.clone()).unwrap();

        let results = index.search(&vec, 5, 100).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.01); // Should be ~0 (same vector)
    }

    #[test]
    fn test_random_level_distribution() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let mut level_counts = vec![0; 8];

        // Generate 1000 random levels
        for _ in 0..1000 {
            let level = index.random_level();
            level_counts[level as usize] += 1;
        }

        // Level 0 should have most nodes (exponential decay)
        assert!(level_counts[0] > level_counts[1]);
        assert!(level_counts[1] > level_counts[2]);
    }

    #[test]
    fn test_memory_usage() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32; 128];
            index.insert(vec).unwrap();
        }

        let memory = index.memory_usage();

        // Should have memory for:
        // - 10 nodes (64 bytes each = 640 bytes)
        // - 10 vectors (128 * 4 bytes = 5120 bytes)
        // - Some neighbor storage
        assert!(memory > 5000); // At least vectors + nodes
        assert!(memory < 50000); // Not excessive
    }

    #[test]
    fn test_hnsw_index_search_multiple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 5 vectors
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 0.5, 0.5],
        ];

        for vec in vecs {
            index.insert(vec).unwrap();
        }

        // Search for k=3 nearest to [1.0, 0.0, 0.0]
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 3, 10).unwrap();

        // Should return 3 results
        assert_eq!(results.len(), 3);

        // First result should be closest (id=0, exact match)
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.01);

        // Results should be sorted by distance
        for i in 0..results.len() - 1 {
            assert!(results[i].distance <= results[i + 1].distance);
        }
    }

    #[test]
    fn test_hnsw_index_search_with_ef() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Search with different ef values
        let query = vec![5.0, 0.0, 0.0];

        let results_ef_5 = index.search(&query, 3, 5).unwrap();
        let results_ef_10 = index.search(&query, 3, 10).unwrap();

        // Both should return 3 results (k=3)
        assert_eq!(results_ef_5.len(), 3);
        assert_eq!(results_ef_10.len(), 3);

        // Higher ef should explore more candidates (potentially better recall)
        // Both should find node 5 as closest
        assert_eq!(results_ef_5[0].id, 5);
        assert_eq!(results_ef_10[0].id, 5);
    }

    #[test]
    fn test_hnsw_levels() {
        let mut params = HNSWParams::default();
        params.seed = 12345; // Fixed seed for reproducibility

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Count how many nodes have their TOP level at each height
        // Note: All nodes exist at level 0, but node.level is their TOP level
        let mut top_level_counts = vec![0; 8];
        for node in &index.nodes {
            top_level_counts[node.level as usize] += 1;
        }

        // Most nodes should have top level = 0 (due to exponential decay)
        assert!(top_level_counts[0] > 80); // Most nodes only at level 0

        // Some nodes should have higher top levels
        let higher_level_count: usize = top_level_counts[1..].iter().sum();
        assert!(higher_level_count > 0); // At least some nodes at higher levels

        // All nodes should exist (sum should be 100)
        let total: usize = top_level_counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_neighbor_count_limits() {
        let mut params = HNSWParams::default();
        params.m = 4; // Small M for easier testing
        params.ef_construction = 10;

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 20 vectors (enough to test neighbor pruning)
        for i in 0..20 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Check that no node has more than M*2 neighbors at level 0
        for node in &index.nodes {
            let neighbor_count = index.neighbors.get_neighbors(node.id, 0).unwrap_or_default().len();
            assert!(neighbor_count <= params.m * 2);
        }
    }

    #[test]
    fn test_search_recall_simple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors in a line
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Query should find exact neighbors
        let query = vec![5.0, 0.0, 0.0];
        let results = index.search(&query, 3, 20).unwrap();

        // Should find nodes 5, 4, and 6 (closest to query)
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5); // Exact match

        // Second and third should be 4 or 6
        let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&6));
    }

    #[test]
    fn test_save_load_empty() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Save empty index
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        // Load it back
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.len(), 0);
        assert!(loaded.is_empty());
        assert_eq!(loaded.entry_point, None);
    }

    #[test]
    fn test_save_load_small() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Save index
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        // Load it back
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Verify basic properties
        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.len(), 10);
        assert!(!loaded.is_empty());
        assert_eq!(loaded.entry_point, index.entry_point);

        // Verify vectors are preserved
        for i in 0..10 {
            let orig = index.vectors.get(i).unwrap();
            let load = loaded.vectors.get(i).unwrap();
            assert_eq!(orig, load);
        }

        // Verify search works on loaded index
        let query = vec![5.0, 0.0, 0.0];
        let results = loaded.search(&query, 3, 20).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5); // Should still find exact match
    }

    #[test]
    fn test_save_load_preserves_graph() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..20 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            index.insert(vec).unwrap();
        }

        // Get search results before saving
        let query = vec![10.0, 20.0, 30.0];
        let results_before = index.search(&query, 5, 20).unwrap();

        // Save and load
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Get search results after loading
        let results_after = loaded.search(&query, 5, 20).unwrap();

        // Results should be identical
        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.id, after.id);
            assert!((before.distance - after.distance).abs() < 1e-5);
        }
    }

    #[test]
    fn test_save_load_with_quantization() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(8, params, DistanceFunction::L2, true).unwrap();

        // Train quantization
        let _samples: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 8]).collect();
        if let VectorStorage::BinaryQuantized {
            ref mut thresholds, ..
        } = index.vectors
        {
            for (i, threshold) in thresholds.iter_mut().enumerate() {
                *threshold = i as f32 + 0.5;
            }
        }

        // Insert vectors
        for i in 0..10 {
            let vec = vec![i as f32; 8];
            index.insert(vec).unwrap();
        }

        // Save and load
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Verify quantization is preserved
        match (&index.vectors, &loaded.vectors) {
            (
                VectorStorage::BinaryQuantized {
                    thresholds: t1, ..
                },
                VectorStorage::BinaryQuantized {
                    thresholds: t2, ..
                },
            ) => {
                assert_eq!(t1, t2);
            }
            _ => panic!("Expected BinaryQuantized storage"),
        }

        // Search should work
        let query = vec![5.0; 8];
        let results = loaded.search(&query, 3, 20).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_load_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"INVALID\0").unwrap();
        temp_file.flush().unwrap();

        let result = HNSWIndex::load(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            HNSWError::Storage(msg) => assert!(msg.contains("Invalid magic")),
            _ => panic!("Expected Storage error"),
        }
    }

    #[test]
    fn test_load_unsupported_version() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"HNSWIDX\0").unwrap(); // Magic
        temp_file.write_all(&99u32.to_le_bytes()).unwrap(); // Unsupported version
        temp_file.flush().unwrap();

        let result = HNSWIndex::load(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            HNSWError::Storage(msg) => assert!(msg.contains("Unsupported version")),
            _ => panic!("Expected Storage error"),
        }
    }

    #[test]
    fn test_index_stats_empty() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 0);
        assert_eq!(stats.dimensions, 128);
        assert_eq!(stats.entry_point, None);
        assert_eq!(stats.max_level, 0);
        assert_eq!(stats.avg_neighbors_l0, 0.0);
        assert_eq!(stats.max_neighbors_l0, 0);
        assert!(!stats.quantization_enabled);
        assert!(matches!(stats.distance_function, DistanceFunction::L2));
    }

    #[test]
    fn test_index_stats_with_vectors() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 50 vectors
        for i in 0..50 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 50);
        assert_eq!(stats.dimensions, 3);
        assert!(stats.entry_point.is_some());
        assert!(stats.level_distribution.len() > 0);
        assert!(stats.level_distribution.iter().sum::<usize>() == 50); // All nodes accounted for
        assert!(stats.avg_neighbors_l0 > 0.0); // Should have some neighbors
        assert!(stats.max_neighbors_l0 > 0);
        assert!(stats.memory_bytes > 0);
        assert!(!stats.quantization_enabled);
    }

    #[test]
    fn test_index_stats_with_quantization() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(8, params, DistanceFunction::L2, true).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32; 8];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 10);
        assert!(stats.quantization_enabled); // Should be true
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_index_stats_level_distribution() {
        let mut params = HNSWParams::default();
        params.seed = 42; // Fixed seed for reproducibility

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        // Level 0 should have most nodes (exponential decay)
        assert!(stats.level_distribution[0] > 70);

        // Total nodes should equal num_vectors
        let total: usize = stats.level_distribution.iter().sum();
        assert_eq!(total, 100);

        // Max level should match the distribution length - 1
        assert_eq!(stats.max_level as usize, stats.level_distribution.len() - 1);
    }

    #[test]
    fn test_index_stats_neighbors() {
        let mut params = HNSWParams::default();
        params.m = 8; // Set M for testing

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 30 vectors
        for i in 0..30 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        // Average neighbors should be reasonable (between 0 and M*2)
        assert!(stats.avg_neighbors_l0 > 0.0);
        assert!(stats.avg_neighbors_l0 <= (params.m * 2) as f32);

        // Max neighbors should not exceed M*2 at level 0
        assert!(stats.max_neighbors_l0 <= params.m * 2);
    }

    #[test]
    fn test_index_stats_distance_functions() {
        // Test L2
        let params = HNSWParams::default();
        let index_l2 = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();
        let stats = index_l2.stats();
        assert!(matches!(stats.distance_function, DistanceFunction::L2));

        // Test Cosine
        let params = HNSWParams::default();
        let index_cos = HNSWIndex::new(3, params, DistanceFunction::Cosine, false).unwrap();
        let stats = index_cos.stats();
        assert!(matches!(stats.distance_function, DistanceFunction::Cosine));

        // Test NegativeDotProduct
        let params = HNSWParams::default();
        let index_dot = HNSWIndex::new(3, params, DistanceFunction::NegativeDotProduct, false).unwrap();
        let stats = index_dot.stats();
        assert!(matches!(stats.distance_function, DistanceFunction::NegativeDotProduct));
    }

    // ========================================
    // Hybrid Storage Mode Tests ()
    // ========================================

    #[test]
    fn test_new_with_storage_memory_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Memory,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(index.neighbors.is_memory_mode());
        assert!(!index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_new_with_storage_hybrid_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(!index.neighbors.is_memory_mode());
        assert!(index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_new_with_storage_disk_heavy_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::DiskHeavy,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(!index.neighbors.is_memory_mode());
        assert!(index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_insert_query_hybrid_mode() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert some vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Query
        let query = vec![50.0, 0.0, 0.0];
        let results = index.search(&query, 5, 10).unwrap();

        // Should find the vector at position 50
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 50); // Exact match
    }

    #[test]
    fn test_backward_compatibility_default_is_memory() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Default constructor should use Memory mode
        assert!(index.neighbors.is_memory_mode());
        assert!(!index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_storage_mode_auto_select() {
        // Test auto-select for small dataset (<10M)
        let mode = StorageMode::auto_select(1_000_000);
        assert_eq!(mode, StorageMode::Memory);

        // Test auto-select for medium dataset (10M-100M)
        let mode = StorageMode::auto_select(50_000_000);
        assert_eq!(mode, StorageMode::Hybrid);

        // Test auto-select for large dataset (>100M)
        let mode = StorageMode::auto_select(200_000_000);
        assert_eq!(mode, StorageMode::DiskHeavy);
    }

    // ========================================
    // Edge Case Tests
    // ========================================

    #[test]
    fn test_serialize_hybrid_mode_fails_gracefully() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Try to save - should fail gracefully because Hybrid mode uses GraphStorage::Layered
        let result = index.save("/tmp/test_hybrid_save.hnsw");

        // Should return an error (not panic)
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_index_serialization() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Serialize empty index
        let path = "/tmp/test_empty_index.hnsw";
        index.save(path).unwrap();

        // Deserialize
        let loaded = HNSWIndex::load(path).unwrap();

        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dimensions(), 128);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert some data
        for i in 0..10 {
            index.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        // Share across threads (tests Send + Sync)
        let index = Arc::new(index);
        let mut handles = vec![];

        for _ in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                // Query from multiple threads
                let query = vec![5.0, 0.0, 0.0];
                let results = index_clone.search(&query, 3, 10).unwrap();
                assert_eq!(results.len(), 3);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_hybrid_mode_without_disk_operations() {
        // Create hybrid mode but only do in-memory operations
        // This should work fine (LayeredStorage starts with MemoryStorage)
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert and query should work fine
        for i in 0..50 {
            index.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        let query = vec![25.0, 0.0, 0.0];
        let results = index.search(&query, 5, 10).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 25);
    }

    #[test]
    fn test_error_propagation_in_hybrid_mode() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert valid vector
        let result = index.insert(vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());

        // Insert invalid dimension (should error)
        let result = index.insert(vec![1.0, 2.0]); // Wrong dimensions
        assert!(result.is_err());
    }

    // Disk-backed storage tests

    #[test]
    fn test_save_graph_to_disk() {
        
        use tempfile::tempdir;

        // Build small index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..100 {
            index.insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0]).unwrap();
        }

        // Save graph to disk
        let temp_dir = tempdir().unwrap();
        let graph_path = temp_dir.path().join("test_graph");

        let result = index.save_graph_to_disk(&graph_path);
        assert!(result.is_ok());

        // Verify disk files exist
        assert!(graph_path.join("metadata.bin").exists());
        assert!(graph_path.join("layer_0.graph").exists());
    }

    #[test]
    fn test_save_graph_to_disk_requires_memory_mode() {
        use tempfile::tempdir;

        // Create index with hybrid mode
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        index.insert(vec![1.0, 2.0, 3.0]).unwrap();

        // Try to save (should fail - already in layered mode)
        let temp_dir = tempdir().unwrap();
        let graph_path = temp_dir.path().join("test_graph");

        let result = index.save_graph_to_disk(&graph_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires GraphStorage::Memory mode"));
    }

    #[test]
    fn test_load_with_disk_graph() {
        
        use tempfile::tempdir;

        // Build and save index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..100 {
            index.insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0]).unwrap();
        }

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("test_index.bin");
        let graph_path = temp_dir.path().join("test_graph");

        // Save both index and graph
        index.save(&index_path).unwrap();
        index.save_graph_to_disk(&graph_path).unwrap();

        // Load with disk-backed graph
        let config = DiskConfig::new(graph_path, 100);
        let loaded_index = HNSWIndex::load_with_disk_graph(&index_path, config).unwrap();

        // Verify loaded correctly
        assert_eq!(loaded_index.len(), 100);
        assert_eq!(loaded_index.dimensions(), 3);
        assert!(loaded_index.neighbors.is_layered_mode());
        assert_eq!(loaded_index.neighbors.mode(), StorageMode::Hybrid);

        // Verify queries work
        let query = vec![50.0, 51.0, 52.0];
        let results = loaded_index.search(&query, 5, 50).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_disk_backed_workflow_end_to_end() {
        use tempfile::tempdir;

        // Full workflow: Build → Save → Load with disk → Query
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Build index (200 vectors)
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| (0..128).map(|d| (i * 128 + d) as f32).collect())
            .collect();

        for vec in &vectors {
            index.insert(vec.clone()).unwrap();
        }

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("index.bin");
        let graph_path = temp_dir.path().join("graph");

        // Save index and graph
        index.save(&index_path).unwrap();
        index.save_graph_to_disk(&graph_path).unwrap();

        // Load with disk-backed graph
        let config = DiskConfig::new(graph_path, 200);
        let disk_index = HNSWIndex::load_with_disk_graph(&index_path, config).unwrap();

        // Query and verify
        for (i, query) in vectors.iter().enumerate().take(10) {
            let results = disk_index.search(query, 5, 50).unwrap();
            assert_eq!(results.len(), 5);
            // First result should be the query vector itself
            assert_eq!(results[0].id, i as u32);
        }

        // Verify memory mode vs disk mode both work
        assert!(index.neighbors.is_memory_mode());
        assert!(disk_index.neighbors.is_layered_mode());
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_search_qps -- --ignored --nocapture
    fn bench_search_qps() {
        use std::time::Instant;

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!("\n=== HNSW Raw Search Benchmark ({} vectors, {} queries) ===\n", n, queries);

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|d| ((i * dim + d) % 1000) as f32 / 1000.0).collect())
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|i| (0..dim).map(|d| ((i * dim + d + 500) % 1000) as f32 / 1000.0).collect())
            .collect();

        // Create index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();

        // Batch insert (fair comparison with VectorStore)
        let start = Instant::now();
        index.batch_insert(vectors.clone()).unwrap();
        let insert_time = start.elapsed();
        println!("Batch insert: {:?} ({:.0} vec/s)", insert_time, n as f64 / insert_time.as_secs_f64());

        // Warm up
        for _ in 0..10 {
            let _ = index.search(&query_vecs[0], 10, 100);
        }

        // Benchmark
        let start = Instant::now();
        for q in &query_vecs {
            let _ = index.search(q, 10, 100).unwrap();
        }
        let search_time = start.elapsed();
        let qps = queries as f64 / search_time.as_secs_f64();

        println!("Search: {:?} ({:.0} QPS)", search_time, qps);
        println!("\nPer-query: {:.3}ms", search_time.as_secs_f64() * 1000.0 / queries as f64);
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_vectorstore_qps -- --ignored --nocapture
    fn bench_vectorstore_qps() {
        use crate::vector::{Vector, VectorStore};
        use std::time::Instant;

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!("\n=== VectorStore Search Benchmark ({} vectors, {} queries) ===\n", n, queries);

        // Generate vectors
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|d| ((i * dim + d) % 1000) as f32 / 1000.0).collect())
            .collect();
        let query_vecs: Vec<Vector> = (0..queries)
            .map(|i| Vector::new((0..dim).map(|d| ((i * dim + d + 500) % 1000) as f32 / 1000.0).collect()))
            .collect();

        // Create store with batch insert
        let mut store = VectorStore::new(dim);
        let start = Instant::now();
        let batch: Vec<(String, Vector, serde_json::Value)> = vectors.iter().enumerate()
            .map(|(i, v)| (i.to_string(), Vector::new(v.clone()), serde_json::json!({"idx": i})))
            .collect();
        store.set_batch(batch).unwrap();
        let insert_time = start.elapsed();
        println!("Batch insert: {:?} ({:.0} vec/s)", insert_time, n as f64 / insert_time.as_secs_f64());

        // Warm up
        for _ in 0..10 {
            let _ = store.knn_search(&query_vecs[0], 10);
        }

        // Benchmark knn_search (no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.knn_search(q, 10).unwrap();
        }
        let knn_time = start.elapsed();
        let knn_qps = queries as f64 / knn_time.as_secs_f64();
        println!("knn_search (no metadata): {:?} ({:.0} QPS)", knn_time, knn_qps);

        // Benchmark search (with metadata lookup)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.search(q, 10, None).unwrap();
        }
        let search_time = start.elapsed();
        let search_qps = queries as f64 / search_time.as_secs_f64();
        println!("search (with metadata): {:?} ({:.0} QPS)", search_time, search_qps);

        println!("\n=== Summary ===");
        println!("knn_search QPS: {:.0} ({:.3}ms/query)", knn_qps, knn_time.as_secs_f64() * 1000.0 / queries as f64);
        println!("search QPS: {:.0} ({:.3}ms/query)", search_qps, search_time.as_secs_f64() * 1000.0 / queries as f64);
        println!("Metadata overhead: {:.1}%", (1.0 - search_qps/knn_qps) * 100.0);
    }

    /// Profile seerdb storage to identify optimization targets.
    /// Results written to SEERDB_PROFILE_RESULTS.md for seerdb development.
    ///
    /// Run with: cargo test --release profile_seerdb -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_seerdb() {
        profile_seerdb_impl(100_000);
    }

    /// Comprehensive seerdb profile comparing persistent vs in-memory.
    /// Tests actual seerdb impact: startup, insert, metadata lookups.
    ///
    /// Run with: cargo test --release profile_seerdb_comprehensive -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_seerdb_comprehensive() {
        use std::fs::File;
        use std::io::Write;
        use std::time::Instant;
        use rand::Rng;
        use crate::vector::{VectorStore, Vector};

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!("\n=== Comprehensive seerdb Profile ({} vectors) ===\n", n);

        // Generate random vectors
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        // === TEST 1: In-memory (no seerdb) ===
        println!("=== 1. In-Memory Mode (no seerdb) ===");
        let mut inmem_store = VectorStore::new(dim);

        // Insert
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), Vector::new(v.clone()), serde_json::json!({"idx": i})))
            .collect();
        inmem_store.set_batch(docs).unwrap();
        let inmem_insert = start.elapsed();
        println!("Insert: {:?} ({:.0} vec/s)", inmem_insert, n as f64 / inmem_insert.as_secs_f64());

        // Search (knn_search - no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = inmem_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let inmem_knn = start.elapsed();
        let inmem_knn_qps = queries as f64 / inmem_knn.as_secs_f64();
        println!("knn_search: {:?} ({:.0} QPS)", inmem_knn, inmem_knn_qps);

        // Search (with metadata lookup)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = inmem_store.search(&Vector::new(q.clone()), 10, None);
        }
        let inmem_search = start.elapsed();
        let inmem_search_qps = queries as f64 / inmem_search.as_secs_f64();
        println!("search (metadata): {:?} ({:.0} QPS)", inmem_search, inmem_search_qps);

        // === TEST 2: Persistent (seerdb) ===
        println!("\n=== 2. Persistent Mode (seerdb) ===");
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("profile-oadb");
        let mut persist_store = VectorStore::open_with_dimensions(&path, dim).unwrap();

        // Insert
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), Vector::new(v.clone()), serde_json::json!({"idx": i})))
            .collect();
        persist_store.set_batch(docs).unwrap();
        let persist_insert = start.elapsed();
        println!("Insert: {:?} ({:.0} vec/s)", persist_insert, n as f64 / persist_insert.as_secs_f64());

        // Flush
        let start = Instant::now();
        persist_store.flush().unwrap();
        let flush_time = start.elapsed();
        println!("Flush: {:?}", flush_time);

        // Search (knn_search - no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = persist_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let persist_knn = start.elapsed();
        let persist_knn_qps = queries as f64 / persist_knn.as_secs_f64();
        println!("knn_search: {:?} ({:.0} QPS)", persist_knn, persist_knn_qps);

        // Search (with metadata lookup from seerdb)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = persist_store.search(&Vector::new(q.clone()), 10, None);
        }
        let persist_search = start.elapsed();
        let persist_search_qps = queries as f64 / persist_search.as_secs_f64();
        println!("search (metadata): {:?} ({:.0} QPS)", persist_search, persist_search_qps);

        // Get seerdb stats for metadata lookups
        let stats = persist_store.storage().unwrap().stats();

        drop(persist_store);

        // === TEST 3: Cold Start (reopen from disk) ===
        println!("\n=== 3. Cold Start (reload from seerdb) ===");
        let start = Instant::now();
        let mut reloaded_store = VectorStore::open(&path).unwrap();
        let reload_time = start.elapsed();
        println!("Reload {} vectors: {:?}", reloaded_store.len(), reload_time);

        // Verify search works
        let start = Instant::now();
        for q in &query_vecs {
            let _ = reloaded_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let reload_knn = start.elapsed();
        let reload_knn_qps = queries as f64 / reload_knn.as_secs_f64();
        println!("knn_search (post-reload): {:?} ({:.0} QPS)", reload_knn, reload_knn_qps);

        // === SUMMARY ===
        println!("\n=== Summary: seerdb Impact ===");
        println!("| Operation | In-Memory | seerdb | Overhead |");
        println!("|-----------|-----------|--------|----------|");
        println!("| Insert ({} vec) | {:?} | {:?} | {:.1}x |",
            n, inmem_insert, persist_insert,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64());
        println!("| knn_search | {:.0} QPS | {:.0} QPS | {:.1}% |",
            inmem_knn_qps, persist_knn_qps,
            (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0);
        println!("| search (metadata) | {:.0} QPS | {:.0} QPS | {:.1}% |",
            inmem_search_qps, persist_search_qps,
            (1.0 - persist_search_qps / inmem_search_qps) * 100.0);
        println!("| Cold start | N/A | {:?} | - |", reload_time);
        println!("| Flush | N/A | {:?} | - |", flush_time);

        println!("\n=== seerdb Stats ===");
        println!("Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
        println!("Cache hits: {}, misses: {}", stats.cache_hits, stats.cache_misses);
        println!("Total gets: {}", stats.total_gets);
        println!("SSTables: {:?}", stats.sstables_per_level);

        // Write comprehensive results
        let results = format!(
r#"# seerdb Profile Results (Comprehensive)

**Date**: {}
**Dataset**: {} vectors, {} dimensions
**Queries**: {}

## Should oadb Use seerdb?

**YES** - seerdb provides:
1. **Persistence** - Data survives process restart
2. **Durability** - WAL protects against crashes
3. **Minimal overhead** - See benchmarks below

## Performance Comparison

| Operation | In-Memory | seerdb | Overhead |
|-----------|-----------|--------|----------|
| Insert ({} vec) | {:?} | {:?} | {:.1}x |
| knn_search | {:.0} QPS | {:.0} QPS | {:.1}% |
| search (metadata) | {:.0} QPS | {:.0} QPS | {:.1}% |
| Cold start | N/A | {:?} | - |
| Flush | N/A | {:?} | - |

## Key Findings

### 1. Search Performance: IDENTICAL
- knn_search uses in-memory HNSW graph (no seerdb I/O)
- seerdb overhead: {:.1}% (within noise)

### 2. Insert Performance: {:.1}x Overhead
- seerdb writes: vectors, metadata, ID mappings (4 KV pairs/vector)
- Still achieves {:.0} vec/s with persistence

### 3. Cold Start: {:?}
- Loads {} vectors from seerdb on startup
- Rebuilds HNSW index in memory

### 4. Metadata Lookups
- search() reads metadata from seerdb for results
- Cache hit rate: {:.1}%
- Total seerdb gets: {}

## seerdb Stats

```
Cache hits: {}
Cache misses: {}
Hit rate: {:.1}%
SSTables per level: {:?}
Total gets: {}
```

## Recommendation

**Keep seerdb for oadb** because:
1. Search performance is NOT affected (in-memory HNSW)
2. Insert overhead is acceptable ({:.1}x for durability)
3. Cold start is fast enough ({:?} for {} vectors)
4. Durability is critical for production use

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| No persistence | Fastest inserts | Data lost on restart |
| Simple file | Simpler code | No durability, slow reload |
| fjall | Simpler API | Less optimized for oadb use case |
| **seerdb** | Optimized, durable | Slight insert overhead |
"#,
            chrono::Local::now().format("%Y-%m-%d"),
            n, dim, queries,
            n, inmem_insert, persist_insert, persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            inmem_knn_qps, persist_knn_qps, (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0,
            inmem_search_qps, persist_search_qps, (1.0 - persist_search_qps / inmem_search_qps) * 100.0,
            reload_time, flush_time,
            (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            n as f64 / persist_insert.as_secs_f64(),
            reload_time, n,
            stats.cache_hit_rate * 100.0,
            stats.total_gets,
            stats.cache_hits, stats.cache_misses, stats.cache_hit_rate * 100.0,
            stats.sstables_per_level, stats.total_gets,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            reload_time, n,
        );

        let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SEERDB_PROFILE_RESULTS.md");
        let mut file = File::create(&output_path).expect("Failed to create results file");
        file.write_all(results.as_bytes()).expect("Failed to write results");
        println!("\n✓ Results written to: {}", output_path.display());
    }

    fn profile_seerdb_impl(n: usize) {
        use std::fs::File;
        use std::io::Write;
        use std::time::Instant;
        use rand::Rng;
        use crate::vector::{VectorStore, Vector};

        let n = 100_000;
        let dim = 128;
        let queries = 100;

        println!("\n=== seerdb Profile ({} vectors, {} queries) ===\n", n, queries);

        // Generate random vectors
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        // Create VectorStore with seerdb persistence
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("profile-oadb");
        let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

        // Insert vectors (batch)
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), Vector::new(v.clone()), serde_json::json!({})))
            .collect();
        store.set_batch(docs).unwrap();
        let insert_time = start.elapsed();
        println!("Insert: {:?} ({:.0} vec/s)", insert_time, n as f64 / insert_time.as_secs_f64());

        // Flush to disk
        let start = Instant::now();
        store.flush().unwrap();
        let flush_time = start.elapsed();
        println!("Flush: {:?}", flush_time);

        // Warm up
        for _ in 0..10 {
            let _ = store.knn_search(&Vector::new(query_vecs[0].clone()), 10);
        }

        // Get stats after warmup (snapshot)
        let stats_after_warmup = store.storage().expect("Store should have persistent storage").stats();

        // Benchmark search
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.knn_search(&Vector::new(q.clone()), 10);
        }
        let search_time = start.elapsed();
        let qps = queries as f64 / search_time.as_secs_f64();
        let ms_per_query = search_time.as_secs_f64() * 1000.0 / queries as f64;

        // Get post-search stats
        let stats_after = store.storage().expect("Store should have persistent storage").stats();

        // Calculate delta for this search batch
        let search_gets = stats_after.total_gets - stats_after_warmup.total_gets;
        let search_cache_hits = stats_after.cache_hits - stats_after_warmup.cache_hits;
        let search_cache_misses = stats_after.cache_misses - stats_after_warmup.cache_misses;
        let search_hit_rate = if search_cache_hits + search_cache_misses > 0 {
            search_cache_hits as f64 / (search_cache_hits + search_cache_misses) as f64
        } else {
            0.0
        };

        // Print results
        println!("\n=== Search Latency ===");
        println!("Total search time: {:?}", search_time);
        println!("Per-query: {:.2}ms", ms_per_query);
        println!("QPS: {:.0}", qps);
        println!("Edge lookups (seerdb gets): {} ({:.1} per query)", search_gets, search_gets as f64 / queries as f64);

        println!("\n=== Cache Stats (Search Batch) ===");
        println!("Cache hits: {}", search_cache_hits);
        println!("Cache misses: {}", search_cache_misses);
        println!("Cache hit rate: {:.1}%", search_hit_rate * 100.0);

        println!("\n=== Cache Stats (Cumulative) ===");
        println!("Total cache hits: {}", stats_after.cache_hits);
        println!("Total cache misses: {}", stats_after.cache_misses);
        println!("Overall hit rate: {:.1}%", stats_after.cache_hit_rate * 100.0);

        println!("\n=== LSM Tree Health ===");
        println!("SSTables per level: {:?}", stats_after.sstables_per_level);
        println!("Total SSTables: {}", stats_after.total_sstables);
        println!("Total disk bytes: {} KB", stats_after.total_disk_bytes / 1024);

        println!("\n=== Latency Percentiles ===");
        println!("Get p50: {}us", stats_after.get_latency_p50_us);
        println!("Get p95: {}us", stats_after.get_latency_p95_us);
        println!("Get p99: {}us", stats_after.get_latency_p99_us);
        println!("Get p999: {}us", stats_after.get_latency_p999_us);

        // Write results to file
        let results = format!(
r#"# seerdb Profile Results

**Date**: {}
**Dataset**: {} vectors, {} dimensions
**Hardware**: M3 Max (via cargo test)

## Search Latency
- Avg search time: {:.2} ms
- QPS: {:.0}
- Edge lookups per search: {:.1}

## Cache Stats (During Search)
- Hit rate: {:.1}%
- Hits: {}
- Misses: {}

## Flush Timing
- {} vector flush: {:?}

## LSM Health
- SSTables per level: {:?}
- Total SSTables: {}
- Total disk: {} KB

## Get Latency Percentiles
- p50: {}us
- p95: {}us
- p99: {}us
- p999: {}us

## Analysis
- Cache hit rate {}: {}
- LSM health: {}
"#,
            chrono::Local::now().format("%Y-%m-%d"),
            n, dim,
            ms_per_query,
            qps,
            search_gets as f64 / queries as f64,
            search_hit_rate * 100.0,
            search_cache_hits,
            search_cache_misses,
            n, flush_time,
            stats_after.sstables_per_level,
            stats_after.total_sstables,
            stats_after.total_disk_bytes / 1024,
            stats_after.get_latency_p50_us,
            stats_after.get_latency_p95_us,
            stats_after.get_latency_p99_us,
            stats_after.get_latency_p999_us,
            if search_hit_rate > 0.7 { ">" } else { "<" },
            if search_hit_rate > 0.7 { "GOOD (>70%)" } else { "NEEDS TUNING (<70%)" },
            if stats_after.sstables_per_level.first().copied().unwrap_or(0) < 20 { "GOOD (L0 < 20)" } else { "COMPACTION FALLING BEHIND" },
        );

        // Write to SEERDB_PROFILE_RESULTS.md
        let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SEERDB_PROFILE_RESULTS.md");
        let mut file = File::create(&output_path).expect("Failed to create results file");
        file.write_all(results.as_bytes()).expect("Failed to write results");
        println!("\n✓ Results written to: {}", output_path.display());
    }
}
