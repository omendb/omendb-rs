//! HNSW index statistics and utilities

use super::{HNSWIndex, IndexStats};
use crate::vector::hnsw::error::{HNSWError, Result};
use tracing::{debug, info, instrument};

impl HNSWIndex {
    /// Get memory usage in bytes (approximate)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Get comprehensive index statistics
    ///
    /// Returns detailed statistics about the index state, useful for
    /// monitoring, debugging, and performance analysis.
    #[instrument(skip(self), fields(index_size = self.len()))]
    pub fn stats(&self) -> IndexStats {
        debug!("Computing index statistics");

        // Level distribution
        let max_level = (0..self.storage.len())
            .map(|i| self.storage.level(i as u32))
            .max()
            .unwrap_or(0);
        let mut level_distribution = vec![0; (max_level + 1) as usize];
        for i in 0..self.storage.len() {
            let level = self.storage.level(i as u32);
            level_distribution[level as usize] += 1;
        }

        // Neighbor statistics at level 0
        let mut total_neighbors = 0;
        let mut max_neighbors = 0;
        for i in 0..self.storage.len() {
            let neighbor_count = self.storage.neighbors(i as u32).len();
            total_neighbors += neighbor_count;
            max_neighbors = max_neighbors.max(neighbor_count);
        }

        let avg_neighbors_l0 = if self.storage.is_empty() {
            0.0
        } else {
            total_neighbors as f32 / self.storage.len() as f32
        };

        // Check if quantization is enabled
        let quantization_enabled = self.storage.is_sq8();

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
    /// Returns edges in format: Vec<(`node_id`, level, neighbors)>
    /// Useful for persisting the graph structure to disk (LSM-VEC flush operation).
    ///
    /// # Returns
    ///
    /// Vector of tuples (`node_id`: u32, level: u8, neighbors: Vec<u32>)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use omendb::vector::hnsw::*;
    /// # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    /// # let mut index = HNSWIndex::new(128, HNSWParams::default(), DistanceFunction::L2, false)?;
    /// // After building index...
    /// let edges = index.get_all_edges();
    /// for (node_id, level, neighbors) in edges {
    ///     // Persist edges to disk storage...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn get_all_edges(&self) -> Vec<(u32, u8, Vec<u32>)> {
        let mut edges = Vec::new();

        // Iterate through all nodes
        for i in 0..self.storage.len() {
            let node_id = i as u32;
            let max_level = self.storage.level(node_id);

            // Get neighbors at each level for this node
            for level in 0..=max_level {
                let neighbors = self.storage.neighbors_at_level(node_id, level);
                if !neighbors.is_empty() {
                    edges.push((node_id, level, neighbors));
                }
            }
        }

        edges
    }

    /// Get all node max levels
    ///
    /// Returns a vector of (`node_id`, `max_level`) pairs for all nodes in the index.
    /// Useful for LSM-VEC to persist node metadata during flush operations.
    ///
    /// # Returns
    /// Vector of tuples (`node_id`: u32, `max_level`: u8)
    ///
    /// # Example
    /// ```ignore
    /// let node_levels = index.get_all_node_levels();
    /// for (node_id, max_level) in node_levels {
    ///     println!("Node {} has max level {}", node_id, max_level);
    /// }
    /// ```
    #[must_use]
    pub fn get_all_node_levels(&self) -> Vec<(u32, u8)> {
        (0..self.storage.len())
            .map(|i| {
                let node_id = i as u32;
                let level = self.storage.level(node_id);
                (node_id, level)
            })
            .collect()
    }

    /// Optimize cache locality by reordering nodes using BFS
    ///
    /// This improves query performance by placing frequently-accessed neighbors
    /// close together in memory. Should be called after index construction
    /// and before querying for best performance.
    ///
    /// Returns the old-to-new node ID mapping (old_to_new[old_id] = new_id).
    /// Callers must use this mapping to update any external state that references node IDs.
    #[instrument(skip(self), fields(num_nodes = self.len()))]
    pub fn optimize_cache_locality(&mut self) -> Result<Vec<u32>> {
        let entry = self.entry_point.ok_or(HNSWError::EmptyIndex)?;

        if self.storage.is_empty() {
            info!("Index is empty, skipping cache optimization");
            return Ok(Vec::new());
        }

        let max_level = (0..self.storage.len())
            .map(|i| self.storage.level(i as u32))
            .max()
            .unwrap_or(0);

        info!(
            num_nodes = self.storage.len(),
            entry_point = entry,
            max_level = max_level,
            "Starting BFS graph reordering for cache locality"
        );

        // Perform BFS to get optimal ordering and reorder storage
        let old_to_new = self.storage.reorder_bfs(entry, max_level);

        // Update entry point
        self.entry_point = Some(old_to_new[entry as usize]);

        info!(
            new_entry_point = self.entry_point,
            num_reordered = old_to_new.len(),
            "BFS graph reordering complete"
        );

        Ok(old_to_new)
    }
}
