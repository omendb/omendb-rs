//! HNSW index statistics and utilities

use super::{HNSWIndex, IndexStats};
use crate::vector::hnsw::error::{HNSWError, Result};
use crate::vector::hnsw::storage::VectorStorage;
use crate::vector::hnsw::types::HNSWNode;
use tracing::{debug, info, instrument};

impl HNSWIndex {
    /// Get memory usage in bytes (approximate)
    #[must_use]
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
            let neighbor_count = self.neighbors.get_neighbors(node.id, 0).len();
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
        for node in &self.nodes {
            let node_id = node.id;
            let max_level = node.level;

            // Get neighbors at each level for this node
            for level in 0..=max_level {
                let neighbors = self.neighbors.get_neighbors(node_id, level);
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
    /// **Important**: Computes `max_level` from actual edge data, not from node.level.
    /// This is because bidirectional edges can create connections at layers higher
    /// than the node's originally assigned level.
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
        self.nodes
            .iter()
            .map(|n| {
                // Compute actual max level from neighbor_counts
                // neighbor_counts[i] > 0 means node has edges at level i
                let max_level = n
                    .neighbor_counts
                    .iter()
                    .enumerate()
                    .rev() // Start from highest level
                    .find(|(_, count)| **count > 0)
                    .map_or(0, |(level, _)| level as u8);
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
    /// Returns the old-to-new node ID mapping (old_to_new[old_id] = new_id).
    /// Callers must use this mapping to update any external state that references node IDs.
    #[instrument(skip(self), fields(num_nodes = self.len()))]
    pub fn optimize_cache_locality(&mut self) -> Result<Vec<u32>> {
        let entry = self.entry_point.ok_or(HNSWError::EmptyIndex)?;

        if self.nodes.is_empty() {
            info!("Index is empty, skipping cache optimization");
            return Ok(Vec::new());
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
            num_reordered = old_to_new.len(),
            "BFS graph reordering complete"
        );

        Ok(old_to_new)
    }
}
