//! Graph storage abstraction for HNSW index
//!
//! Provides a unified API for the in-memory neighbor list storage.
//! Persistence is handled by serializing the entire `HNSWIndex` to .omen format.

use super::storage::NeighborLists;
use serde::{Deserialize, Serialize};

/// Graph storage backend for HNSW index
///
/// Wraps `NeighborLists` for in-memory neighbor storage.
/// Persistence is handled externally by `.omen` format serialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStorage(NeighborLists);

impl GraphStorage {
    /// Create new storage with max levels
    #[must_use]
    pub fn new(max_levels: usize) -> Self {
        Self(NeighborLists::new(max_levels))
    }

    /// Create storage with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self(NeighborLists::with_capacity(num_nodes, max_levels, m))
    }

    /// Create from existing neighbor lists (used when loading from persistence)
    #[must_use]
    pub fn from_neighbor_lists(lists: NeighborLists) -> Self {
        Self(lists)
    }

    /// Get neighbors for a node at a specific level
    #[inline]
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        self.0.get_neighbors(node_id, level)
    }

    /// Execute closure with read access to neighbors (zero-copy)
    #[inline]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        self.0.with_neighbors(node_id, level, f)
    }

    /// Set neighbors for a node at a specific level
    #[inline]
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        self.0.set_neighbors(node_id, level, neighbors);
    }

    /// Add bidirectional link between two nodes
    #[inline]
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) {
        self.0.add_bidirectional_link(node_a, node_b, level);
    }

    /// Add bidirectional link (parallel version)
    #[inline]
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        self.0
            .add_bidirectional_link_parallel(node_a, node_b, level);
    }

    /// Remove unidirectional link (parallel version)
    #[inline]
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        self.0.remove_link_parallel(node_a, node_b, level);
    }

    /// Set neighbors (parallel version)
    #[inline]
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        self.0.set_neighbors_parallel(node_id, level, neighbors);
    }

    /// Get `M_max` (max neighbors per node)
    #[must_use]
    pub fn m_max(&self) -> usize {
        self.0.m_max()
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.0.memory_usage()
    }

    /// Prefetch neighbor list into CPU cache
    ///
    /// Hints to CPU that we'll need neighbor data soon. Only beneficial on
    /// x86/ARM servers - disabled on Apple Silicon where DMP handles this.
    #[inline]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        self.0.prefetch(node_id, level);
    }

    /// Reorder graph using BFS for cache locality
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        self.0.reorder_bfs(entry_point, start_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_storage_new() {
        let storage = GraphStorage::new(8);
        assert_eq!(storage.m_max(), 32);
    }

    #[test]
    fn test_graph_storage_get_set_neighbors() {
        let mut storage = GraphStorage::new(8);

        storage.set_neighbors(0, 0, vec![1, 2, 3]);
        storage.set_neighbors(0, 1, vec![4, 5]);

        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(0, 1), vec![4, 5]);
        assert_eq!(storage.get_neighbors(99, 0), Vec::<u32>::new());
    }

    #[test]
    fn test_graph_storage_add_bidirectional_link() {
        let mut storage = GraphStorage::new(8);

        storage.add_bidirectional_link(0, 1, 0);

        let neighbors_0 = storage.get_neighbors(0, 0);
        let neighbors_1 = storage.get_neighbors(1, 0);

        assert!(neighbors_0.contains(&1));
        assert!(neighbors_1.contains(&0));
    }

    #[test]
    fn test_graph_storage_serialization() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);

        let serialized = postcard::to_allocvec(&storage).unwrap();
        let deserialized: GraphStorage = postcard::from_bytes(&serialized).unwrap();

        assert_eq!(deserialized.get_neighbors(0, 0), vec![1, 2, 3]);
    }
}
