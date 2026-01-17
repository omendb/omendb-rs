//! Graph storage abstraction for HNSW index
//!
//! Provides a unified API for the in-memory neighbor list storage.
//! Persistence is handled by serializing the entire `HNSWIndex` to .omen format.

use super::storage::{NeighborLists, NeighborStorage};
use serde::{Deserialize, Serialize};

/// Graph storage backend for HNSW index
///
/// Wraps `NeighborStorage` for atomic, lock-free neighbor access.
/// Persistence is handled externally by `.omen` format serialization.
///
/// **Phase 7.1**: Uses atomic slot storage for:
/// - Lock-free reads (hot path for search)
/// - O(1) inserts without copy-on-write
/// - Per-node mutex for parallel construction
#[derive(Debug)]
pub struct GraphStorage {
    storage: NeighborStorage,
}

impl GraphStorage {
    /// Create new storage with max levels and M parameter
    #[must_use]
    pub fn new(max_levels: usize) -> Self {
        Self {
            storage: NeighborStorage::new(max_levels, 16), // Default M=16
        }
    }

    /// Create storage with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self {
            storage: NeighborStorage::with_capacity(num_nodes, max_levels, m),
        }
    }

    /// Create from existing neighbor storage (used when loading from persistence)
    #[must_use]
    pub fn from_neighbor_storage(storage: NeighborStorage) -> Self {
        Self { storage }
    }

    /// Create from legacy NeighborLists (for v2 persistence compatibility)
    #[must_use]
    pub fn from_neighbor_lists(lists: NeighborLists) -> Self {
        // Convert NeighborLists to NeighborStorage
        let num_nodes = lists.num_nodes();
        let max_levels = lists.max_levels();
        let m = lists.m_max() / 2; // m_max = M * 2

        let mut storage = NeighborStorage::new(max_levels, m);

        // Copy all neighbor data
        for node_id in 0..num_nodes {
            // Get max level for this node by finding highest non-empty level
            let mut node_max_level = 0u8;
            for level in (0..max_levels).rev() {
                if !lists.get_neighbors(node_id as u32, level as u8).is_empty() {
                    node_max_level = level as u8;
                    break;
                }
            }

            // Allocate node
            storage.allocate_node(node_id as u32, node_max_level);

            // Copy neighbors at each level
            for level in 0..=node_max_level {
                let neighbors = lists.get_neighbors(node_id as u32, level);
                if !neighbors.is_empty() {
                    storage.set_neighbors(node_id as u32, level, neighbors);
                }
            }
        }

        Self { storage }
    }

    /// Allocate storage for a new node at given level
    ///
    /// Must be called before adding neighbors to a node.
    #[inline]
    pub fn allocate_node(&mut self, node_id: u32, level: u8) {
        self.storage.allocate_node(node_id, level);
    }

    /// Get neighbors for a node at a specific level
    #[inline]
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        self.storage.get_neighbors(node_id, level)
    }

    /// Execute closure with read access to neighbors (zero-copy, lock-free)
    #[inline]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        self.storage.with_neighbors(node_id, level, f)
    }

    /// Set neighbors for a node at a specific level
    #[inline]
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        self.storage.set_neighbors(node_id, level, neighbors);
    }

    /// Add bidirectional link between two nodes
    #[inline]
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) {
        self.storage.add_bidirectional_link(node_a, node_b, level);
    }

    /// Add bidirectional link (parallel version, thread-safe)
    #[inline]
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        self.storage
            .add_bidirectional_link_parallel(node_a, node_b, level);
    }

    /// Remove unidirectional link (parallel version, thread-safe)
    #[inline]
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        self.storage.remove_link_parallel(node_a, node_b, level);
    }

    /// Set neighbors (parallel version, thread-safe)
    #[inline]
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        self.storage
            .set_neighbors_parallel(node_id, level, neighbors);
    }

    /// Get `M_max` (max neighbors per node at level 0)
    #[must_use]
    pub fn m_max(&self) -> usize {
        self.storage.m_max()
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Prefetch neighbor list into CPU cache
    ///
    /// Hints to CPU that we'll need neighbor data soon. Only beneficial on
    /// x86/ARM servers - disabled on Apple Silicon where DMP handles this.
    #[inline]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        self.storage.prefetch(node_id, level);
    }

    /// Reorder graph using BFS for cache locality
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        self.storage.reorder_bfs(entry_point, start_level)
    }

    /// Get internal storage reference (for serialization)
    pub fn inner(&self) -> &NeighborStorage {
        &self.storage
    }

    /// Get mutable internal storage reference
    pub fn inner_mut(&mut self) -> &mut NeighborStorage {
        &mut self.storage
    }
}

/// Serializable form of GraphStorage for persistence
///
/// Converts atomic storage to plain data for serialization.
#[derive(Serialize, Deserialize)]
pub struct GraphStorageData {
    /// Level 0 neighbors: [node_id] -> [neighbor_ids]
    pub level0: Vec<Vec<u32>>,
    /// Upper level neighbors: [node_id] -> [level] -> [neighbor_ids]
    pub upper: Vec<Vec<Vec<u32>>>,
    pub max_m: usize,
    pub max_m0: usize,
    pub max_levels: usize,
}

impl GraphStorage {
    /// Convert to serializable data
    pub fn to_data(&self) -> GraphStorageData {
        let num_nodes = self.storage.num_nodes();
        let mut level0 = Vec::with_capacity(num_nodes);
        let mut upper = Vec::with_capacity(num_nodes);

        for node_id in 0..num_nodes {
            // Level 0
            level0.push(self.storage.get_neighbors(node_id as u32, 0));

            // Upper levels
            let mut node_upper = Vec::new();
            for level in 1..self.storage.max_levels() {
                let neighbors = self.storage.get_neighbors(node_id as u32, level as u8);
                if !neighbors.is_empty() || !node_upper.is_empty() {
                    node_upper.push(neighbors);
                }
            }
            upper.push(node_upper);
        }

        GraphStorageData {
            level0,
            upper,
            max_m: self.storage.m(),
            max_m0: self.storage.m_max(),
            max_levels: self.storage.max_levels(),
        }
    }

    /// Create from serialized data
    pub fn from_data(data: GraphStorageData) -> Self {
        let num_nodes = data.level0.len();
        let m = data.max_m;
        let max_levels = data.max_levels;

        let mut storage = NeighborStorage::new(max_levels, m);

        for node_id in 0..num_nodes {
            // Determine max level for this node
            let upper_levels = &data.upper[node_id];
            let max_level = if upper_levels.is_empty() {
                0
            } else {
                upper_levels.len() as u8
            };

            // Allocate node
            storage.allocate_node(node_id as u32, max_level);

            // Set level 0 neighbors
            if !data.level0[node_id].is_empty() {
                storage.set_neighbors(node_id as u32, 0, data.level0[node_id].clone());
            }

            // Set upper level neighbors
            for (level_idx, neighbors) in upper_levels.iter().enumerate() {
                let level = (level_idx + 1) as u8;
                if !neighbors.is_empty() {
                    storage.set_neighbors(node_id as u32, level, neighbors.clone());
                }
            }
        }

        Self { storage }
    }
}

// Implement Serialize/Deserialize via GraphStorageData
impl Serialize for GraphStorage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_data().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GraphStorage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = GraphStorageData::deserialize(deserializer)?;
        Ok(Self::from_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_storage_new() {
        let storage = GraphStorage::new(8);
        assert_eq!(storage.m_max(), 32); // M=16, M_max = M*2 = 32
    }

    #[test]
    fn test_graph_storage_get_set_neighbors() {
        let mut storage = GraphStorage::with_capacity(10, 8, 16);

        // Allocate nodes first
        storage.allocate_node(0, 1);

        storage.set_neighbors(0, 0, vec![1, 2, 3]);
        storage.set_neighbors(0, 1, vec![4, 5]);

        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(0, 1), vec![4, 5]);
        assert_eq!(storage.get_neighbors(99, 0), Vec::<u32>::new());
    }

    #[test]
    fn test_graph_storage_add_bidirectional_link() {
        let mut storage = GraphStorage::with_capacity(10, 8, 16);

        // Allocate nodes first
        storage.allocate_node(0, 0);
        storage.allocate_node(1, 0);

        storage.add_bidirectional_link(0, 1, 0);

        let neighbors_0 = storage.get_neighbors(0, 0);
        let neighbors_1 = storage.get_neighbors(1, 0);

        assert!(neighbors_0.contains(&1));
        assert!(neighbors_1.contains(&0));
    }

    #[test]
    fn test_graph_storage_serialization() {
        let mut storage = GraphStorage::with_capacity(10, 8, 16);
        storage.allocate_node(0, 0);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);

        let serialized = postcard::to_allocvec(&storage).unwrap();
        let deserialized: GraphStorage = postcard::from_bytes(&serialized).unwrap();

        assert_eq!(deserialized.get_neighbors(0, 0), vec![1, 2, 3]);
    }

    #[test]
    fn test_graph_storage_from_neighbor_lists() {
        let mut lists = NeighborLists::new(8);
        lists.set_neighbors(0, 0, vec![1, 2, 3]);
        lists.set_neighbors(1, 0, vec![0, 2]);

        let storage = GraphStorage::from_neighbor_lists(lists);

        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(1, 0), vec![0, 2]);
    }
}
