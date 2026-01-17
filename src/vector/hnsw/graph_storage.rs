//! Graph storage abstraction for HNSW index
//!
//! Provides a unified API for the in-memory neighbor list storage.
//! Supports both dynamic (construction) and frozen (search-optimized) modes.
//! Persistence is handled by serializing the entire `HNSWIndex` to .omen format.

use super::storage::{FrozenNeighborLists, NeighborLists};
use serde::{Deserialize, Serialize};

/// Graph storage backend for HNSW index
///
/// Supports two modes:
/// - **Dynamic:** Uses `NeighborLists` with `ArcSwap` for thread-safe construction
/// - **Frozen:** Uses `FrozenNeighborLists` with contiguous storage for fast search
///
/// Call `freeze()` after construction to switch to search-optimized mode.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStorage {
    /// Dynamic storage for construction (None after freeze)
    dynamic: Option<NeighborLists>,

    /// Frozen storage for search (None until freeze)
    frozen: Option<FrozenNeighborLists>,
}

impl GraphStorage {
    /// Create new storage with max levels (starts in dynamic mode)
    #[must_use]
    pub fn new(max_levels: usize) -> Self {
        Self {
            dynamic: Some(NeighborLists::new(max_levels)),
            frozen: None,
        }
    }

    /// Create storage with pre-allocated capacity (starts in dynamic mode)
    #[must_use]
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self {
            dynamic: Some(NeighborLists::with_capacity(num_nodes, max_levels, m)),
            frozen: None,
        }
    }

    /// Create from existing neighbor lists (used when loading from persistence)
    #[must_use]
    pub fn from_neighbor_lists(lists: NeighborLists) -> Self {
        Self {
            dynamic: Some(lists),
            frozen: None,
        }
    }

    /// Create from frozen neighbor lists (used when loading optimized format)
    #[must_use]
    pub fn from_frozen(frozen: FrozenNeighborLists) -> Self {
        Self {
            dynamic: None,
            frozen: Some(frozen),
        }
    }

    /// Check if storage is in frozen (search-optimized) mode
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen.is_some()
    }

    /// Freeze the storage for search optimization (10-20% QPS gain)
    ///
    /// Converts dynamic `ArcSwap`-based storage to contiguous memory layout.
    /// After freezing, modification methods will panic.
    ///
    /// Safe to call multiple times (no-op if already frozen).
    pub fn freeze(&mut self) {
        if self.frozen.is_some() {
            return; // Already frozen
        }

        if let Some(ref dynamic) = self.dynamic {
            self.frozen = Some(dynamic.freeze());
            self.dynamic = None;
        }
    }

    /// Unfreeze the storage to allow modifications
    ///
    /// Converts frozen storage back to dynamic mode by deserializing.
    /// This is expensive and should be avoided in normal operation.
    pub fn unfreeze(&mut self) {
        if self.dynamic.is_some() {
            return; // Already unfrozen
        }

        // We need to rebuild dynamic from frozen
        // This is expensive but necessary for modifications
        if let Some(ref frozen) = self.frozen {
            let max_levels = frozen.max_levels();
            let num_nodes = frozen.num_nodes();
            let m_max = frozen.m_max();

            let mut lists = NeighborLists::with_capacity(num_nodes, max_levels, m_max / 2);

            // Rebuild from frozen data
            for node_id in 0..num_nodes {
                for level in 0..max_levels {
                    let neighbors = frozen.get_neighbors(node_id as u32, level as u8);
                    if !neighbors.is_empty() {
                        lists.set_neighbors(node_id as u32, level as u8, neighbors.to_vec());
                    }
                }
            }

            self.dynamic = Some(lists);
            self.frozen = None;
        }
    }

    /// Get neighbors for a node at a specific level
    #[inline]
    #[must_use]
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Vec<u32> {
        if let Some(ref frozen) = self.frozen {
            frozen.get_neighbors(node_id, level).to_vec()
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.get_neighbors(node_id, level)
        } else {
            Vec::new()
        }
    }

    /// Execute closure with read access to neighbors (zero-copy)
    ///
    /// This is the hot path for search. Uses contiguous storage if frozen.
    #[inline]
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> R
    where
        F: FnOnce(&[u32]) -> R,
    {
        if let Some(ref frozen) = self.frozen {
            frozen.with_neighbors(node_id, level, f)
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.with_neighbors(node_id, level, f)
        } else {
            f(&[])
        }
    }

    /// Set neighbors for a node at a specific level
    ///
    /// # Panics
    /// Panics if storage is frozen. Call `unfreeze()` first.
    #[inline]
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        if let Some(ref mut dynamic) = self.dynamic {
            dynamic.set_neighbors(node_id, level, neighbors);
        } else {
            panic!("Cannot modify frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Add bidirectional link between two nodes
    ///
    /// # Panics
    /// Panics if storage is frozen. Call `unfreeze()` first.
    #[inline]
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) {
        if let Some(ref mut dynamic) = self.dynamic {
            dynamic.add_bidirectional_link(node_a, node_b, level);
        } else {
            panic!("Cannot modify frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Add bidirectional link (parallel version)
    ///
    /// # Panics
    /// Panics if storage is frozen.
    #[inline]
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        if let Some(ref dynamic) = self.dynamic {
            dynamic.add_bidirectional_link_parallel(node_a, node_b, level);
        } else {
            panic!("Cannot modify frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Remove unidirectional link (parallel version)
    ///
    /// # Panics
    /// Panics if storage is frozen.
    #[inline]
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) {
        if let Some(ref dynamic) = self.dynamic {
            dynamic.remove_link_parallel(node_a, node_b, level);
        } else {
            panic!("Cannot modify frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Set neighbors (parallel version)
    ///
    /// # Panics
    /// Panics if storage is frozen.
    #[inline]
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors: Vec<u32>) {
        if let Some(ref dynamic) = self.dynamic {
            dynamic.set_neighbors_parallel(node_id, level, neighbors);
        } else {
            panic!("Cannot modify frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Get `M_max` (max neighbors per node)
    #[must_use]
    pub fn m_max(&self) -> usize {
        if let Some(ref frozen) = self.frozen {
            frozen.m_max()
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.m_max()
        } else {
            32 // Default
        }
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        if let Some(ref frozen) = self.frozen {
            frozen.memory_usage()
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.memory_usage()
        } else {
            0
        }
    }

    /// Get total number of neighbor entries
    #[must_use]
    pub fn total_neighbors(&self) -> usize {
        if let Some(ref frozen) = self.frozen {
            frozen.total_neighbors()
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.total_neighbors()
        } else {
            0
        }
    }

    /// Prefetch neighbor list into CPU cache
    ///
    /// Hints to CPU that we'll need neighbor data soon. Only beneficial on
    /// x86/ARM servers - disabled on Apple Silicon where DMP handles this.
    #[inline]
    pub fn prefetch(&self, node_id: u32, level: u8) {
        if let Some(ref frozen) = self.frozen {
            frozen.prefetch(node_id, level);
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.prefetch(node_id, level);
        }
    }

    /// Reorder graph using BFS for cache locality
    ///
    /// # Panics
    /// Panics if storage is frozen.
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        if let Some(ref mut dynamic) = self.dynamic {
            dynamic.reorder_bfs(entry_point, start_level)
        } else {
            panic!("Cannot reorder frozen GraphStorage. Call unfreeze() first.");
        }
    }

    /// Get number of nodes
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        if let Some(ref frozen) = self.frozen {
            frozen.num_nodes()
        } else if let Some(ref dynamic) = self.dynamic {
            dynamic.num_nodes()
        } else {
            0
        }
    }

    /// Get the dynamic NeighborLists for serialization
    ///
    /// If storage is frozen, rebuilds the NeighborLists from frozen data.
    /// This is used during persistence to maintain backwards compatibility.
    #[must_use]
    pub fn get_neighbor_lists_for_save(&self) -> NeighborLists {
        if let Some(ref dynamic) = self.dynamic {
            // Clone the dynamic storage
            let num_nodes = dynamic.num_nodes();
            let max_levels = 8; // Default max levels
            let m_max = dynamic.m_max();
            let mut lists = NeighborLists::with_capacity(num_nodes, max_levels, m_max / 2);

            for node_id in 0..num_nodes {
                for level in 0..max_levels {
                    let neighbors = dynamic.get_neighbors(node_id as u32, level as u8);
                    if !neighbors.is_empty() {
                        lists.set_neighbors(node_id as u32, level as u8, neighbors);
                    }
                }
            }
            lists
        } else if let Some(ref frozen) = self.frozen {
            // Rebuild from frozen
            let num_nodes = frozen.num_nodes();
            let max_levels = frozen.max_levels();
            let m_max = frozen.m_max();
            let mut lists = NeighborLists::with_capacity(num_nodes, max_levels, m_max / 2);

            for node_id in 0..num_nodes {
                for level in 0..max_levels {
                    let neighbors = frozen.get_neighbors(node_id as u32, level as u8);
                    if !neighbors.is_empty() {
                        lists.set_neighbors(node_id as u32, level as u8, neighbors.to_vec());
                    }
                }
            }
            lists
        } else {
            NeighborLists::new(8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_storage_new() {
        let storage = GraphStorage::new(8);
        assert_eq!(storage.m_max(), 32);
        assert!(!storage.is_frozen());
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
    fn test_graph_storage_freeze() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);
        storage.set_neighbors(1, 0, vec![0, 2]);

        assert!(!storage.is_frozen());

        // Freeze for search optimization
        storage.freeze();
        assert!(storage.is_frozen());

        // Can still read after freeze
        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(1, 0), vec![0, 2]);

        // with_neighbors still works
        storage.with_neighbors(0, 0, |neighbors| {
            assert_eq!(neighbors, &[1, 2, 3]);
        });
    }

    #[test]
    fn test_graph_storage_unfreeze() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);

        storage.freeze();
        assert!(storage.is_frozen());

        storage.unfreeze();
        assert!(!storage.is_frozen());

        // Can modify again after unfreeze
        storage.set_neighbors(0, 0, vec![1, 2, 3, 4]);
        assert_eq!(storage.get_neighbors(0, 0), vec![1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "Cannot modify frozen")]
    fn test_graph_storage_freeze_prevents_modification() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);

        storage.freeze();

        // This should panic
        storage.set_neighbors(0, 0, vec![4, 5, 6]);
    }

    #[test]
    fn test_graph_storage_serialization() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);

        let serialized = postcard::to_allocvec(&storage).unwrap();
        let deserialized: GraphStorage = postcard::from_bytes(&serialized).unwrap();

        assert_eq!(deserialized.get_neighbors(0, 0), vec![1, 2, 3]);
    }

    #[test]
    fn test_graph_storage_frozen_serialization() {
        let mut storage = GraphStorage::new(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]);
        storage.set_neighbors(1, 0, vec![0, 2]);

        storage.freeze();

        let serialized = postcard::to_allocvec(&storage).unwrap();
        let deserialized: GraphStorage = postcard::from_bytes(&serialized).unwrap();

        assert!(deserialized.is_frozen());
        assert_eq!(deserialized.get_neighbors(0, 0), vec![1, 2, 3]);
        assert_eq!(deserialized.get_neighbors(1, 0), vec![0, 2]);
    }
}
