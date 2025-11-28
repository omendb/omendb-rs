//! Node storage abstraction for HNSW-IF
//!
//! Provides a storage abstraction layer that allows switching between:
//! - `MemoryStorage`: Fast in-memory storage (current behavior)
//! - `DiskStorage`: mmap-based disk storage for large datasets
//! - `CachedStorage`: LRU cache over `DiskStorage` for hybrid mode
//!
//! This enables automatic tiering:
//! - <10M vectors: Pure memory (no overhead)
//! - 10M-100M: Hybrid (30% cache, 70% disk)
//! - 100M+: Disk-heavy (10% cache, 90% disk)

use super::error::{HNSWError, Result};
use serde::{Deserialize, Serialize};

/// Node ID type (u32 for up to 4B vectors)
pub type NodeId = u32;

/// Level type (u8 for up to 256 levels, typically max 8)
pub type Level = u8;

/// Storage abstraction for HNSW graph nodes
///
/// This trait allows the HNSW index to work with different storage backends
/// without knowing implementation details.
pub trait NodeStorage: Send + Sync {
    /// Read neighbors for a node at a given level
    ///
    /// # Arguments
    /// * `node_id` - ID of the node
    /// * `level` - Level in the HNSW hierarchy (0 = base layer)
    ///
    /// # Returns
    /// Vec of neighbor node IDs (may be empty if no neighbors)
    ///
    /// # Errors
    /// Returns error if node doesn't exist or I/O fails
    fn read_neighbors(&self, node_id: NodeId, level: Level) -> Result<Vec<NodeId>>;

    /// Write neighbors for a node at a given level
    ///
    /// # Arguments
    /// * `node_id` - ID of the node
    /// * `level` - Level in the HNSW hierarchy
    /// * `neighbors` - List of neighbor node IDs
    ///
    /// # Errors
    /// Returns error if node doesn't exist or I/O fails
    fn write_neighbors(
        &mut self,
        node_id: NodeId,
        level: Level,
        neighbors: &[NodeId],
    ) -> Result<()>;

    /// Write all neighbors for a node (all levels at once)
    ///
    /// This is an optimization for storage backends that prefer bulk writes.
    /// Default implementation calls `write_neighbors()` for each level sequentially.
    ///
    /// # Arguments
    /// * `node_id` - ID of the node
    /// * `neighbors_per_level` - Neighbors for each level (level 0, 1, 2, ...)
    ///
    /// # Errors
    /// Returns error if write fails
    ///
    /// # Note
    /// Storage backends like `WritableDiskStorage` can override this for atomic writes.
    fn write_node(&mut self, node_id: NodeId, neighbors_per_level: &[Vec<NodeId>]) -> Result<()> {
        // Default: Write each level sequentially
        for (level, neighbors) in neighbors_per_level.iter().enumerate() {
            self.write_neighbors(node_id, level as Level, neighbors)?;
        }
        Ok(())
    }

    /// Check if a node exists in storage
    ///
    /// # Arguments
    /// * `node_id` - ID of the node to check
    ///
    /// # Returns
    /// true if node exists, false otherwise
    fn exists(&self, node_id: NodeId) -> bool;

    /// Get number of levels for a node
    ///
    /// # Arguments
    /// * `node_id` - ID of the node
    ///
    /// # Returns
    /// Number of levels (0-based, so level 0 = 1 level)
    ///
    /// # Errors
    /// Returns error if node doesn't exist
    fn num_levels(&self, node_id: NodeId) -> Result<usize>;

    /// Get total number of nodes in storage
    fn len(&self) -> usize;

    /// Check if storage is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush any pending writes to persistent storage
    ///
    /// No-op for `MemoryStorage`, important for `DiskStorage`
    fn flush(&mut self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Get memory usage in bytes (approximate)
    fn memory_usage(&self) -> usize;
}

/// In-memory storage for HNSW graph nodes
///
/// This is the default storage for <10M vectors.
/// Wraps the existing `NeighborLists` implementation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryStorage {
    /// Neighbor storage: neighbors[`node_id`][level] = Vec<`neighbor_ids`>
    neighbors: Vec<Vec<Vec<NodeId>>>,

    /// Maximum levels supported (typically 8)
    max_levels: usize,

    /// `M_max` (max neighbors = M * 2) for pre-allocation
    m_max: usize,
}

impl MemoryStorage {
    /// Create empty memory storage
    ///
    /// # Arguments
    /// * `max_levels` - Maximum number of levels (typically 8)
    #[must_use]
    pub fn new(max_levels: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            max_levels,
            m_max: 32, // Default M*2 = 16*2
        }
    }

    /// Create with pre-allocated capacity and M parameter
    ///
    /// # Arguments
    /// * `num_nodes` - Expected number of nodes (for pre-allocation)
    /// * `max_levels` - Maximum number of levels
    /// * `m` - M parameter (max neighbors = M * 2)
    #[must_use]
    pub fn with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self {
            neighbors: Vec::with_capacity(num_nodes),
            max_levels,
            m_max: m * 2,
        }
    }

    /// Ensure node exists (allocate levels if needed)
    ///
    /// Internal helper for write operations
    fn ensure_node(&mut self, node_id: NodeId) {
        let node_idx = node_id as usize;

        // Extend if needed
        while self.neighbors.len() <= node_idx {
            let mut levels = Vec::with_capacity(self.max_levels);
            for _ in 0..self.max_levels {
                levels.push(Vec::with_capacity(self.m_max));
            }
            self.neighbors.push(levels);
        }
    }

    /// Get total number of neighbors across all nodes and levels
    #[must_use]
    pub fn total_neighbors(&self) -> usize {
        self.neighbors
            .iter()
            .flat_map(|node| node.iter())
            .map(std::vec::Vec::len)
            .sum()
    }
}

impl NodeStorage for MemoryStorage {
    fn read_neighbors(&self, node_id: NodeId, level: Level) -> Result<Vec<NodeId>> {
        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if node_idx >= self.neighbors.len() {
            return Ok(Vec::new()); // No neighbors (node doesn't exist)
        }

        if level_idx >= self.neighbors[node_idx].len() {
            return Ok(Vec::new()); // No neighbors at this level
        }

        Ok(self.neighbors[node_idx][level_idx].clone())
    }

    fn write_neighbors(
        &mut self,
        node_id: NodeId,
        level: Level,
        neighbors: &[NodeId],
    ) -> Result<()> {
        self.ensure_node(node_id);

        let node_idx = node_id as usize;
        let level_idx = level as usize;

        if level_idx >= self.max_levels {
            return Err(HNSWError::InvalidLevel {
                level: level_idx,
                max_levels: self.max_levels,
            });
        }

        self.neighbors[node_idx][level_idx] = neighbors.to_vec();
        Ok(())
    }

    fn exists(&self, node_id: NodeId) -> bool {
        (node_id as usize) < self.neighbors.len()
    }

    fn num_levels(&self, node_id: NodeId) -> Result<usize> {
        let node_idx = node_id as usize;

        if node_idx >= self.neighbors.len() {
            return Err(HNSWError::NodeNotFound(node_id));
        }

        // Count non-empty levels from top down
        let mut num_levels = 0;
        for (level, neighbors) in self.neighbors[node_idx].iter().enumerate() {
            if !neighbors.is_empty() {
                num_levels = level + 1; // 1-based count
            }
        }

        Ok(num_levels)
    }

    fn len(&self) -> usize {
        self.neighbors.len()
    }

    fn memory_usage(&self) -> usize {
        let mut total = 0;

        // Size of outer Vec
        total += self.neighbors.capacity() * std::mem::size_of::<Vec<Vec<NodeId>>>();

        // Size of each node's level vecs
        for node in &self.neighbors {
            total += node.capacity() * std::mem::size_of::<Vec<NodeId>>();

            // Size of actual neighbor data
            for level in node {
                total += level.len() * std::mem::size_of::<NodeId>();
            }
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage_new() {
        let storage = MemoryStorage::new(8);
        assert_eq!(storage.len(), 0);
        assert!(storage.is_empty());
        assert_eq!(storage.max_levels, 8);
    }

    #[test]
    fn test_memory_storage_write_read() {
        let mut storage = MemoryStorage::new(8);

        // Write neighbors for node 0, level 0
        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        // Read back
        let neighbors = storage.read_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2, 3]);
    }

    #[test]
    fn test_memory_storage_multi_level() {
        let mut storage = MemoryStorage::new(8);

        // Write to multiple levels
        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        storage.write_neighbors(0, 1, &[4, 5]).unwrap();
        storage.write_neighbors(0, 2, &[6]).unwrap();

        // Read back
        assert_eq!(storage.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(storage.read_neighbors(0, 1).unwrap(), vec![4, 5]);
        assert_eq!(storage.read_neighbors(0, 2).unwrap(), vec![6]);
    }

    #[test]
    fn test_memory_storage_exists() {
        let mut storage = MemoryStorage::new(8);

        assert!(!storage.exists(0));

        storage.write_neighbors(0, 0, &[1]).unwrap();
        assert!(storage.exists(0));

        assert!(!storage.exists(1)); // Not written yet
    }

    #[test]
    fn test_memory_storage_num_levels() {
        let mut storage = MemoryStorage::new(8);

        // Write to levels 0, 1, 2
        storage.write_neighbors(0, 0, &[1]).unwrap();
        storage.write_neighbors(0, 1, &[2]).unwrap();
        storage.write_neighbors(0, 2, &[3]).unwrap();

        // Should have 3 levels (0, 1, 2)
        assert_eq!(storage.num_levels(0).unwrap(), 3);
    }

    #[test]
    fn test_memory_storage_empty_node() {
        let storage = MemoryStorage::new(8);

        // Reading non-existent node returns empty vec
        let neighbors = storage.read_neighbors(99, 0).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_memory_storage_empty_level() {
        let mut storage = MemoryStorage::new(8);

        // Write only to level 0
        storage.write_neighbors(0, 0, &[1, 2]).unwrap();

        // Level 1 should be empty
        let neighbors = storage.read_neighbors(0, 1).unwrap();
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_memory_storage_overwrite() {
        let mut storage = MemoryStorage::new(8);

        // Write neighbors
        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        // Overwrite
        storage.write_neighbors(0, 0, &[4, 5]).unwrap();

        // Should have new values
        assert_eq!(storage.read_neighbors(0, 0).unwrap(), vec![4, 5]);
    }

    #[test]
    fn test_memory_storage_multiple_nodes() {
        let mut storage = MemoryStorage::new(8);

        // Write to multiple nodes
        storage.write_neighbors(0, 0, &[1, 2]).unwrap();
        storage.write_neighbors(1, 0, &[0, 2]).unwrap();
        storage.write_neighbors(2, 0, &[0, 1]).unwrap();

        // Read back
        assert_eq!(storage.read_neighbors(0, 0).unwrap(), vec![1, 2]);
        assert_eq!(storage.read_neighbors(1, 0).unwrap(), vec![0, 2]);
        assert_eq!(storage.read_neighbors(2, 0).unwrap(), vec![0, 1]);

        assert_eq!(storage.len(), 3);
    }

    #[test]
    fn test_memory_storage_memory_usage() {
        let mut storage = MemoryStorage::new(8);

        let initial_usage = storage.memory_usage();
        assert_eq!(initial_usage, 0); // Empty storage

        // Add some data
        storage.write_neighbors(0, 0, &[1, 2, 3, 4, 5]).unwrap();

        let after_usage = storage.memory_usage();
        assert!(after_usage > initial_usage);
    }

    #[test]
    fn test_node_storage_trait() {
        // Test that MemoryStorage implements NodeStorage trait
        let mut storage: Box<dyn NodeStorage> = Box::new(MemoryStorage::new(8));

        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        let neighbors = storage.read_neighbors(0, 0).unwrap();

        assert_eq!(neighbors, vec![1, 2, 3]);
        assert!(storage.exists(0));
        assert_eq!(storage.len(), 1);
    }
}
