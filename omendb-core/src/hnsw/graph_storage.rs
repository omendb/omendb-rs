//! Graph storage abstraction for HNSW index
//!
//! Provides a unified API over different storage backends:
//! - Memory: Fast in-memory storage (`NeighborLists`)
//! - Layered: Hybrid disk+cache storage (`LayeredStorage`)
//!
//! This enum dispatch pattern allows:
//! - Serialization support (Memory mode)
//! - Type-safe dispatch (no runtime errors)
//! - Backward compatibility (existing API works)
//! - Clean abstraction (easy to understand)

use super::error::Result;
use super::layered_storage::LayeredStorage;
use super::node_storage::NodeStorage;
use super::storage::NeighborLists;
use super::storage_tiering::StorageMode;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use tracing::warn;

/// Configuration for disk-backed storage
///
/// Used when creating a `LayeredStorage` with `DiskStorage` backend
/// for layer 0 (hybrid or disk-heavy mode).
#[derive(Debug, Clone)]
pub struct DiskConfig {
    /// Path to disk storage file for layer 0
    pub path: PathBuf,

    /// LRU cache capacity (number of nodes to cache)
    ///
    /// Typical values:
    /// - 10% of nodes: Minimal memory, higher latency
    /// - 30% of nodes: Balanced (recommended)
    /// - 50% of nodes: Lower latency, higher memory
    pub cache_capacity: NonZeroUsize,

    /// Pre-fault mmap pages (populate flag)
    ///
    /// - `true`: Pre-load all pages into memory (faster first access, slower startup)
    /// - `false`: Lazy load pages (faster startup, slower first access)
    ///
    /// Recommended: `false` for large files, let OS handle paging
    pub populate: bool,
}

impl DiskConfig {
    /// Create disk config with default cache capacity (30% of nodes)
    #[must_use]
    pub fn new(path: PathBuf, num_nodes: usize) -> Self {
        let cache_capacity = (num_nodes as f64 * 0.3).max(1000.0) as usize;
        Self {
            path,
            cache_capacity: NonZeroUsize::new(cache_capacity).unwrap(),
            populate: false,
        }
    }

    /// Create disk config with custom cache capacity
    #[must_use]
    pub fn with_cache(path: PathBuf, cache_capacity: NonZeroUsize) -> Self {
        Self {
            path,
            cache_capacity,
            populate: false,
        }
    }

    /// Enable mmap populate (pre-fault pages)
    #[must_use]
    pub fn with_populate(mut self) -> Self {
        self.populate = true;
        self
    }
}

/// Graph storage backend for HNSW index
///
/// Dispatches to appropriate storage implementation based on mode.
///
/// **Note**: Not Clone due to `LayeredStorage` containing trait objects.
/// Use persistence APIs (save/load) instead of cloning.
#[derive(Debug)]
pub enum GraphStorage {
    /// In-memory storage (current behavior, serializable)
    ///
    /// - Fast access (no I/O)
    /// - Fully serializable
    /// - Suitable for <10M vectors
    Memory(NeighborLists),

    /// Layered storage (hybrid/disk-heavy mode, not serializable)
    ///
    /// - Layer 0: Mode-dependent (disk + cache)
    /// - Layers 1-N: Always in memory
    /// - Suitable for 10M-1B vectors
    /// - **Not serializable** (use `save_to_disk()` instead)
    Layered(Box<LayeredStorage>),
}

impl GraphStorage {
    /// Create memory storage (default)
    #[must_use]
    pub fn new_memory(max_levels: usize) -> Self {
        Self::Memory(NeighborLists::new(max_levels))
    }

    /// Create memory storage with capacity
    #[must_use]
    pub fn new_memory_with_capacity(num_nodes: usize, max_levels: usize, m: usize) -> Self {
        Self::Memory(NeighborLists::with_capacity(num_nodes, max_levels, m))
    }

    /// Create layered storage (hybrid/disk-heavy mode)
    ///
    /// **Note**: Uses `MemoryStorage` for layer 0 (architectural validation).
    /// For actual disk storage, use `new_layered_with_disk()`.
    #[must_use]
    pub fn new_layered(max_levels: usize) -> Self {
        Self::Layered(Box::new(LayeredStorage::new_memory(max_levels)))
    }

    /// Create layered storage with disk-backed layer 0
    ///
    /// Layer 0 stored on disk with LRU cache, upper layers in memory.
    ///
    /// # Example
    /// ```ignore
    /// use std::path::PathBuf;
    /// use std::num::NonZeroUsize;
    ///
    /// let config = DiskConfig::new(
    ///     PathBuf::from("/tmp/graph.bin"),
    ///     10_000_000, // 10M nodes
    /// );
    ///
    /// let storage = GraphStorage::new_layered_with_disk(config, 8)?;
    /// ```
    ///
    /// # Errors
    /// - File I/O errors
    /// - Invalid disk path
    /// - mmap failures
    pub fn new_layered_with_disk(disk_config: &DiskConfig, max_levels: usize) -> Result<Self> {
        let layered = LayeredStorage::new_hybrid(
            &disk_config.path,
            disk_config.cache_capacity,
            max_levels,
            disk_config.populate,
        )?;
        Ok(Self::Layered(Box::new(layered)))
    }

    /// Create from storage mode
    ///
    /// **Note**: Without disk config, Hybrid/DiskHeavy modes use `MemoryStorage`.
    /// For actual disk storage, use `from_mode_with_disk()`.
    #[must_use]
    pub fn from_mode(mode: StorageMode, max_levels: usize) -> Self {
        match mode {
            StorageMode::Memory => Self::new_memory(max_levels),
            StorageMode::Hybrid | StorageMode::DiskHeavy => Self::new_layered(max_levels),
        }
    }

    /// Create from storage mode with optional disk config
    ///
    /// If `disk_config` is Some and mode is Hybrid/DiskHeavy, creates disk-backed storage.
    /// Otherwise falls back to `from_mode()`.
    pub fn from_mode_with_disk(
        mode: StorageMode,
        max_levels: usize,
        disk_config: Option<DiskConfig>,
    ) -> Result<Self> {
        match (mode, disk_config) {
            (StorageMode::Memory, _) => Ok(Self::new_memory(max_levels)),
            (StorageMode::Hybrid | StorageMode::DiskHeavy, Some(ref config)) => {
                Self::new_layered_with_disk(config, max_levels)
            }
            (StorageMode::Hybrid | StorageMode::DiskHeavy, None) => {
                Ok(Self::new_layered(max_levels))
            }
        }
    }

    /// Get storage mode
    #[must_use]
    pub fn mode(&self) -> StorageMode {
        match self {
            Self::Memory(_) => StorageMode::Memory,
            Self::Layered(storage) => storage.mode(),
        }
    }

    /// Get neighbors for a node at a specific level
    ///
    /// # Returns
    /// Owned Vec (allows both memory and disk backends)
    ///
    /// # Errors
    /// - `NodeNotFound`: Node doesn't exist
    /// - `InvalidLevel`: Level exceeds `max_levels`
    /// - Storage errors from disk I/O
    pub fn get_neighbors(&self, node_id: u32, level: u8) -> Result<Vec<u32>> {
        match self {
            Self::Memory(lists) => Ok(lists.get_neighbors(node_id, level)),
            Self::Layered(storage) => storage.read_neighbors(node_id, level),
        }
    }

    /// Execute a closure with read access to neighbors (zero-copy for Memory mode)
    ///
    /// This avoids cloning the neighbor list when you only need to iterate.
    /// Critical for search performance in parallel workloads.
    ///
    /// # Returns
    /// The result of the closure
    pub fn with_neighbors<F, R>(&self, node_id: u32, level: u8, f: F) -> Result<R>
    where
        F: FnOnce(&[u32]) -> R,
    {
        match self {
            Self::Memory(lists) => Ok(lists.with_neighbors(node_id, level, f)),
            Self::Layered(storage) => {
                // Layered mode still needs to clone (disk I/O)
                let neighbors = storage.read_neighbors(node_id, level)?;
                Ok(f(&neighbors))
            }
        }
    }

    /// Set neighbors for a node at a specific level
    ///
    /// # Errors
    /// - `InvalidLevel`: Level exceeds `max_levels`
    /// - Storage errors from disk I/O
    pub fn set_neighbors(&mut self, node_id: u32, level: u8, neighbors: Vec<u32>) -> Result<()> {
        match self {
            Self::Memory(lists) => {
                lists.set_neighbors(node_id, level, neighbors);
                Ok(())
            }
            Self::Layered(storage) => storage.write_neighbors(node_id, level, &neighbors),
        }
    }

    /// Add a bidirectional link between two nodes at a level
    ///
    /// Helper method that:
    /// 1. Reads neighbor lists for both nodes
    /// 2. Adds bidirectional links (if not present)
    /// 3. Writes back updated lists
    ///
    /// # Errors
    /// - Storage errors from read/write operations
    pub fn add_bidirectional_link(&mut self, node_a: u32, node_b: u32, level: u8) -> Result<()> {
        match self {
            Self::Memory(lists) => {
                // Fast path: use NeighborLists::add_bidirectional_link (no allocation)
                lists.add_bidirectional_link(node_a, node_b, level);
                Ok(())
            }
            Self::Layered(_) => {
                // Slow path: read, modify, write (allocates)
                let mut neighbors_a = self.get_neighbors(node_a, level)?;
                let mut neighbors_b = self.get_neighbors(node_b, level)?;

                // Add bidirectional links (if not present)
                if !neighbors_a.contains(&node_b) {
                    neighbors_a.push(node_b);
                }
                if !neighbors_b.contains(&node_a) {
                    neighbors_b.push(node_a);
                }

                // Write back
                self.set_neighbors(node_a, level, neighbors_a)?;
                self.set_neighbors(node_b, level, neighbors_b)?;

                Ok(())
            }
        }
    }

    /// Add bidirectional link (parallel version - assumes nodes pre-allocated)
    ///
    /// Thread-safe version for parallel graph construction.
    /// Only works with Memory mode. Returns false if called on Layered mode.
    #[allow(clippy::must_use_candidate)]
    pub fn add_bidirectional_link_parallel(&self, node_a: u32, node_b: u32, level: u8) -> bool {
        match self {
            Self::Memory(lists) => {
                lists.add_bidirectional_link_parallel(node_a, node_b, level);
                true
            }
            Self::Layered(_) => false,
        }
    }

    /// Remove unidirectional link (parallel version - assumes nodes pre-allocated)
    ///
    /// Removes link from `node_a` to `node_b` (NOT bidirectional).
    /// Thread-safe version for parallel graph construction.
    /// Only works with Memory mode. Returns false if called on Layered mode.
    #[allow(clippy::must_use_candidate)]
    pub fn remove_link_parallel(&self, node_a: u32, node_b: u32, level: u8) -> bool {
        match self {
            Self::Memory(lists) => {
                lists.remove_link_parallel(node_a, node_b, level);
                true
            }
            Self::Layered(_) => false,
        }
    }

    /// Set neighbors (parallel version - assumes node pre-allocated)
    ///
    /// Thread-safe version for parallel graph construction.
    /// Only works with Memory mode. Returns false if called on Layered mode.
    #[allow(clippy::must_use_candidate)]
    pub fn set_neighbors_parallel(&self, node_id: u32, level: u8, neighbors: Vec<u32>) -> bool {
        match self {
            Self::Memory(lists) => {
                lists.set_neighbors_parallel(node_id, level, neighbors);
                true
            }
            Self::Layered(_) => false,
        }
    }

    /// Get `M_max` (max neighbors)
    #[must_use]
    pub fn m_max(&self) -> usize {
        match self {
            Self::Memory(lists) => lists.m_max(),
            Self::Layered(_) => 32, // Default M*2 = 16*2
        }
    }

    /// Check if storage is in memory mode
    #[must_use]
    pub fn is_memory_mode(&self) -> bool {
        matches!(self, Self::Memory(_))
    }

    /// Check if storage is in layered mode
    #[must_use]
    pub fn is_layered_mode(&self) -> bool {
        matches!(self, Self::Layered(_))
    }

    /// Get memory usage in bytes (approximate)
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Memory(lists) => lists.memory_usage(),
            Self::Layered(_) => {
                // LayeredStorage memory is dominated by mmap (not counted in RSS)
                0
            }
        }
    }

    /// Reorder graph nodes using BFS for cache locality
    ///
    /// Returns a mapping from `old_id` -> `new_id`
    ///
    /// **Note**: Only supported in Memory mode. Returns identity mapping for Layered mode.
    pub fn reorder_bfs(&mut self, entry_point: u32, start_level: u8) -> Vec<u32> {
        match self {
            Self::Memory(lists) => lists.reorder_bfs(entry_point, start_level),
            Self::Layered(_) => {
                // Reordering not supported for layered storage
                // Return identity mapping (no reordering)
                warn!("BFS reordering not supported for layered storage, skipping");
                vec![]
            }
        }
    }
}

// Custom serialization: only Memory mode is serializable
impl Serialize for GraphStorage {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Memory(lists) => lists.serialize(serializer),
            Self::Layered(_) => Err(serde::ser::Error::custom(
                "LayeredStorage cannot be serialized directly. Use save_to_disk() instead.",
            )),
        }
    }
}

// Custom deserialization: always deserializes as Memory mode
impl<'de> Deserialize<'de> for GraphStorage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let lists = NeighborLists::deserialize(deserializer)?;
        Ok(Self::Memory(lists))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_storage_new_memory() {
        let storage = GraphStorage::new_memory(8);
        assert!(storage.is_memory_mode());
        assert!(!storage.is_layered_mode());
        assert_eq!(storage.mode(), StorageMode::Memory);
    }

    #[test]
    fn test_graph_storage_new_layered() {
        let storage = GraphStorage::new_layered(8);
        assert!(storage.is_layered_mode());
        assert!(!storage.is_memory_mode());
        assert_eq!(storage.mode(), StorageMode::Memory); // LayeredStorage starts as Memory internally
    }

    #[test]
    fn test_graph_storage_from_mode() {
        let memory = GraphStorage::from_mode(StorageMode::Memory, 8);
        assert!(memory.is_memory_mode());

        let hybrid = GraphStorage::from_mode(StorageMode::Hybrid, 8);
        assert!(hybrid.is_layered_mode());

        let disk_heavy = GraphStorage::from_mode(StorageMode::DiskHeavy, 8);
        assert!(disk_heavy.is_layered_mode());
    }

    #[test]
    fn test_graph_storage_get_set_neighbors_memory() {
        let mut storage = GraphStorage::new_memory(8);

        // Set neighbors
        storage.set_neighbors(0, 0, vec![1, 2, 3]).unwrap();
        storage.set_neighbors(0, 1, vec![4, 5]).unwrap();

        // Get neighbors
        assert_eq!(storage.get_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(0, 1).unwrap(), vec![4, 5]);

        // Empty neighbors
        assert_eq!(storage.get_neighbors(99, 0).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn test_graph_storage_get_set_neighbors_layered() {
        let mut storage = GraphStorage::new_layered(8);

        // Set neighbors
        storage.set_neighbors(0, 0, vec![1, 2, 3]).unwrap();
        storage.set_neighbors(0, 1, vec![4, 5]).unwrap();

        // Get neighbors
        assert_eq!(storage.get_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(storage.get_neighbors(0, 1).unwrap(), vec![4, 5]);

        // Empty neighbors
        assert_eq!(storage.get_neighbors(99, 0).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn test_graph_storage_add_bidirectional_link_memory() {
        let mut storage = GraphStorage::new_memory(8);

        // Add bidirectional link
        storage.add_bidirectional_link(0, 1, 0).unwrap();

        // Verify both directions
        let neighbors_0 = storage.get_neighbors(0, 0).unwrap();
        let neighbors_1 = storage.get_neighbors(1, 0).unwrap();

        assert!(neighbors_0.contains(&1));
        assert!(neighbors_1.contains(&0));
    }

    #[test]
    fn test_graph_storage_add_bidirectional_link_layered() {
        let mut storage = GraphStorage::new_layered(8);

        // Add bidirectional link
        storage.add_bidirectional_link(0, 1, 0).unwrap();

        // Verify both directions
        let neighbors_0 = storage.get_neighbors(0, 0).unwrap();
        let neighbors_1 = storage.get_neighbors(1, 0).unwrap();

        assert!(neighbors_0.contains(&1));
        assert!(neighbors_1.contains(&0));
    }

    #[test]
    fn test_graph_storage_add_bidirectional_link_idempotent() {
        let mut storage = GraphStorage::new_memory(8);

        // Add same link twice
        storage.add_bidirectional_link(0, 1, 0).unwrap();
        storage.add_bidirectional_link(0, 1, 0).unwrap();

        // Should only appear once
        let neighbors_0 = storage.get_neighbors(0, 0).unwrap();
        assert_eq!(neighbors_0.iter().filter(|&&n| n == 1).count(), 1);
    }

    #[test]
    fn test_graph_storage_m_max() {
        let memory = GraphStorage::new_memory(8);
        assert_eq!(memory.m_max(), 32); // Default M*2

        let layered = GraphStorage::new_layered(8);
        assert_eq!(layered.m_max(), 32); // Default M*2
    }

    #[test]
    fn test_graph_storage_serialization_memory() {
        let mut storage = GraphStorage::new_memory(8);
        storage.set_neighbors(0, 0, vec![1, 2, 3]).unwrap();

        // Serialize
        let serialized = bincode::serialize(&storage).unwrap();

        // Deserialize
        let deserialized: GraphStorage = bincode::deserialize(&serialized).unwrap();

        // Verify
        assert!(deserialized.is_memory_mode());
        assert_eq!(deserialized.get_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_graph_storage_serialization_layered_fails() {
        let storage = GraphStorage::new_layered(8);

        // Serialization should fail
        let result = bincode::serialize(&storage);
        assert!(result.is_err());
    }

    // DiskConfig tests (basic configuration)

    #[test]
    fn test_disk_config_new() {
        let config = DiskConfig::new(PathBuf::from("/tmp/test.bin"), 10_000_000);

        // 30% of 10M = 3M cache capacity
        assert_eq!(config.cache_capacity.get(), 3_000_000);
        assert!(!config.populate);
        assert_eq!(config.path, PathBuf::from("/tmp/test.bin"));
    }

    #[test]
    fn test_disk_config_with_cache() {
        let config = DiskConfig::with_cache(
            PathBuf::from("/tmp/test.bin"),
            NonZeroUsize::new(1_000_000).unwrap(),
        );

        assert_eq!(config.cache_capacity.get(), 1_000_000);
        assert!(!config.populate);
    }

    #[test]
    fn test_disk_config_with_populate() {
        let config = DiskConfig::new(PathBuf::from("/tmp/test.bin"), 10_000).with_populate();

        assert!(config.populate);
    }

    #[test]
    fn test_from_mode_with_disk_memory() {
        let result = GraphStorage::from_mode_with_disk(StorageMode::Memory, 8, None).unwrap();

        // Memory mode ignores disk config
        assert!(result.is_memory_mode());
    }

    #[test]
    fn test_from_mode_with_disk_hybrid_no_config() {
        let result = GraphStorage::from_mode_with_disk(StorageMode::Hybrid, 8, None).unwrap();

        // Falls back to in-memory layered
        assert!(result.is_layered_mode());
        assert_eq!(result.mode(), StorageMode::Memory); // LayeredStorage with MemoryStorage
    }

    // DiskStorage is read-only by design:
    // 1. Build index in MemoryStorage
    // 2. Save to disk with DiskStorage::create()
    // 3. Load from disk with DiskStorage::open() for queries
    // 4. Query through CachedStorage wrapper
}
