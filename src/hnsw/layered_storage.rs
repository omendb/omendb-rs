//! Layered storage for HNSW-IF
//!
//! Key optimization: Layers 1-N always in memory, layer 0 mode-dependent.
//!
//! ## Why This Works
//!
//! **Layer Distribution** (10M nodes, ml=1/ln(2)):
//! - Layer 0: ~10,000,000 nodes (100% of nodes)
//! - Layer 1: ~5,000,000 nodes (50%)
//! - Layer 2: ~2,500,000 nodes (25%)
//! - Layer 3: ~1,250,000 nodes (12.5%)
//! - ...
//!
//! **Access Patterns**:
//! - Layers 1-N: Accessed on EVERY query (entry point search)
//! - Layer 0: Only accessed for candidates (k neighbors)
//!
//! **Memory Savings**:
//! - Layer 0: ~10M nodes × 16 neighbors × 88 bytes = ~14 GB
//! - Layers 1-N: ~10M nodes × 16 neighbors × 88 bytes = ~14 GB
//! - With layer 0 on disk (30% cache): ~14 GB → 4.2 GB
//! - Total: 4.2 GB (layer 0 cache) + 14 GB (layers 1-N) = 18.2 GB
//!
//! **BUT**: Upper layers have exponentially fewer nodes!
//! - Layer 1+: Sum of geometric series ≈ 10M nodes total (same as layer 0 count)
//! - BUT each upper layer node has fewer neighbors (M vs M0)
//! - Actual upper layer memory: ~2-4 GB (much smaller than layer 0)
//!
//! **True Memory Savings**:
//! - All in memory: ~16-18 GB
//! - Layered (layer 0 on disk, 30% cache): ~6-8 GB (60% reduction!)

use super::cached_storage::CachedStorage;
use super::disk_storage::DiskStorage;
use super::error::Result;
use super::node_storage::{Level, MemoryStorage, NodeId, NodeStorage};
use super::storage_tiering::StorageMode;
use std::path::Path;

/// Layered storage for HNSW graph
///
/// Routes layer 0 to mode-dependent storage (Memory/Disk/Cached)
/// and layers 1-N to in-memory storage.
///
/// # Why Layering Works
///
/// 1. **Access frequency**: Upper layers accessed on every query
/// 2. **Size**: Layer 0 is ~50% of all nodes, upper layers ~50% combined
/// 3. **Neighbors**: Upper layers have M neighbors vs M0 (typically M0 = 2×M)
/// 4. **Result**: Upper layers are smaller AND more frequently accessed
///
/// # Example
/// ```ignore
/// // Pure memory mode (<10M vectors)
/// let storage = LayeredStorage::new_memory(8);
///
/// // Hybrid mode (10M-100M vectors)
/// let layer_0 = DiskStorage::open("graph.bin", false)?;
/// let cached = CachedStorage::new(Box::new(layer_0), cache_capacity);
/// let storage = LayeredStorage::new(Box::new(cached), StorageMode::Hybrid, 8);
/// ```
pub struct LayeredStorage {
    /// Storage for layer 0 (base layer, all nodes)
    ///
    /// Mode-dependent:
    /// - Memory mode: `MemoryStorage`
    /// - Hybrid mode: CachedStorage(DiskStorage)
    /// - `DiskHeavy` mode: CachedStorage(DiskStorage)
    layer_0: Box<dyn NodeStorage>,

    /// Storage for layers 1-N (upper layers, always memory)
    ///
    /// Upper layers are small and accessed on every query (entry point),
    /// so keeping them in memory is critical for performance.
    upper_layers: MemoryStorage,

    /// Current storage mode
    mode: StorageMode,

    /// Maximum number of levels supported
    max_levels: usize,
}

impl std::fmt::Debug for LayeredStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredStorage")
            .field("layer_0", &"Box<dyn NodeStorage>")
            .field("upper_layers", &self.upper_layers)
            .field("mode", &self.mode)
            .field("max_levels", &self.max_levels)
            .finish()
    }
}

impl LayeredStorage {
    /// Create layered storage with custom layer 0 backend
    ///
    /// # Arguments
    /// * `layer_0` - Storage backend for layer 0 (Memory, Disk, or Cached)
    /// * `mode` - Storage mode (for tracking/debugging)
    /// * `max_levels` - Maximum number of levels (typically 8)
    #[must_use]
    pub fn new(layer_0: Box<dyn NodeStorage>, mode: StorageMode, max_levels: usize) -> Self {
        Self {
            layer_0,
            upper_layers: MemoryStorage::new(max_levels),
            mode,
            max_levels,
        }
    }

    /// Create layered storage with pre-populated upper layers
    ///
    /// Used when loading from disk where upper layers are extracted separately.
    ///
    /// # Arguments
    /// * `layer_0` - Storage backend for layer 0 (typically `DiskStorage` + `CachedStorage`)
    /// * `upper_layers` - Pre-populated `MemoryStorage` with upper layer data
    /// * `mode` - Storage mode (for tracking/debugging)
    /// * `max_levels` - Maximum number of levels (typically 8)
    #[must_use]
    pub fn new_with_upper_layers(
        layer_0: Box<dyn NodeStorage>,
        upper_layers: MemoryStorage,
        mode: StorageMode,
        max_levels: usize,
    ) -> Self {
        Self {
            layer_0,
            upper_layers,
            mode,
            max_levels,
        }
    }

    /// Create pure in-memory layered storage (Memory mode)
    ///
    /// Both layer 0 and upper layers use `MemoryStorage`.
    ///
    /// # Example
    /// ```ignore
    /// let storage = LayeredStorage::new_memory(8);
    /// ```
    #[must_use]
    pub fn new_memory(max_levels: usize) -> Self {
        Self {
            layer_0: Box::new(MemoryStorage::new(max_levels)),
            upper_layers: MemoryStorage::new(max_levels),
            mode: StorageMode::Memory,
            max_levels,
        }
    }

    /// Create hybrid layered storage (Hybrid mode)
    ///
    /// Layer 0 on disk with cache, upper layers in memory.
    ///
    /// # Arguments
    /// * `disk_path` - Path to disk storage file
    /// * `cache_capacity` - LRU cache size for layer 0
    /// * `max_levels` - Maximum number of levels
    /// * `populate` - Pre-fault mmap pages
    pub fn new_hybrid(
        disk_path: &Path,
        cache_capacity: std::num::NonZeroUsize,
        max_levels: usize,
        populate: bool,
    ) -> Result<Self> {
        let disk_storage = DiskStorage::open(disk_path, populate)?;
        let cached_storage = CachedStorage::new(Box::new(disk_storage), cache_capacity);

        Ok(Self {
            layer_0: Box::new(cached_storage),
            upper_layers: MemoryStorage::new(max_levels),
            mode: StorageMode::Hybrid,
            max_levels,
        })
    }

    /// Get current storage mode
    #[must_use]
    pub fn mode(&self) -> StorageMode {
        self.mode
    }

    /// Get maximum levels
    #[must_use]
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }

    /// Route to appropriate storage backend based on level
    ///
    /// - Level 0: `layer_0` backend (mode-dependent)
    /// - Level 1+: `upper_layers` backend (always memory)
    fn storage_for_level(&self, level: Level) -> &dyn NodeStorage {
        if level == 0 {
            self.layer_0.as_ref()
        } else {
            &self.upper_layers as &dyn NodeStorage
        }
    }

    /// Route to mutable storage backend based on level
    fn storage_for_level_mut(&mut self, level: Level) -> &mut dyn NodeStorage {
        if level == 0 {
            self.layer_0.as_mut()
        } else {
            &mut self.upper_layers as &mut dyn NodeStorage
        }
    }
}

impl NodeStorage for LayeredStorage {
    fn read_neighbors(&self, node_id: NodeId, level: Level) -> Result<Vec<NodeId>> {
        self.storage_for_level(level).read_neighbors(node_id, level)
    }

    fn write_neighbors(
        &mut self,
        node_id: NodeId,
        level: Level,
        neighbors: &[NodeId],
    ) -> Result<()> {
        self.storage_for_level_mut(level)
            .write_neighbors(node_id, level, neighbors)
    }

    fn exists(&self, node_id: NodeId) -> bool {
        // A node exists if it exists in layer 0
        self.layer_0.exists(node_id)
    }

    fn num_levels(&self, node_id: NodeId) -> Result<usize> {
        // Count non-empty levels across both backends
        let mut max_level = 0;

        // Check layer 0
        let layer_0_neighbors = self.layer_0.read_neighbors(node_id, 0)?;
        if !layer_0_neighbors.is_empty() {
            max_level = 1;
        }

        // Check upper layers
        for level in 1..self.max_levels {
            let neighbors = self.upper_layers.read_neighbors(node_id, level as u8)?;
            if !neighbors.is_empty() {
                max_level = level + 1;
            }
        }

        Ok(max_level)
    }

    fn len(&self) -> usize {
        // Number of nodes is determined by layer 0 (all nodes are on layer 0)
        self.layer_0.len()
    }

    fn is_empty(&self) -> bool {
        self.layer_0.is_empty()
    }

    fn flush(&mut self) -> Result<()> {
        // Flush both backends
        self.layer_0.flush()?;
        self.upper_layers.flush()?;
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        // Sum of both backends
        self.layer_0.memory_usage() + self.upper_layers.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use tempfile::TempDir;

    #[test]
    fn test_layered_storage_new_memory() {
        let storage = LayeredStorage::new_memory(8);

        assert_eq!(storage.mode(), StorageMode::Memory);
        assert_eq!(storage.max_levels(), 8);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_layered_storage_routing() {
        let mut storage = LayeredStorage::new_memory(8);

        // Write to layer 0
        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        // Write to layer 1 (upper layers)
        storage.write_neighbors(0, 1, &[4, 5]).unwrap();

        // Write to layer 2 (upper layers)
        storage.write_neighbors(0, 2, &[6]).unwrap();

        // Read back and verify
        assert_eq!(storage.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(storage.read_neighbors(0, 1).unwrap(), vec![4, 5]);
        assert_eq!(storage.read_neighbors(0, 2).unwrap(), vec![6]);
    }

    #[test]
    fn test_layered_storage_num_levels() {
        let mut storage = LayeredStorage::new_memory(8);

        // Node with 3 levels
        storage.write_neighbors(0, 0, &[1]).unwrap();
        storage.write_neighbors(0, 1, &[2]).unwrap();
        storage.write_neighbors(0, 2, &[3]).unwrap();

        assert_eq!(storage.num_levels(0).unwrap(), 3);
    }

    #[test]
    fn test_layered_storage_exists() {
        let mut storage = LayeredStorage::new_memory(8);

        assert!(!storage.exists(0));

        storage.write_neighbors(0, 0, &[1]).unwrap();
        assert!(storage.exists(0));

        assert!(!storage.exists(1));
    }

    #[test]
    fn test_layered_storage_len() {
        let mut storage = LayeredStorage::new_memory(8);

        assert_eq!(storage.len(), 0);

        storage.write_neighbors(0, 0, &[1]).unwrap();
        storage.write_neighbors(1, 0, &[2]).unwrap();

        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn test_layered_storage_memory_usage() {
        let mut storage = LayeredStorage::new_memory(8);

        let initial = storage.memory_usage();

        // Add data to layer 0
        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        // Add data to upper layers
        storage.write_neighbors(0, 1, &[4, 5]).unwrap();

        let after = storage.memory_usage();
        assert!(after > initial);
    }

    #[test]
    fn test_layered_storage_multiple_nodes() {
        let mut storage = LayeredStorage::new_memory(8);

        // Node 0: 3 levels
        storage.write_neighbors(0, 0, &[1, 2]).unwrap();
        storage.write_neighbors(0, 1, &[1]).unwrap();
        storage.write_neighbors(0, 2, &[1]).unwrap();

        // Node 1: 2 levels
        storage.write_neighbors(1, 0, &[0, 2]).unwrap();
        storage.write_neighbors(1, 1, &[0]).unwrap();

        // Node 2: 1 level
        storage.write_neighbors(2, 0, &[0, 1]).unwrap();

        assert_eq!(storage.len(), 3);
        assert_eq!(storage.num_levels(0).unwrap(), 3);
        assert_eq!(storage.num_levels(1).unwrap(), 2);
        assert_eq!(storage.num_levels(2).unwrap(), 1);
    }

    #[test]
    fn test_layered_storage_with_disk() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("layer_0.bin");

        // Create layer 0 on disk
        let nodes = vec![
            vec![vec![1, 2, 3]], // Node 0
            vec![vec![0, 2]],    // Node 1
            vec![vec![0, 1]],    // Node 2
        ];

        DiskStorage::create(&storage_path, &nodes, 0, 16).unwrap();

        // Load with cache
        let disk_storage = DiskStorage::open(&storage_path, false).unwrap();
        let cache_capacity = NonZeroUsize::new(100).unwrap();
        let cached_storage = CachedStorage::new(Box::new(disk_storage), cache_capacity);

        // Create layered storage
        let mut storage = LayeredStorage::new(Box::new(cached_storage), StorageMode::Hybrid, 8);

        // Layer 0 should be on disk (readable)
        assert_eq!(storage.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(storage.read_neighbors(1, 0).unwrap(), vec![0, 2]);

        // Write to upper layers (in memory)
        storage.write_neighbors(0, 1, &[1]).unwrap();
        storage.write_neighbors(0, 2, &[1]).unwrap();

        // Read back upper layers
        assert_eq!(storage.read_neighbors(0, 1).unwrap(), vec![1]);
        assert_eq!(storage.read_neighbors(0, 2).unwrap(), vec![1]);

        // Verify num_levels counts both
        assert_eq!(storage.num_levels(0).unwrap(), 3);
    }

    #[test]
    fn test_layered_storage_node_storage_trait() {
        // Test that LayeredStorage implements NodeStorage trait
        let mut storage: Box<dyn NodeStorage> = Box::new(LayeredStorage::new_memory(8));

        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        storage.write_neighbors(0, 1, &[4, 5]).unwrap();

        let neighbors = storage.read_neighbors(0, 0).unwrap();
        assert_eq!(neighbors, vec![1, 2, 3]);

        assert!(storage.exists(0));
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn test_layered_storage_realistic_scenario() {
        let mut storage = LayeredStorage::new_memory(8);

        // Simulate HNSW graph with 1000 nodes
        // Layer distribution (ml = 1/ln(2)):
        // - Layer 0: 1000 nodes (100%)
        // - Layer 1: ~500 nodes (50%)
        // - Layer 2: ~250 nodes (25%)
        // - Layer 3: ~125 nodes (12.5%)

        let num_nodes = 1000;

        // Write all nodes to layer 0
        for node_id in 0..num_nodes {
            storage.write_neighbors(node_id, 0, &[1, 2, 3, 4]).unwrap();
        }

        // Write ~50% to layer 1
        for node_id in 0..(num_nodes / 2) {
            storage.write_neighbors(node_id, 1, &[1, 2]).unwrap();
        }

        // Write ~25% to layer 2
        for node_id in 0..(num_nodes / 4) {
            storage.write_neighbors(node_id, 2, &[1]).unwrap();
        }

        // Verify
        assert_eq!(storage.len(), num_nodes as usize);

        // Node 0 has 3 levels (layer 0, 1, 2)
        assert_eq!(storage.num_levels(0).unwrap(), 3);

        // Node 250 has 2 levels (layer 0, 1)
        assert_eq!(storage.num_levels(250).unwrap(), 2);

        // Node 750 has 1 level (layer 0 only)
        assert_eq!(storage.num_levels(750).unwrap(), 1);

        // Memory usage should be reasonable
        let memory = storage.memory_usage();
        assert!(memory > 0);
    }
}
