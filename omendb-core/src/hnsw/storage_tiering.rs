//! Automatic storage tiering for HNSW-IF
//!
//! Automatically selects storage mode based on dataset size:
//! - <10M vectors: Pure memory (`MemoryStorage`)
//! - 10M-100M: Hybrid (`CachedStorage` + `DiskStorage`, 30% cache)
//! - 100M+: Disk-heavy (`CachedStorage` + `DiskStorage`, 10% cache)
//!
//! Key insight from Gorgeous 2025: Cache adjacency lists only (not vectors)
//! gives 88% hit rate vs 10% for traditional caching.

use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// Storage mode for HNSW graph
///
/// Automatically selected based on number of vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageMode {
    /// Pure in-memory storage (<10M vectors)
    ///
    /// - Graph stored in `MemoryStorage`
    /// - Fast writes and reads
    /// - No disk I/O overhead
    /// - Memory: ~50 bytes per neighbor × `num_nodes` × `avg_levels`
    Memory,

    /// Hybrid mode (10M-100M vectors)
    ///
    /// - Graph stored on disk (`DiskStorage` with mmap)
    /// - 30% of adjacency lists cached in LRU cache
    /// - Expected cache hit rate: 70-80%
    /// - Memory: ~30% of graph size
    Hybrid,

    /// Disk-heavy mode (100M+ vectors)
    ///
    /// - Graph stored on disk (`DiskStorage` with mmap)
    /// - 10% of adjacency lists cached in LRU cache
    /// - Expected cache hit rate: 50-70%
    /// - Memory: ~10% of graph size
    DiskHeavy,
}

impl StorageMode {
    /// Automatically select storage mode based on number of vectors
    ///
    /// # Thresholds
    /// - <10M vectors: Memory mode
    /// - 10M-100M vectors: Hybrid mode (30% cache)
    /// - 100M+ vectors: `DiskHeavy` mode (10% cache)
    ///
    /// # Example
    /// ```ignore
    /// let mode = StorageMode::auto_select(5_000_000);
    /// assert_eq!(mode, StorageMode::Memory);
    ///
    /// let mode = StorageMode::auto_select(50_000_000);
    /// assert_eq!(mode, StorageMode::Hybrid);
    ///
    /// let mode = StorageMode::auto_select(500_000_000);
    /// assert_eq!(mode, StorageMode::DiskHeavy);
    /// ```
    #[must_use]
    pub fn auto_select(num_vectors: usize) -> Self {
        const HYBRID_THRESHOLD: usize = 10_000_000; // 10M
        const DISK_HEAVY_THRESHOLD: usize = 100_000_000; // 100M

        if num_vectors < HYBRID_THRESHOLD {
            StorageMode::Memory
        } else if num_vectors < DISK_HEAVY_THRESHOLD {
            StorageMode::Hybrid
        } else {
            StorageMode::DiskHeavy
        }
    }

    /// Calculate cache capacity for this storage mode
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes in graph
    /// * `avg_levels` - Average number of levels per node (typically 3-4)
    ///
    /// # Returns
    /// - Memory mode: None (no cache needed)
    /// - Hybrid mode: 30% of total adjacency lists
    /// - `DiskHeavy` mode: 10% of total adjacency lists
    ///
    /// # Example
    /// ```ignore
    /// // 10M nodes, 3 avg levels = 30M total adjacency lists
    /// // Hybrid: 30% × 30M = 9M cached entries
    /// let capacity = StorageMode::Hybrid.cache_capacity(10_000_000, 3.0);
    /// assert_eq!(capacity, Some(9_000_000));
    /// ```
    #[must_use]
    pub fn cache_capacity(&self, num_nodes: usize, avg_levels: f64) -> Option<usize> {
        match self {
            StorageMode::Memory => None, // No cache needed (already in memory)
            StorageMode::Hybrid => {
                // 30% cache
                let total_adjacency_lists = (num_nodes as f64 * avg_levels) as usize;
                let capacity = (total_adjacency_lists as f64 * 0.30) as usize;
                Some(capacity.max(1000)) // Minimum 1000 entries
            }
            StorageMode::DiskHeavy => {
                // 10% cache
                let total_adjacency_lists = (num_nodes as f64 * avg_levels) as usize;
                let capacity = (total_adjacency_lists as f64 * 0.10) as usize;
                Some(capacity.max(1000)) // Minimum 1000 entries
            }
        }
    }

    /// Get cache capacity as `NonZeroUsize` (for `LruCache` constructor)
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes in graph
    /// * `avg_levels` - Average number of levels per node
    ///
    /// # Returns
    /// `NonZeroUsize` for cache capacity, or None if Memory mode
    pub fn cache_capacity_nonzero(
        &self,
        num_nodes: usize,
        avg_levels: f64,
    ) -> Option<NonZeroUsize> {
        self.cache_capacity(num_nodes, avg_levels)
            .and_then(NonZeroUsize::new)
    }

    /// Estimate memory usage for this storage mode
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes
    /// * `avg_neighbors` - Average neighbors per level (typically M = 16)
    /// * `avg_levels` - Average levels per node (typically 3-4)
    ///
    /// # Returns
    /// Estimated memory usage in bytes
    ///
    /// # Formula
    /// - Memory per adjacency list: ~58 bytes (16 neighbors × 4 bytes + overhead)
    /// - Memory mode: `num_nodes` × `avg_levels` × 58 bytes
    /// - Hybrid mode: 30% of Memory mode
    /// - `DiskHeavy` mode: 10% of Memory mode
    #[must_use]
    pub fn estimated_memory_bytes(
        &self,
        num_nodes: usize,
        avg_neighbors: usize,
        avg_levels: f64,
    ) -> usize {
        const BYTES_PER_NEIGHBOR: usize = 4; // u32
        const OVERHEAD_PER_LIST: usize = 24; // Vec overhead

        let bytes_per_list = avg_neighbors * BYTES_PER_NEIGHBOR + OVERHEAD_PER_LIST;
        let total_lists = (num_nodes as f64 * avg_levels) as usize;
        let full_memory = total_lists * bytes_per_list;

        match self {
            StorageMode::Memory => full_memory,
            StorageMode::Hybrid => (full_memory as f64 * 0.30) as usize,
            StorageMode::DiskHeavy => (full_memory as f64 * 0.10) as usize,
        }
    }

    /// Get cache percentage for this mode
    #[must_use]
    pub fn cache_percentage(&self) -> Option<f64> {
        match self {
            StorageMode::Memory => None,
            StorageMode::Hybrid => Some(0.30),
            StorageMode::DiskHeavy => Some(0.10),
        }
    }

    /// Check if this mode requires disk storage
    #[must_use]
    pub fn requires_disk(&self) -> bool {
        matches!(self, StorageMode::Hybrid | StorageMode::DiskHeavy)
    }

    /// Check if this mode requires cache
    #[must_use]
    pub fn requires_cache(&self) -> bool {
        matches!(self, StorageMode::Hybrid | StorageMode::DiskHeavy)
    }

    /// Get human-readable description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            StorageMode::Memory => "Pure in-memory storage (<10M vectors)",
            StorageMode::Hybrid => "Hybrid disk+cache (10M-100M vectors, 30% cache)",
            StorageMode::DiskHeavy => "Disk-heavy (100M+ vectors, 10% cache)",
        }
    }
}

/// Storage tiering configuration
///
/// Encapsulates automatic mode selection and cache sizing.
#[derive(Debug, Clone)]
pub struct TieringConfig {
    /// Current storage mode
    pub mode: StorageMode,

    /// Cache capacity (None for Memory mode)
    pub cache_capacity: Option<NonZeroUsize>,

    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,

    /// Number of nodes
    pub num_nodes: usize,

    /// Average levels per node
    pub avg_levels: f64,
}

impl TieringConfig {
    /// Create tiering config with automatic mode selection
    ///
    /// # Arguments
    /// * `num_vectors` - Number of vectors (= `num_nodes`)
    /// * `avg_levels` - Average levels per node (typically 3-4)
    /// * `avg_neighbors` - Average neighbors per level (typically M = 16)
    ///
    /// # Example
    /// ```ignore
    /// let config = TieringConfig::auto(10_000_000, 3.5, 16);
    /// assert_eq!(config.mode, StorageMode::Hybrid);
    /// assert!(config.cache_capacity.is_some());
    /// ```
    #[must_use]
    pub fn auto(num_vectors: usize, avg_levels: f64, avg_neighbors: usize) -> Self {
        let mode = StorageMode::auto_select(num_vectors);
        let cache_capacity = mode.cache_capacity_nonzero(num_vectors, avg_levels);
        let estimated_memory_bytes =
            mode.estimated_memory_bytes(num_vectors, avg_neighbors, avg_levels);

        Self {
            mode,
            cache_capacity,
            estimated_memory_bytes,
            num_nodes: num_vectors,
            avg_levels,
        }
    }

    /// Create tiering config with explicit mode
    #[must_use]
    pub fn with_mode(
        mode: StorageMode,
        num_vectors: usize,
        avg_levels: f64,
        avg_neighbors: usize,
    ) -> Self {
        let cache_capacity = mode.cache_capacity_nonzero(num_vectors, avg_levels);
        let estimated_memory_bytes =
            mode.estimated_memory_bytes(num_vectors, avg_neighbors, avg_levels);

        Self {
            mode,
            cache_capacity,
            estimated_memory_bytes,
            num_nodes: num_vectors,
            avg_levels,
        }
    }

    /// Get memory usage in human-readable format
    #[must_use]
    pub fn memory_usage_human(&self) -> String {
        let bytes = self.estimated_memory_bytes;
        if bytes < 1024 {
            format!("{bytes} bytes")
        } else if bytes < 1024 * 1024 {
            format!("{:.2} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_mode_auto_select() {
        // Memory mode (<10M)
        assert_eq!(StorageMode::auto_select(1_000_000), StorageMode::Memory);
        assert_eq!(StorageMode::auto_select(5_000_000), StorageMode::Memory);
        assert_eq!(StorageMode::auto_select(9_999_999), StorageMode::Memory);

        // Hybrid mode (10M-100M)
        assert_eq!(StorageMode::auto_select(10_000_000), StorageMode::Hybrid);
        assert_eq!(StorageMode::auto_select(50_000_000), StorageMode::Hybrid);
        assert_eq!(StorageMode::auto_select(99_999_999), StorageMode::Hybrid);

        // DiskHeavy mode (100M+)
        assert_eq!(
            StorageMode::auto_select(100_000_000),
            StorageMode::DiskHeavy
        );
        assert_eq!(
            StorageMode::auto_select(500_000_000),
            StorageMode::DiskHeavy
        );
        assert_eq!(
            StorageMode::auto_select(1_000_000_000),
            StorageMode::DiskHeavy
        );
    }

    #[test]
    fn test_cache_capacity() {
        // Memory mode: no cache
        assert_eq!(StorageMode::Memory.cache_capacity(10_000, 3.0), None);

        // Hybrid mode: 30% cache
        let capacity = StorageMode::Hybrid.cache_capacity(10_000_000, 3.0).unwrap();
        assert_eq!(capacity, 9_000_000); // 10M × 3 × 0.3 = 9M

        // DiskHeavy mode: 10% cache
        let capacity = StorageMode::DiskHeavy
            .cache_capacity(100_000_000, 3.5)
            .unwrap();
        assert_eq!(capacity, 35_000_000); // 100M × 3.5 × 0.1 = 35M

        // Minimum capacity
        let capacity = StorageMode::Hybrid.cache_capacity(100, 3.0).unwrap();
        assert_eq!(capacity, 1000); // Minimum 1000
    }

    #[test]
    fn test_cache_capacity_nonzero() {
        // Memory mode: None
        assert!(StorageMode::Memory
            .cache_capacity_nonzero(10_000, 3.0)
            .is_none());

        // Hybrid mode: Some(NonZeroUsize)
        let capacity = StorageMode::Hybrid
            .cache_capacity_nonzero(10_000_000, 3.0)
            .unwrap();
        assert_eq!(capacity.get(), 9_000_000);

        // DiskHeavy mode: Some(NonZeroUsize)
        let capacity = StorageMode::DiskHeavy
            .cache_capacity_nonzero(100_000_000, 3.5)
            .unwrap();
        assert_eq!(capacity.get(), 35_000_000);
    }

    #[test]
    fn test_estimated_memory_bytes() {
        // 10M nodes, 16 neighbors, 3 levels
        let memory_mode = StorageMode::Memory.estimated_memory_bytes(10_000_000, 16, 3.0);
        // 10M × 3 levels × (16 neighbors × 4 bytes + 24 overhead) = 10M × 3 × 88 = 2.64GB
        assert!(memory_mode > 2_000_000_000); // >2GB

        // Hybrid: 30% of memory mode
        let hybrid_mode = StorageMode::Hybrid.estimated_memory_bytes(10_000_000, 16, 3.0);
        assert_eq!(hybrid_mode, (memory_mode as f64 * 0.30) as usize);

        // DiskHeavy: 10% of memory mode
        let disk_heavy_mode = StorageMode::DiskHeavy.estimated_memory_bytes(10_000_000, 16, 3.0);
        assert_eq!(disk_heavy_mode, (memory_mode as f64 * 0.10) as usize);
    }

    #[test]
    fn test_cache_percentage() {
        assert_eq!(StorageMode::Memory.cache_percentage(), None);
        assert_eq!(StorageMode::Hybrid.cache_percentage(), Some(0.30));
        assert_eq!(StorageMode::DiskHeavy.cache_percentage(), Some(0.10));
    }

    #[test]
    fn test_requires_disk() {
        assert!(!StorageMode::Memory.requires_disk());
        assert!(StorageMode::Hybrid.requires_disk());
        assert!(StorageMode::DiskHeavy.requires_disk());
    }

    #[test]
    fn test_requires_cache() {
        assert!(!StorageMode::Memory.requires_cache());
        assert!(StorageMode::Hybrid.requires_cache());
        assert!(StorageMode::DiskHeavy.requires_cache());
    }

    #[test]
    fn test_description() {
        assert!(StorageMode::Memory.description().contains("in-memory"));
        assert!(StorageMode::Hybrid.description().contains("30%"));
        assert!(StorageMode::DiskHeavy.description().contains("10%"));
    }

    #[test]
    fn test_tiering_config_auto() {
        // 5M vectors: should be Memory mode
        let config = TieringConfig::auto(5_000_000, 3.0, 16);
        assert_eq!(config.mode, StorageMode::Memory);
        assert!(config.cache_capacity.is_none());
        assert_eq!(config.num_nodes, 5_000_000);

        // 50M vectors: should be Hybrid mode
        let config = TieringConfig::auto(50_000_000, 3.5, 16);
        assert_eq!(config.mode, StorageMode::Hybrid);
        assert!(config.cache_capacity.is_some());
        assert_eq!(config.num_nodes, 50_000_000);

        // 500M vectors: should be DiskHeavy mode
        let config = TieringConfig::auto(500_000_000, 4.0, 16);
        assert_eq!(config.mode, StorageMode::DiskHeavy);
        assert!(config.cache_capacity.is_some());
        assert_eq!(config.num_nodes, 500_000_000);
    }

    #[test]
    fn test_tiering_config_with_mode() {
        // Force Hybrid mode for 1M vectors (would normally be Memory)
        let config = TieringConfig::with_mode(StorageMode::Hybrid, 1_000_000, 3.0, 16);
        assert_eq!(config.mode, StorageMode::Hybrid);
        assert!(config.cache_capacity.is_some());
    }

    #[test]
    fn test_memory_usage_human() {
        // Small: bytes
        let config = TieringConfig::auto(100, 3.0, 16);
        let human = config.memory_usage_human();
        assert!(human.contains("bytes") || human.contains("KB"));

        // Medium: MB
        let config = TieringConfig::auto(100_000, 3.0, 16);
        let human = config.memory_usage_human();
        assert!(human.contains("MB"));

        // Large: GB
        let config = TieringConfig::auto(10_000_000, 3.0, 16);
        let human = config.memory_usage_human();
        assert!(human.contains("GB") || human.contains("MB"));
    }

    #[test]
    fn test_realistic_10m_scenario() {
        // 10M vectors, 3.5 avg levels, 16 neighbors (M=16)
        let config = TieringConfig::auto(10_000_000, 3.5, 16);

        assert_eq!(config.mode, StorageMode::Hybrid);
        assert!(config.cache_capacity.is_some());

        // Cache should be 30% of 10M × 3.5 = 30% of 35M = 10.5M entries
        let cache_cap = config.cache_capacity.unwrap().get();
        assert_eq!(cache_cap, 10_500_000);

        // Memory should be reasonable (<1GB for cache)
        assert!(config.estimated_memory_bytes < 1_000_000_000);
    }

    #[test]
    fn test_realistic_100m_scenario() {
        // 100M vectors, 3.5 avg levels, 16 neighbors (M=16)
        let config = TieringConfig::auto(100_000_000, 3.5, 16);

        assert_eq!(config.mode, StorageMode::DiskHeavy);
        assert!(config.cache_capacity.is_some());

        // Cache should be 10% of 100M × 3.5 = 10% of 350M = 35M entries
        let cache_cap = config.cache_capacity.unwrap().get();
        assert_eq!(cache_cap, 35_000_000);

        // Memory should be reasonable (<4GB for cache)
        // 100M × 3.5 levels × 88 bytes/list × 0.1 = ~3.08 GB
        assert!(config.estimated_memory_bytes < 4_000_000_000);
    }
}
