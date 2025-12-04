//! Configuration for `OmenDB` storage
//!
//! Provides configuration options for the underlying storage engine.

use serde::{Deserialize, Serialize};

/// Configuration for `OmenDB` storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// WAL sync policy (fsync behavior)
    /// Default: false (`SyncPolicy::None` - for performance)
    pub sync_writes: bool,

    /// Memtable capacity in bytes
    /// Default: 128MB
    pub memtable_capacity: usize,

    /// Block cache capacity in bytes
    /// Default: 128MB
    pub block_cache_capacity: usize,

    /// Enable background compaction
    /// Default: true
    pub background_compaction: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            sync_writes: false, // Default to performance for vector graphs (derived data)
            memtable_capacity: 128 * 1024 * 1024,
            block_cache_capacity: 128 * 1024 * 1024,
            background_compaction: true,
        }
    }
}

impl StorageConfig {
    /// Create a new config builder
    #[must_use]
    pub fn builder() -> StorageConfigBuilder {
        StorageConfigBuilder::default()
    }
}

/// Builder for `StorageConfig`
#[derive(Debug, Default)]
pub struct StorageConfigBuilder {
    sync_writes: Option<bool>,
    memtable_capacity: Option<usize>,
    block_cache_capacity: Option<usize>,
    background_compaction: Option<bool>,
}

impl StorageConfigBuilder {
    /// Set sync writes policy
    #[must_use]
    pub fn sync_writes(mut self, enabled: bool) -> Self {
        self.sync_writes = Some(enabled);
        self
    }

    /// Set memtable capacity
    #[must_use]
    pub fn memtable_capacity(mut self, capacity: usize) -> Self {
        self.memtable_capacity = Some(capacity);
        self
    }

    /// Set block cache capacity
    #[must_use]
    pub fn block_cache_capacity(mut self, capacity: usize) -> Self {
        self.block_cache_capacity = Some(capacity);
        self
    }

    /// Enable/disable background compaction
    #[must_use]
    pub fn background_compaction(mut self, enabled: bool) -> Self {
        self.background_compaction = Some(enabled);
        self
    }

    /// Build configuration
    #[must_use]
    pub fn build(self) -> StorageConfig {
        let defaults = StorageConfig::default();

        StorageConfig {
            sync_writes: self.sync_writes.unwrap_or(defaults.sync_writes),
            memtable_capacity: self.memtable_capacity.unwrap_or(defaults.memtable_capacity),
            block_cache_capacity: self
                .block_cache_capacity
                .unwrap_or(defaults.block_cache_capacity),
            background_compaction: self
                .background_compaction
                .unwrap_or(defaults.background_compaction),
        }
    }
}
