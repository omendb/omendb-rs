//! Node metadata storage using seerdb
//!
//! Stores per-node metadata (max_level) for HNSW hierarchical navigation.
//! Key format: node_id (8 bytes big-endian)
//! Value: max_level (1 byte)

use crate::{config::SeerdbVectorConfig, Result, SeerdbVectorError};
use seerdb::db::{DBOptions, DB};
use seerdb::SyncPolicy;
use std::path::PathBuf;
use std::sync::Arc;

/// Node metadata storage using seerdb
///
/// Stores node max levels for persistent HNSW hierarchical navigation.
/// Used to determine starting layer for disk-based HNSW search.
#[derive(Clone)]
pub struct NodeMetadataStorage {
    db: Arc<DB>,
}

impl NodeMetadataStorage {
    /// Create new node metadata storage at path
    pub fn new(path: PathBuf, config: &SeerdbVectorConfig) -> Result<Self> {
        let options = DBOptions {
            data_dir: path,

            // Configurable durability
            wal_sync_policy: if config.sync_writes {
                SyncPolicy::SyncData
            } else {
                SyncPolicy::None
            },

            // Smaller memtable for metadata (default 16MB or derived from config)
            // We use 1/8th of the main config capacity for metadata
            memtable_capacity: config.memtable_capacity / 8,

            background_compaction: config.background_compaction,

            ..Default::default()
        };

        let db = DB::open(options).map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Create from existing seerdb instance
    pub fn from_db(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Get reference to underlying seerdb
    pub fn db(&self) -> &DB {
        &self.db
    }

    /// Encode key: node_id as big-endian bytes
    fn encode_key(node_id: u64) -> [u8; 8] {
        node_id.to_be_bytes()
    }

    /// Set max level for a node
    ///
    /// # Arguments
    /// * `node_id` - Node ID
    /// * `max_level` - Maximum HNSW level for this node (0-15)
    pub fn set_max_level(&self, node_id: u64, max_level: u8) -> Result<()> {
        let key = Self::encode_key(node_id);
        self.db
            .put(key, [max_level])
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Set max levels for multiple nodes in batch
    pub fn set_max_levels_batch(&self, levels: &[(u64, u8)]) -> Result<()> {
        let mut batch = self.db.batch_with_capacity(levels.len());

        for &(node_id, max_level) in levels {
            let key = Self::encode_key(node_id);
            batch.put(key, [max_level]);
        }

        batch
            .commit()
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Get max level for a node
    ///
    /// Returns None if node not found.
    pub fn get_max_level(&self, node_id: u64) -> Result<Option<u8>> {
        let key = Self::encode_key(node_id);

        match self
            .db
            .get(key)
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?
        {
            Some(value) => {
                if value.len() != 1 {
                    return Err(SeerdbVectorError::InvalidData(format!(
                        "Invalid node metadata value length: {}",
                        value.len()
                    )));
                }
                Ok(Some(value[0]))
            }
            None => Ok(None),
        }
    }

    /// Remove metadata for a node
    pub fn remove_node(&self, node_id: u64) -> Result<()> {
        let key = Self::encode_key(node_id);
        self.db
            .delete(key)
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Remove metadata for multiple nodes in batch
    pub fn remove_nodes_batch(&self, node_ids: &[u64]) -> Result<()> {
        let mut batch = self.db.batch_with_capacity(node_ids.len());

        for &node_id in node_ids {
            let key = Self::encode_key(node_id);
            batch.delete(key);
        }

        batch
            .commit()
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| SeerdbVectorError::Backend(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_set_and_get_max_level() {
        let temp_dir = TempDir::new().unwrap();
        let storage = NodeMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        storage.set_max_level(42, 3).unwrap();

        let level = storage.get_max_level(42).unwrap();
        assert_eq!(level, Some(3));
    }

    #[test]
    fn test_get_nonexistent_node() {
        let temp_dir = TempDir::new().unwrap();
        let storage = NodeMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        let level = storage.get_max_level(999).unwrap();
        assert_eq!(level, None);
    }

    #[test]
    fn test_update_max_level() {
        let temp_dir = TempDir::new().unwrap();
        let storage = NodeMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        storage.set_max_level(42, 2).unwrap();
        assert_eq!(storage.get_max_level(42).unwrap(), Some(2));

        storage.set_max_level(42, 5).unwrap();
        assert_eq!(storage.get_max_level(42).unwrap(), Some(5));
    }

    #[test]
    fn test_remove_node() {
        let temp_dir = TempDir::new().unwrap();
        let storage = NodeMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        storage.set_max_level(42, 3).unwrap();
        assert_eq!(storage.get_max_level(42).unwrap(), Some(3));

        storage.remove_node(42).unwrap();
        assert_eq!(storage.get_max_level(42).unwrap(), None);
    }

    #[test]
    fn test_batch_operations() {
        let temp_dir = TempDir::new().unwrap();
        let storage = NodeMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &SeerdbVectorConfig::default(),
        )
        .unwrap();

        // Batch set
        let levels = vec![(1, 0), (2, 1), (3, 2), (4, 3)];
        storage.set_max_levels_batch(&levels).unwrap();

        assert_eq!(storage.get_max_level(1).unwrap(), Some(0));
        assert_eq!(storage.get_max_level(2).unwrap(), Some(1));
        assert_eq!(storage.get_max_level(3).unwrap(), Some(2));
        assert_eq!(storage.get_max_level(4).unwrap(), Some(3));

        // Batch remove
        storage.remove_nodes_batch(&[2, 3]).unwrap();
        assert_eq!(storage.get_max_level(1).unwrap(), Some(0));
        assert_eq!(storage.get_max_level(2).unwrap(), None);
        assert_eq!(storage.get_max_level(3).unwrap(), None);
        assert_eq!(storage.get_max_level(4).unwrap(), Some(3));
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Create storage and set level
        {
            let storage =
                NodeMetadataStorage::new(path.clone(), &SeerdbVectorConfig::default()).unwrap();
            storage.set_max_level(42, 3).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify persistence
        {
            let storage = NodeMetadataStorage::new(path, &SeerdbVectorConfig::default()).unwrap();
            assert_eq!(storage.get_max_level(42).unwrap(), Some(3));
        }
    }
}
