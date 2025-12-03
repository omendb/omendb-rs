//! Vector metadata storage using seerdb
//!
//! Stores user-facing JSON metadata for vectors.
//! Key format: "`v:{vector_id`}" as UTF-8 bytes
//! Value: JSON-serialized metadata bytes

use crate::{config::StorageConfig, Result, OmenDBError};
use seerdb::db::{DBOptions, DB};
use seerdb::SyncPolicy;
use std::path::PathBuf;
use std::sync::Arc;

/// Vector metadata storage using seerdb
///
/// Stores JSON metadata for vectors. This is user-facing metadata
/// used for filtering queries, not internal HNSW graph metadata.
#[derive(Clone)]
pub struct VectorMetadataStorage {
    db: Arc<DB>,
}

impl VectorMetadataStorage {
    /// Create new vector metadata storage at path
    pub fn new(path: PathBuf, config: &StorageConfig) -> Result<Self> {
        let options = DBOptions {
            data_dir: path,

            // Configurable durability
            wal_sync_policy: if config.sync_writes {
                SyncPolicy::SyncData
            } else {
                SyncPolicy::None
            },

            // Medium memtable for metadata (32MB default or derived)
            // We use 1/4th of the main config capacity
            memtable_capacity: config.memtable_capacity / 4,

            background_compaction: config.background_compaction,

            ..Default::default()
        };

        let db = DB::open(options).map_err(|e| OmenDBError::Backend(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Create from existing seerdb instance
    pub fn from_db(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Get reference to underlying seerdb
    #[must_use] 
    pub fn db(&self) -> &DB {
        &self.db
    }

    /// Store metadata for a vector
    ///
    /// # Arguments
    /// * `key` - Key string (typically "v:{id}")
    /// * `value` - JSON-serialized metadata bytes
    pub fn insert(&self, key: &str, value: &[u8]) -> Result<()> {
        self.db
            .put(key.as_bytes(), value)
            .map_err(|e| OmenDBError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Store metadata for multiple vectors in batch
    pub fn insert_batch(&self, items: &[(&str, &[u8])]) -> Result<()> {
        let mut batch = self.db.batch_with_capacity(items.len());

        for &(key, value) in items {
            batch.put(key.as_bytes(), value);
        }

        batch
            .commit()
            .map_err(|e| OmenDBError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Get metadata for a vector
    ///
    /// Returns None if key not found.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let value = self
            .db
            .get(key.as_bytes())
            .map_err(|e| OmenDBError::Backend(e.to_string()))?;
        Ok(value.map(|bytes| bytes.to_vec()))
    }

    /// Remove metadata for a vector
    pub fn remove(&self, key: &str) -> Result<()> {
        self.db
            .delete(key.as_bytes())
            .map_err(|e| OmenDBError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.db
            .flush()
            .map_err(|e| OmenDBError::Backend(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_insert_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &StorageConfig::default(),
        )
        .unwrap();

        let key = "v:42";
        let value = br#"{"category": "test", "score": 0.95}"#;
        storage.insert(key, value).unwrap();

        let retrieved = storage.get(key).unwrap();
        assert_eq!(retrieved, Some(value.to_vec()));
    }

    #[test]
    fn test_get_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &StorageConfig::default(),
        )
        .unwrap();

        let result = storage.get("v:999").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_update_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &StorageConfig::default(),
        )
        .unwrap();

        let key = "v:42";
        storage.insert(key, br#"{"v": 1}"#).unwrap();
        storage.insert(key, br#"{"v": 2}"#).unwrap();

        let retrieved = storage.get(key).unwrap();
        assert_eq!(retrieved, Some(br#"{"v": 2}"#.to_vec()));
    }

    #[test]
    fn test_remove_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &StorageConfig::default(),
        )
        .unwrap();

        let key = "v:42";
        storage.insert(key, br#"{"test": true}"#).unwrap();
        assert!(storage.get(key).unwrap().is_some());

        storage.remove(key).unwrap();
        assert_eq!(storage.get(key).unwrap(), None);
    }

    #[test]
    fn test_multiple_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let storage = VectorMetadataStorage::new(
            temp_dir.path().to_path_buf(),
            &StorageConfig::default(),
        )
        .unwrap();

        storage.insert("v:1", br#"{"id": 1}"#).unwrap();
        storage.insert("v:2", br#"{"id": 2}"#).unwrap();
        storage.insert("v:3", br#"{"id": 3}"#).unwrap();

        assert_eq!(storage.get("v:1").unwrap(), Some(br#"{"id": 1}"#.to_vec()));
        assert_eq!(storage.get("v:2").unwrap(), Some(br#"{"id": 2}"#.to_vec()));
        assert_eq!(storage.get("v:3").unwrap(), Some(br#"{"id": 3}"#.to_vec()));
    }

    #[test]
    fn test_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Create storage and insert
        {
            let storage =
                VectorMetadataStorage::new(path.clone(), &StorageConfig::default()).unwrap();
            storage.insert("v:42", br#"{"persistent": true}"#).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let storage = VectorMetadataStorage::new(path, &StorageConfig::default()).unwrap();
            let retrieved = storage.get("v:42").unwrap();
            assert_eq!(retrieved, Some(br#"{"persistent": true}"#.to_vec()));
        }
    }
}
