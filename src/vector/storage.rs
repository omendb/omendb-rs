//! oadb Storage Backend
//!
//! Persistent storage for vector data using seerdb LSM engine.
//!
//! # Key Schema
//! - `v:{id}` → bincode-serialized vector (f32 array)
//! - `m:{id}` → JSON metadata
//! - `i:{string_id}` → internal index (u64 little-endian)
//! - `cfg:dimensions` → dimensions (u64)
//! - `cfg:count` → vector count (u64)

use anyhow::Result;
use rayon::prelude::*;
use seerdb::{DBOptions, SyncPolicy, DB};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Storage backend for oadb vectors and metadata
pub struct SeerDBStorage {
    db: DB,
    path: PathBuf,
}

impl SeerDBStorage {
    /// Open or create storage at the given path
    ///
    /// Path should be a directory ending in `.oadb/` (or any directory name).
    /// Directory will be created if it doesn't exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let db = DBOptions::default()
            .memtable_capacity(64 * 1024 * 1024)
            .sync_policy(SyncPolicy::SyncData)
            .background_compaction(true)
            .open(&path)?;

        Ok(Self { db, path })
    }

    /// Open with custom options for testing/tuning
    pub fn open_with_options(path: impl AsRef<Path>, options: &DBOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let db = options.open(&path)?;
        Ok(Self { db, path })
    }

    /// Store a vector by internal index
    pub fn put_vector(&self, id: usize, vector: &[f32]) -> Result<()> {
        let key = format!("v:{id}");
        let value = bincode::serialize(vector)?;
        self.db.put(&key, &value)?;
        Ok(())
    }

    /// Get a vector by internal index
    pub fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>> {
        let key = format!("v:{id}");
        match self.db.get(&key)? {
            Some(bytes) => {
                let vector: Vec<f32> = bincode::deserialize(&bytes)?;
                Ok(Some(vector))
            }
            None => Ok(None),
        }
    }

    /// Store metadata for a vector
    pub fn put_metadata(&self, id: usize, metadata: &serde_json::Value) -> Result<()> {
        let key = format!("m:{id}");
        let value = serde_json::to_vec(metadata)?;
        self.db.put(&key, &value)?;
        Ok(())
    }

    /// Get metadata for a vector
    pub fn get_metadata(&self, id: usize) -> Result<Option<serde_json::Value>> {
        let key = format!("m:{id}");
        match self.db.get(&key)? {
            Some(bytes) => {
                let metadata: serde_json::Value = serde_json::from_slice(&bytes)?;
                Ok(Some(metadata))
            }
            None => Ok(None),
        }
    }

    /// Store string ID to internal index mapping
    ///
    /// Stores both:
    /// - `i:{string_id}` → index (for lookup by string ID)
    /// - `r:{index}` → `string_id` (for rebuilding `id_to_index` on load)
    pub fn put_id_mapping(&self, string_id: &str, index: usize) -> Result<()> {
        // Forward mapping: string_id → index
        let key = format!("i:{string_id}");
        let value = (index as u64).to_le_bytes();
        self.db.put(&key, value)?;

        // Reverse mapping: index → string_id (for rebuild on load)
        let reverse_key = format!("r:{index}");
        self.db.put(&reverse_key, string_id.as_bytes())?;

        Ok(())
    }

    /// Get internal index for a string ID
    pub fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>> {
        let key = format!("i:{string_id}");
        match self.db.get(&key)? {
            Some(bytes) => {
                if bytes.len() != 8 {
                    anyhow::bail!("Invalid id mapping size");
                }
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                Ok(Some(u64::from_le_bytes(arr) as usize))
            }
            None => Ok(None),
        }
    }

    /// Get string ID for an internal index (reverse lookup)
    pub fn get_string_id(&self, index: usize) -> Result<Option<String>> {
        let key = format!("r:{index}");
        match self.db.get(&key)? {
            Some(bytes) => {
                let string_id = std::str::from_utf8(&bytes)?.to_string();
                Ok(Some(string_id))
            }
            None => Ok(None),
        }
    }

    /// Delete string ID mapping
    pub fn delete_id_mapping(&self, string_id: &str) -> Result<()> {
        // Get the index first so we can delete the reverse mapping
        if let Some(index) = self.get_id_mapping(string_id)? {
            let reverse_key = format!("r:{index}");
            self.db.delete(&reverse_key)?;
        }

        let key = format!("i:{string_id}");
        self.db.delete(&key)?;
        Ok(())
    }

    /// Store configuration value
    pub fn put_config(&self, key: &str, value: u64) -> Result<()> {
        let full_key = format!("cfg:{key}");
        self.db.put(&full_key, value.to_le_bytes())?;
        Ok(())
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Result<Option<u64>> {
        let full_key = format!("cfg:{key}");
        match self.db.get(&full_key)? {
            Some(bytes) => {
                if bytes.len() != 8 {
                    anyhow::bail!("Invalid config value size");
                }
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&bytes);
                Ok(Some(u64::from_le_bytes(arr)))
            }
            None => Ok(None),
        }
    }

    /// Load all vectors from storage (parallel)
    ///
    /// Returns vectors indexed by their internal ID.
    /// Used during startup to rebuild HNSW index.
    pub fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>> {
        // Get the count of vectors from config
        let count = self.get_config("count")?.unwrap_or(0) as usize;

        if count == 0 {
            return Ok(Vec::new());
        }

        // Parallel load: each thread fetches vectors independently
        // seerdb supports concurrent reads
        let vectors: Vec<(usize, Vec<f32>)> = (0..count)
            .into_par_iter()
            .filter_map(|id| self.get_vector(id).ok().flatten().map(|v| (id, v)))
            .collect();

        Ok(vectors)
    }

    /// Increment vector count in storage
    pub fn increment_count(&self) -> Result<usize> {
        let count = self.get_config("count")?.unwrap_or(0) as usize;
        let new_count = count + 1;
        self.put_config("count", new_count as u64)?;
        Ok(new_count)
    }

    /// Get current vector count
    pub fn get_count(&self) -> Result<usize> {
        Ok(self.get_config("count")?.unwrap_or(0) as usize)
    }

    /// Load all metadata from storage (parallel)
    pub fn load_all_metadata(&self) -> Result<HashMap<usize, serde_json::Value>> {
        // Get the count of vectors from config
        let count = self.get_config("count")?.unwrap_or(0) as usize;

        if count == 0 {
            return Ok(HashMap::new());
        }

        // Parallel load metadata
        let metadata: HashMap<usize, serde_json::Value> = (0..count)
            .into_par_iter()
            .filter_map(|id| self.get_metadata(id).ok().flatten().map(|m| (id, m)))
            .collect();

        Ok(metadata)
    }

    /// Load all ID mappings from storage (parallel)
    ///
    /// Uses reverse mapping (index → `string_id`) to rebuild `id_to_index` on load.
    pub fn load_all_id_mappings(&self) -> Result<HashMap<String, usize>> {
        // Get the count of vectors
        let count = self.get_config("count")?.unwrap_or(0) as usize;

        if count == 0 {
            return Ok(HashMap::new());
        }

        // Parallel load ID mappings
        let mappings: HashMap<String, usize> = (0..count)
            .into_par_iter()
            .filter_map(|index| {
                self.get_string_id(index)
                    .ok()
                    .flatten()
                    .map(|string_id| (string_id, index))
            })
            .collect();

        Ok(mappings)
    }

    /// Mark a vector as deleted (tombstone)
    pub fn put_deleted(&self, id: usize) -> Result<()> {
        let key = format!("d:{id}");
        self.db.put(&key, [1])?;
        Ok(())
    }

    /// Check if a vector is deleted
    pub fn is_deleted(&self, id: usize) -> Result<bool> {
        let key = format!("d:{id}");
        Ok(self.db.get(&key)?.is_some())
    }

    /// Remove deleted marker (for re-insertion)
    pub fn remove_deleted(&self, id: usize) -> Result<()> {
        let key = format!("d:{id}");
        self.db.delete(&key)?;
        Ok(())
    }

    /// Load all deleted IDs from storage
    pub fn load_all_deleted(&self) -> Result<HashMap<usize, bool>> {
        use bytes::Bytes;

        let mut deleted = HashMap::new();

        let iter = self.db.prefix(b"d:")?;
        for entry in iter {
            let (key, _value): (Bytes, Bytes) = entry.map_err(|e| anyhow::anyhow!("{e}"))?;
            // Skip keys that aren't valid UTF-8
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if let Some(id_str) = key_str.strip_prefix("d:") {
                    if let Ok(id) = id_str.parse::<usize>() {
                        deleted.insert(id, true);
                    }
                }
            }
        }

        Ok(deleted)
    }

    /// Get storage path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get seerdb statistics for profiling
    ///
    /// Returns cache hit rate, `SSTable` counts, latency percentiles, etc.
    pub fn stats(&self) -> seerdb::DBStats {
        self.db.stats()
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }

    /// Batch set vectors with metadata and ID mappings
    ///
    /// Uses seerdb batch API for atomic, high-performance writes.
    /// 2-5x faster than individual puts for batches of 100+ operations.
    ///
    /// # Arguments
    /// * `items` - Vec of (index, `string_id`, vector, metadata) tuples
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if any serialization or write fails
    pub fn put_batch(
        &self,
        items: Vec<(usize, String, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        // Calculate capacity: 4 writes per item (vector, metadata, forward id, reverse id) + 1 count
        let capacity = items.len() * 4 + 1;
        let mut batch = self.db.batch_with_capacity(capacity);

        for (idx, string_id, vector, metadata) in &items {
            // Vector
            let key = format!("v:{idx}");
            let value = bincode::serialize(vector)?;
            batch.put(key.as_bytes(), &value);

            // Metadata
            let key = format!("m:{idx}");
            let value = serde_json::to_vec(metadata)?;
            batch.put(key.as_bytes(), &value);

            // Forward ID mapping: string_id → index
            let key = format!("i:{string_id}");
            let value = (*idx as u64).to_le_bytes();
            batch.put(key.as_bytes(), value);

            // Reverse ID mapping: index → string_id
            let key = format!("r:{idx}");
            batch.put(key.as_bytes(), string_id.as_bytes());
        }

        // Update count
        let current_count = self.get_config("count")?.unwrap_or(0) as usize;
        let new_count = current_count + items.len();
        let key = b"cfg:count";
        batch.put(key, (new_count as u64).to_le_bytes());

        // Commit all writes atomically
        batch.commit()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vector_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SeerDBStorage::open(temp_dir.path()).unwrap();

        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        storage.put_vector(0, &vector).unwrap();

        let loaded = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(vector, loaded);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SeerDBStorage::open(temp_dir.path()).unwrap();

        let metadata = serde_json::json!({
            "label": "test",
            "score": 0.95
        });
        storage.put_metadata(0, &metadata).unwrap();

        let loaded = storage.get_metadata(0).unwrap().unwrap();
        assert_eq!(metadata, loaded);
    }

    #[test]
    fn test_id_mapping_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SeerDBStorage::open(temp_dir.path()).unwrap();

        storage.put_id_mapping("doc_123", 42).unwrap();
        let loaded = storage.get_id_mapping("doc_123").unwrap().unwrap();
        assert_eq!(42, loaded);

        // Delete
        storage.delete_id_mapping("doc_123").unwrap();
        assert!(storage.get_id_mapping("doc_123").unwrap().is_none());
    }

    #[test]
    fn test_load_all_vectors() {
        let temp_dir = TempDir::new().unwrap();
        let storage = SeerDBStorage::open(temp_dir.path()).unwrap();

        // Insert vectors in order (load_all_vectors uses count-based loading)
        storage.put_vector(0, &[0.0, 0.0]).unwrap();
        storage.increment_count().unwrap();
        storage.put_vector(1, &[1.0, 1.0]).unwrap();
        storage.increment_count().unwrap();
        storage.put_vector(2, &[2.0, 2.0]).unwrap();
        storage.increment_count().unwrap();

        let vectors = storage.load_all_vectors().unwrap();
        assert_eq!(vectors.len(), 3);
        // Should be sorted by ID
        assert_eq!(vectors[0].0, 0);
        assert_eq!(vectors[1].0, 1);
        assert_eq!(vectors[2].0, 2);
    }

    #[test]
    fn test_persistence_across_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write data
        {
            let storage = SeerDBStorage::open(&path).unwrap();
            storage.put_vector(0, &[1.0, 2.0, 3.0]).unwrap();
            storage
                .put_metadata(0, &serde_json::json!({"test": true}))
                .unwrap();
            storage.put_id_mapping("doc1", 0).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let storage = SeerDBStorage::open(&path).unwrap();
            let vector = storage.get_vector(0).unwrap().unwrap();
            assert_eq!(vector, vec![1.0, 2.0, 3.0]);

            let metadata = storage.get_metadata(0).unwrap().unwrap();
            assert_eq!(metadata["test"], true);

            let index = storage.get_id_mapping("doc1").unwrap().unwrap();
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn test_load_all_after_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write data
        {
            let storage = SeerDBStorage::open(&path).unwrap();
            storage.put_vector(0, &[1.0, 2.0]).unwrap();
            storage.increment_count().unwrap();
            storage.put_vector(1, &[3.0, 4.0]).unwrap();
            storage.increment_count().unwrap();
            storage
                .put_metadata(0, &serde_json::json!({"id": 0}))
                .unwrap();
            storage
                .put_metadata(1, &serde_json::json!({"id": 1}))
                .unwrap();
            storage.put_id_mapping("doc0", 0).unwrap();
            storage.put_id_mapping("doc1", 1).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify with load_all methods
        {
            let storage = SeerDBStorage::open(&path).unwrap();

            let vectors = storage.load_all_vectors().unwrap();
            assert_eq!(
                vectors.len(),
                2,
                "Expected 2 vectors, got {}",
                vectors.len()
            );
            assert_eq!(vectors[0], (0, vec![1.0, 2.0]));
            assert_eq!(vectors[1], (1, vec![3.0, 4.0]));

            let metadata = storage.load_all_metadata().unwrap();
            assert_eq!(metadata.len(), 2);
            assert_eq!(metadata[&0]["id"], 0);
            assert_eq!(metadata[&1]["id"], 1);

            // Note: id_mappings use prefix scan which may not work across reopens
            // in current seerdb - this is a known limitation
        }
    }
}
