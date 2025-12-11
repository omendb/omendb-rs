//! OmenStorage - storage backend using .omen format
//!
//! Drop-in replacement for SeerDBStorage with same interface.
//!
//! # Key Schema (same as SeerDBStorage)
//! - `v:{id}` → bincode-serialized vector (f32 array)
//! - `m:{id}` → JSON metadata
//! - `i:{string_id}` → internal index (u64 little-endian)
//! - `r:{index}` → string ID (for reverse lookup)
//! - `d:{index}` → deleted marker
//! - `cfg:{key}` → config value (u64)

use crate::vector::storage_trait::Storage;
use anyhow::Result;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Storage backend using .omen format
pub struct OmenStorage {
    path: PathBuf,
    // In-memory indexes for fast lookup
    vectors: RwLock<HashMap<usize, Vec<f32>>>,
    metadata: RwLock<HashMap<usize, serde_json::Value>>,
    id_to_index: RwLock<HashMap<String, usize>>,
    index_to_id: RwLock<HashMap<usize, String>>,
    deleted: RwLock<HashMap<usize, bool>>,
    config: RwLock<HashMap<String, u64>>,
    // Dirty flag for flush
    dirty: RwLock<bool>,
}

impl OmenStorage {
    /// Open or create storage at the given path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create directory if needed
        if !path.exists() {
            fs::create_dir_all(&path)?;
        }

        let storage = Self {
            path,
            vectors: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(HashMap::new()),
            deleted: RwLock::new(HashMap::new()),
            config: RwLock::new(HashMap::new()),
            dirty: RwLock::new(false),
        };

        // Load existing data
        storage.load_from_disk()?;

        Ok(storage)
    }

    /// Load all data from disk
    fn load_from_disk(&self) -> Result<()> {
        let index_path = self.path.join("index.json");
        if !index_path.exists() {
            return Ok(());
        }

        let file = File::open(&index_path)?;
        let reader = BufReader::new(file);

        #[derive(serde::Deserialize)]
        struct StorageIndex {
            #[serde(default)]
            vectors: HashMap<String, Vec<f32>>,
            #[serde(default)]
            metadata: HashMap<String, serde_json::Value>,
            #[serde(default)]
            id_to_index: HashMap<String, usize>,
            #[serde(default)]
            deleted: HashMap<String, bool>,
            #[serde(default)]
            config: HashMap<String, u64>,
        }

        let index: StorageIndex = serde_json::from_reader(reader)?;

        // Convert string keys to usize for vectors
        let mut vectors = self.vectors.write();
        for (k, v) in index.vectors {
            if let Ok(id) = k.parse::<usize>() {
                vectors.insert(id, v);
            }
        }

        // Convert string keys for metadata
        let mut metadata = self.metadata.write();
        for (k, v) in index.metadata {
            if let Ok(id) = k.parse::<usize>() {
                metadata.insert(id, v);
            }
        }

        // ID mappings
        let mut id_to_index = self.id_to_index.write();
        let mut index_to_id = self.index_to_id.write();
        for (string_id, idx) in index.id_to_index {
            id_to_index.insert(string_id.clone(), idx);
            index_to_id.insert(idx, string_id);
        }

        // Deleted markers
        let mut deleted = self.deleted.write();
        for (k, v) in index.deleted {
            if let Ok(id) = k.parse::<usize>() {
                deleted.insert(id, v);
            }
        }

        // Config
        let mut config = self.config.write();
        for (k, v) in index.config {
            config.insert(k, v);
        }

        Ok(())
    }

    /// Save all data to disk
    fn save_to_disk(&self) -> Result<()> {
        let index_path = self.path.join("index.json");

        #[derive(serde::Serialize)]
        struct StorageIndex {
            vectors: HashMap<String, Vec<f32>>,
            metadata: HashMap<String, serde_json::Value>,
            id_to_index: HashMap<String, usize>,
            deleted: HashMap<String, bool>,
            config: HashMap<String, u64>,
        }

        let vectors = self.vectors.read();
        let metadata = self.metadata.read();
        let id_to_index = self.id_to_index.read();
        let deleted = self.deleted.read();
        let config = self.config.read();

        let index = StorageIndex {
            vectors: vectors
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            id_to_index: id_to_index.clone(),
            deleted: deleted.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            config: config.clone(),
        };

        let file = File::create(&index_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &index)?;

        *self.dirty.write() = false;
        Ok(())
    }

    /// Store a vector by internal index
    pub fn put_vector(&self, id: usize, vector: &[f32]) -> Result<()> {
        self.vectors.write().insert(id, vector.to_vec());
        *self.dirty.write() = true;
        Ok(())
    }

    /// Get a vector by internal index
    pub fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>> {
        Ok(self.vectors.read().get(&id).cloned())
    }

    /// Store metadata for a vector
    pub fn put_metadata(&self, id: usize, metadata: &serde_json::Value) -> Result<()> {
        self.metadata.write().insert(id, metadata.clone());
        *self.dirty.write() = true;
        Ok(())
    }

    /// Get metadata for a vector
    pub fn get_metadata(&self, id: usize) -> Result<Option<serde_json::Value>> {
        Ok(self.metadata.read().get(&id).cloned())
    }

    /// Store string ID to internal index mapping
    pub fn put_id_mapping(&self, string_id: &str, index: usize) -> Result<()> {
        self.id_to_index
            .write()
            .insert(string_id.to_string(), index);
        self.index_to_id
            .write()
            .insert(index, string_id.to_string());
        *self.dirty.write() = true;
        Ok(())
    }

    /// Get internal index for a string ID
    pub fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>> {
        Ok(self.id_to_index.read().get(string_id).copied())
    }

    /// Get string ID for an internal index (reverse lookup)
    pub fn get_string_id(&self, index: usize) -> Result<Option<String>> {
        Ok(self.index_to_id.read().get(&index).cloned())
    }

    /// Delete string ID mapping
    pub fn delete_id_mapping(&self, string_id: &str) -> Result<()> {
        if let Some(index) = self.id_to_index.write().remove(string_id) {
            self.index_to_id.write().remove(&index);
        }
        *self.dirty.write() = true;
        Ok(())
    }

    /// Store configuration value
    pub fn put_config(&self, key: &str, value: u64) -> Result<()> {
        self.config.write().insert(key.to_string(), value);
        *self.dirty.write() = true;
        Ok(())
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.config.read().get(key).copied())
    }

    /// Load all vectors from storage
    pub fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>> {
        let vectors = self.vectors.read();
        let mut result: Vec<(usize, Vec<f32>)> =
            vectors.iter().map(|(k, v)| (*k, v.clone())).collect();
        result.sort_by_key(|(id, _)| *id);
        Ok(result)
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

    /// Store quantization mode
    pub fn put_quantization_mode(&self, mode: u64) -> Result<()> {
        self.put_config("quantization", mode)
    }

    /// Get quantization mode
    pub fn get_quantization_mode(&self) -> Result<Option<u64>> {
        self.get_config("quantization")
    }

    /// Check if store was created with quantization
    pub fn is_quantized(&self) -> Result<bool> {
        Ok(self.get_quantization_mode()?.unwrap_or(0) > 0)
    }

    /// Load all metadata from storage
    pub fn load_all_metadata(&self) -> Result<HashMap<usize, serde_json::Value>> {
        Ok(self.metadata.read().clone())
    }

    /// Load all ID mappings from storage
    pub fn load_all_id_mappings(&self) -> Result<HashMap<String, usize>> {
        Ok(self.id_to_index.read().clone())
    }

    /// Mark a vector as deleted
    pub fn put_deleted(&self, id: usize) -> Result<()> {
        self.deleted.write().insert(id, true);
        *self.dirty.write() = true;
        Ok(())
    }

    /// Check if a vector is deleted
    pub fn is_deleted(&self, id: usize) -> Result<bool> {
        Ok(self.deleted.read().get(&id).copied().unwrap_or(false))
    }

    /// Remove deleted marker
    pub fn remove_deleted(&self, id: usize) -> Result<()> {
        self.deleted.write().remove(&id);
        *self.dirty.write() = true;
        Ok(())
    }

    /// Load all deleted IDs
    pub fn load_all_deleted(&self) -> Result<HashMap<usize, bool>> {
        Ok(self.deleted.read().clone())
    }

    /// Get storage path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        if *self.dirty.read() {
            self.save_to_disk()?;
        }
        Ok(())
    }

    /// Batch set vectors with metadata and ID mappings
    pub fn put_batch(
        &self,
        items: Vec<(usize, String, Vec<f32>, serde_json::Value)>,
    ) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        {
            let mut vectors = self.vectors.write();
            let mut metadata = self.metadata.write();
            let mut id_to_index = self.id_to_index.write();
            let mut index_to_id = self.index_to_id.write();

            for (idx, string_id, vector, meta) in items {
                vectors.insert(idx, vector);
                metadata.insert(idx, meta);
                id_to_index.insert(string_id.clone(), idx);
                index_to_id.insert(idx, string_id);
            }
        }

        // Update count
        let current_count = self.get_config("count")?.unwrap_or(0) as usize;
        let vectors_len = self.vectors.read().len();
        if vectors_len > current_count {
            self.put_config("count", vectors_len as u64)?;
        }

        *self.dirty.write() = true;
        Ok(())
    }

    /// Get storage statistics (stub for compatibility)
    pub fn stats(&self) -> OmenStorageStats {
        OmenStorageStats {
            vector_count: self.vectors.read().len(),
            metadata_count: self.metadata.read().len(),
        }
    }
}

impl Drop for OmenStorage {
    fn drop(&mut self) {
        // Auto-flush on drop
        let _ = self.flush();
    }
}

impl Storage for OmenStorage {
    fn put_vector(&self, id: usize, vector: &[f32]) -> Result<()> {
        OmenStorage::put_vector(self, id, vector)
    }

    fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>> {
        OmenStorage::get_vector(self, id)
    }

    fn put_metadata(&self, id: usize, metadata: &JsonValue) -> Result<()> {
        OmenStorage::put_metadata(self, id, metadata)
    }

    fn get_metadata(&self, id: usize) -> Result<Option<JsonValue>> {
        OmenStorage::get_metadata(self, id)
    }

    fn put_id_mapping(&self, string_id: &str, index: usize) -> Result<()> {
        OmenStorage::put_id_mapping(self, string_id, index)
    }

    fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>> {
        OmenStorage::get_id_mapping(self, string_id)
    }

    fn get_string_id(&self, index: usize) -> Result<Option<String>> {
        OmenStorage::get_string_id(self, index)
    }

    fn delete_id_mapping(&self, string_id: &str) -> Result<()> {
        OmenStorage::delete_id_mapping(self, string_id)
    }

    fn put_config(&self, key: &str, value: u64) -> Result<()> {
        OmenStorage::put_config(self, key, value)
    }

    fn get_config(&self, key: &str) -> Result<Option<u64>> {
        OmenStorage::get_config(self, key)
    }

    fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>> {
        OmenStorage::load_all_vectors(self)
    }

    fn increment_count(&self) -> Result<usize> {
        OmenStorage::increment_count(self)
    }

    fn get_count(&self) -> Result<usize> {
        OmenStorage::get_count(self)
    }

    fn put_quantization_mode(&self, mode: u64) -> Result<()> {
        OmenStorage::put_quantization_mode(self, mode)
    }

    fn get_quantization_mode(&self) -> Result<Option<u64>> {
        OmenStorage::get_quantization_mode(self)
    }

    fn is_quantized(&self) -> Result<bool> {
        OmenStorage::is_quantized(self)
    }

    fn load_all_metadata(&self) -> Result<HashMap<usize, JsonValue>> {
        OmenStorage::load_all_metadata(self)
    }

    fn load_all_id_mappings(&self) -> Result<HashMap<String, usize>> {
        OmenStorage::load_all_id_mappings(self)
    }

    fn put_deleted(&self, id: usize) -> Result<()> {
        OmenStorage::put_deleted(self, id)
    }

    fn is_deleted(&self, id: usize) -> Result<bool> {
        OmenStorage::is_deleted(self, id)
    }

    fn remove_deleted(&self, id: usize) -> Result<()> {
        OmenStorage::remove_deleted(self, id)
    }

    fn load_all_deleted(&self) -> Result<HashMap<usize, bool>> {
        OmenStorage::load_all_deleted(self)
    }

    fn flush(&self) -> Result<()> {
        OmenStorage::flush(self)
    }

    fn put_batch(&self, items: Vec<(usize, String, Vec<f32>, JsonValue)>) -> Result<()> {
        OmenStorage::put_batch(self, items)
    }

    fn path(&self) -> &Path {
        OmenStorage::path(self)
    }
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct OmenStorageStats {
    pub vector_count: usize,
    pub metadata_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_vector_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = OmenStorage::open(temp_dir.path()).unwrap();

        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        storage.put_vector(0, &vector).unwrap();

        let loaded = storage.get_vector(0).unwrap().unwrap();
        assert_eq!(vector, loaded);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let storage = OmenStorage::open(temp_dir.path()).unwrap();

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
        let storage = OmenStorage::open(temp_dir.path()).unwrap();

        storage.put_id_mapping("doc_123", 42).unwrap();
        let loaded = storage.get_id_mapping("doc_123").unwrap().unwrap();
        assert_eq!(42, loaded);

        // Delete
        storage.delete_id_mapping("doc_123").unwrap();
        assert!(storage.get_id_mapping("doc_123").unwrap().is_none());
    }

    #[test]
    fn test_persistence_across_reopen() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();

        // Write data
        {
            let storage = OmenStorage::open(&path).unwrap();
            storage.put_vector(0, &[1.0, 2.0, 3.0]).unwrap();
            storage
                .put_metadata(0, &serde_json::json!({"test": true}))
                .unwrap();
            storage.put_id_mapping("doc1", 0).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let storage = OmenStorage::open(&path).unwrap();
            let vector = storage.get_vector(0).unwrap().unwrap();
            assert_eq!(vector, vec![1.0, 2.0, 3.0]);

            let metadata = storage.get_metadata(0).unwrap().unwrap();
            assert_eq!(metadata["test"], true);

            let index = storage.get_id_mapping("doc1").unwrap().unwrap();
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn test_batch_insert() {
        let temp_dir = TempDir::new().unwrap();
        let storage = OmenStorage::open(temp_dir.path()).unwrap();

        let items = vec![
            (
                0,
                "doc0".to_string(),
                vec![1.0, 2.0],
                serde_json::json!({"id": 0}),
            ),
            (
                1,
                "doc1".to_string(),
                vec![3.0, 4.0],
                serde_json::json!({"id": 1}),
            ),
        ];

        storage.put_batch(items).unwrap();

        assert_eq!(storage.get_vector(0).unwrap().unwrap(), vec![1.0, 2.0]);
        assert_eq!(storage.get_vector(1).unwrap().unwrap(), vec![3.0, 4.0]);
        assert_eq!(storage.get_id_mapping("doc0").unwrap().unwrap(), 0);
        assert_eq!(storage.get_id_mapping("doc1").unwrap().unwrap(), 1);
    }

    #[test]
    fn test_deleted_markers() {
        let temp_dir = TempDir::new().unwrap();
        let storage = OmenStorage::open(temp_dir.path()).unwrap();

        storage.put_deleted(5).unwrap();
        assert!(storage.is_deleted(5).unwrap());
        assert!(!storage.is_deleted(6).unwrap());

        storage.remove_deleted(5).unwrap();
        assert!(!storage.is_deleted(5).unwrap());
    }
}
