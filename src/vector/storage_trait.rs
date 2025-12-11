//! Storage backend trait for VectorStore
//!
//! Defines the interface that storage backends must implement.
//! Currently supported: SeerDBStorage, OmenStorage

use anyhow::Result;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;

/// Storage backend trait for vector persistence
pub trait Storage: Send + Sync {
    /// Store a vector by internal index
    fn put_vector(&self, id: usize, vector: &[f32]) -> Result<()>;

    /// Get a vector by internal index
    fn get_vector(&self, id: usize) -> Result<Option<Vec<f32>>>;

    /// Store metadata for a vector
    fn put_metadata(&self, id: usize, metadata: &JsonValue) -> Result<()>;

    /// Get metadata for a vector
    fn get_metadata(&self, id: usize) -> Result<Option<JsonValue>>;

    /// Store string ID to internal index mapping (bidirectional)
    fn put_id_mapping(&self, string_id: &str, index: usize) -> Result<()>;

    /// Get internal index for a string ID
    fn get_id_mapping(&self, string_id: &str) -> Result<Option<usize>>;

    /// Get string ID for an internal index (reverse lookup)
    fn get_string_id(&self, index: usize) -> Result<Option<String>>;

    /// Delete string ID mapping
    fn delete_id_mapping(&self, string_id: &str) -> Result<()>;

    /// Store configuration value
    fn put_config(&self, key: &str, value: u64) -> Result<()>;

    /// Get configuration value
    fn get_config(&self, key: &str) -> Result<Option<u64>>;

    /// Load all vectors from storage
    fn load_all_vectors(&self) -> Result<Vec<(usize, Vec<f32>)>>;

    /// Increment vector count
    fn increment_count(&self) -> Result<usize>;

    /// Get current vector count
    fn get_count(&self) -> Result<usize>;

    /// Store quantization mode (0=none, 1=sq8, 2=rabitq-4, 3=rabitq-2, 4=rabitq-8)
    fn put_quantization_mode(&self, mode: u64) -> Result<()>;

    /// Get quantization mode
    fn get_quantization_mode(&self) -> Result<Option<u64>>;

    /// Check if store was created with quantization
    fn is_quantized(&self) -> Result<bool>;

    /// Load all metadata from storage
    fn load_all_metadata(&self) -> Result<HashMap<usize, JsonValue>>;

    /// Load all ID mappings from storage
    fn load_all_id_mappings(&self) -> Result<HashMap<String, usize>>;

    /// Mark a vector as deleted
    fn put_deleted(&self, id: usize) -> Result<()>;

    /// Check if a vector is deleted
    fn is_deleted(&self, id: usize) -> Result<bool>;

    /// Remove deleted marker
    fn remove_deleted(&self, id: usize) -> Result<()>;

    /// Load all deleted IDs
    fn load_all_deleted(&self) -> Result<HashMap<usize, bool>>;

    /// Flush pending writes to disk
    fn flush(&self) -> Result<()>;

    /// Batch set vectors with metadata and ID mappings
    fn put_batch(&self, items: Vec<(usize, String, Vec<f32>, JsonValue)>) -> Result<()>;

    /// Get storage path
    fn path(&self) -> &Path;

    /// Get storage statistics (for profiling)
    fn stats_map(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
