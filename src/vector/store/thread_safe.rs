//! Thread-safe wrapper for VectorStore
//!
//! Provides `ThreadSafeVectorStore` - a wrapper that uses `Arc<RwLock<VectorStore>>`
//! for safe concurrent access across threads.
//!
//! # Usage
//!
//! ```ignore
//! use omendb::ThreadSafeVectorStore;
//!
//! let store = ThreadSafeVectorStore::new(128);
//!
//! // Clone for multiple threads
//! let store2 = store.clone();
//!
//! // Concurrent reads
//! std::thread::spawn(move || {
//!     let results = store2.read().search(&query, 10).unwrap();
//! });
//!
//! // Writes are exclusive
//! store.write().set("id1".to_string(), vec, metadata).unwrap();
//! ```

use super::{SearchResult, VectorStore, VectorStoreOptions};
use crate::vector::types::Vector;
use anyhow::Result;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::path::Path;
use std::sync::Arc;

/// Thread-safe wrapper for `VectorStore`
///
/// Uses `Arc<RwLock<VectorStore>>` internally for safe concurrent access.
///
/// For basic operations, use the convenience methods directly on this type.
/// For advanced operations, use `read()` or `write()` to get access to the
/// underlying `VectorStore`.
///
/// # Example
///
/// ```ignore
/// use omendb::ThreadSafeVectorStore;
///
/// let store = ThreadSafeVectorStore::new(128);
///
/// // Basic write
/// store.set("id1".to_string(), vec, metadata)?;
///
/// // Basic search
/// let results = store.search(&query, 10)?;
///
/// // Advanced: use read() for full API access
/// let results = store.read().search_with_options(&query, 10, None, None, None)?;
/// ```
#[derive(Clone)]
pub struct ThreadSafeVectorStore {
    inner: Arc<RwLock<VectorStore>>,
}

impl ThreadSafeVectorStore {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create new thread-safe vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VectorStore::new(dimensions))),
        }
    }

    /// Open existing store from path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(VectorStore::open(path)?)),
        })
    }

    /// Build with options
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(VectorStore::build_with_options(options)?)),
        })
    }

    /// Wrap an existing VectorStore
    #[must_use]
    pub fn from_store(store: VectorStore) -> Self {
        Self {
            inner: Arc::new(RwLock::new(store)),
        }
    }

    // ========================================================================
    // Basic Write Operations (exclusive lock)
    // ========================================================================

    /// Insert vector with ID and metadata
    pub fn set(&self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        self.inner.write().set(id, vector, metadata)
    }

    /// Batch insert
    pub fn set_batch(&self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        self.inner.write().set_batch(batch)
    }

    /// Delete by ID
    pub fn delete(&self, id: &str) -> Result<()> {
        self.inner.write().delete(id)
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.inner.write().flush()
    }

    // ========================================================================
    // Basic Read Operations (shared lock)
    // ========================================================================

    /// Search for k nearest neighbors
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        self.inner.read().search(query, k, None)
    }

    /// Get by ID
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        self.inner.read().get(id)
    }

    /// Check if ID exists
    pub fn contains(&self, id: &str) -> bool {
        self.inner.read().contains(id)
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Get all IDs
    pub fn ids(&self) -> Vec<String> {
        self.inner.read().ids()
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.inner.read().records.dimensions() as usize
    }

    // ========================================================================
    // Direct Access (for advanced use)
    // ========================================================================

    /// Get read lock to underlying store for advanced operations
    ///
    /// Use this for search methods with filters, ef parameters, hybrid search, etc.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, VectorStore> {
        self.inner.read()
    }

    /// Get write lock to underlying store for advanced operations
    ///
    /// Use this for batch operations, rebuild_index, optimize, etc.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, VectorStore> {
        self.inner.write()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_concurrent_reads() {
        let store = ThreadSafeVectorStore::new(3);

        // Insert some data
        for i in 0..100 {
            store
                .set(
                    format!("vec{i}"),
                    Vector::new(vec![i as f32, 0.0, 0.0]),
                    serde_json::json!({"i": i}),
                )
                .unwrap();
        }

        // Spawn multiple reader threads
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = store.clone();
                thread::spawn(move || {
                    let query = Vector::new(vec![50.0, 0.0, 0.0]);
                    for _ in 0..10 {
                        let results = store.search(&query, 5).unwrap();
                        assert_eq!(results.len(), 5);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_write_read() {
        let store = ThreadSafeVectorStore::new(3);

        // Writer thread
        let writer_store = store.clone();
        let writer = thread::spawn(move || {
            for i in 0..50 {
                writer_store
                    .set(
                        format!("vec{i}"),
                        Vector::new(vec![i as f32, 0.0, 0.0]),
                        serde_json::json!({"i": i}),
                    )
                    .unwrap();
            }
        });

        // Reader thread
        let reader_store = store.clone();
        let reader = thread::spawn(move || {
            let query = Vector::new(vec![25.0, 0.0, 0.0]);
            for _ in 0..20 {
                // May return fewer results if writer hasn't finished
                let _ = reader_store.search(&query, 5);
                thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        assert_eq!(store.len(), 50);
    }

    #[test]
    fn test_clone_shares_state() {
        let store1 = ThreadSafeVectorStore::new(3);
        let store2 = store1.clone();

        store1
            .set(
                "vec1".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();

        // store2 sees the same data
        assert!(store2.contains("vec1"));
        assert_eq!(store2.len(), 1);
    }

    #[test]
    fn test_advanced_via_read() {
        let store = ThreadSafeVectorStore::new(3);

        store
            .set(
                "vec1".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0]),
                serde_json::json!({"category": "a"}),
            )
            .unwrap();

        // Use read() for advanced search
        let query = Vector::new(vec![1.0, 0.0, 0.0]);
        let results = store
            .read()
            .search_with_options(&query, 10, None, None, None)
            .unwrap();
        assert_eq!(results.len(), 1);
    }
}
