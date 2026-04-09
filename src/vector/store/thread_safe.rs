//! Thread-safe wrapper for VectorStore
//!
//! Provides `ThreadSafeVectorStore` - a thin `Arc<VectorStore>` wrapper for
//! sharing an internally synchronized store across threads.
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
//! // Mutations are synchronized inside VectorStore
//! store.read().set("id1", vec, metadata).unwrap();
//! ```

use super::{SearchResult, VectorStore, VectorStoreOptions};
use crate::catalog::CollectionSchema;
use crate::vector::types::Vector;
use anyhow::Result;
use serde_json::Value as JsonValue;
use std::path::Path;
use std::sync::Arc;

/// Thread-safe wrapper for `VectorStore`
///
/// Uses `Arc<VectorStore>` internally and relies on `VectorStore`'s internal
/// synchronization for safe concurrent access.
///
/// For basic operations, use the convenience methods directly on this type.
/// For advanced operations, use `read()` to access the underlying `VectorStore`.
///
/// # Example
///
/// ```ignore
/// use omendb::ThreadSafeVectorStore;
///
/// let store = ThreadSafeVectorStore::new(128);
///
/// // Basic write
/// store.set("id1", vec, metadata)?;
///
/// // Basic search
/// let results = store.search(&query, 10)?;
///
/// // Advanced: use read() for full API access
/// let results = store.read().search_with_options(&query, 10, None, None, None)?;
/// ```
#[derive(Clone)]
pub struct ThreadSafeVectorStore {
    inner: Arc<VectorStore>,
}

impl ThreadSafeVectorStore {
    /// Create new thread-safe vector store
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            inner: Arc::new(VectorStore::new(dimensions)),
        }
    }

    /// Open existing store from path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(VectorStore::open(path)?),
        })
    }

    /// Create a new persistent store from an explicit collection schema.
    pub fn create(path: impl AsRef<Path>, schema: CollectionSchema) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(VectorStore::create(path, schema)?),
        })
    }

    /// Create a new in-memory store from an explicit collection schema.
    pub fn create_in_memory(schema: CollectionSchema) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(VectorStore::create_in_memory(schema)?),
        })
    }

    /// Build with options
    pub fn build_with_options(options: &VectorStoreOptions) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(VectorStore::build_with_options(options)?),
        })
    }

    /// Wrap an existing VectorStore
    #[must_use]
    pub fn from_store(store: VectorStore) -> Self {
        Self {
            inner: Arc::new(store),
        }
    }

    /// Insert vector with ID and metadata
    pub fn set(&self, id: &str, vector: Vector, metadata: JsonValue) -> Result<usize> {
        self.inner.set(id, vector, metadata)
    }

    /// Batch insert
    pub fn set_batch<S: Into<String>>(
        &self,
        batch: Vec<(S, Vector, JsonValue)>,
    ) -> Result<Vec<usize>> {
        self.inner.set_batch(batch)
    }

    /// Delete by ID
    pub fn delete(&self, id: &str) -> Result<()> {
        self.inner.delete(id)
    }

    /// Flush to disk
    pub fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        self.inner.search(query, k, None)
    }

    /// Get by ID
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        self.inner.get(id)
    }

    /// Check if ID exists
    pub fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get all IDs
    pub fn ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Get shared access to the underlying store for advanced operations.
    ///
    /// `VectorStore` handles its own synchronization internally.
    pub fn read(&self) -> &VectorStore {
        &self.inner
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
                    &format!("vec{i}"),
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
                        &format!("vec{i}"),
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
                "vec1",
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
                "vec1",
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
