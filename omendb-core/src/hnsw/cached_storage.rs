//! LRU-cached storage wrapper for HNSW-IF
//!
//! Implements the `CachedStorage` layer that wraps any `NodeStorage` backend
//! (`MemoryStorage` or `DiskStorage`) with an LRU cache.
//!
//! Key design decisions (from Gorgeous 2025 paper):
//! - **Cache adjacency lists only** (not full nodes): 88% hit rate vs 10%
//! - **Read-through caching**: Miss → load from backend → cache
//! - **Write-through caching**: Write to backend + cache
//! - **LRU eviction**: Simple and effective for HNSW access patterns
//!
//! Automatic tiering:
//! - <10M vectors: `MemoryStorage` (no cache needed)
//! - 10M-100M: CachedStorage(DiskStorage) with 30% cache
//! - 100M+: CachedStorage(DiskStorage) with 10% cache

use super::error::Result;
use super::node_storage::{Level, NodeId, NodeStorage};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Cache key: (`node_id`, level)
///
/// We cache individual adjacency lists, not full nodes.
/// This gives 88% hit rate vs 10% for full nodes (Gorgeous 2025).
type CacheKey = (NodeId, Level);

/// Cached storage wrapper with LRU eviction
///
/// Wraps any `NodeStorage` backend (`MemoryStorage` or `DiskStorage`) with an LRU cache.
///
/// # Example
/// ```ignore
/// // Create disk storage
/// let disk_storage = DiskStorage::open("graph.bin", false)?;
///
/// // Wrap with 1M entry cache (30% of 10M nodes × 3 levels)
/// let cached_storage = CachedStorage::new(
///     Box::new(disk_storage),
///     NonZeroUsize::new(1_000_000).unwrap(),
/// );
///
/// // Use like any NodeStorage
/// let neighbors = cached_storage.read_neighbors(42, 0)?;
/// ```
pub struct CachedStorage {
    /// Underlying storage backend
    backend: Box<dyn NodeStorage>,

    /// LRU cache for adjacency lists
    ///
    /// Key: (`node_id`, level)
    /// Value: Vec<NodeId> (neighbor list)
    ///
    /// Mutex because LRU cache needs mutable access for `get()` (updates LRU order).
    /// Alternative: `RwLock` + separate dirty tracking, but more complex.
    cache: Arc<Mutex<LruCache<CacheKey, Vec<NodeId>>>>,

    /// Cache statistics
    hits: Arc<Mutex<usize>>,
    misses: Arc<Mutex<usize>>,
}

impl CachedStorage {
    /// Create new cached storage wrapper
    ///
    /// # Arguments
    /// * `backend` - Underlying storage (`MemoryStorage` or `DiskStorage`)
    /// * `capacity` - Maximum number of cached adjacency lists
    ///
    /// # Capacity Guidelines
    /// - 10M-100M vectors: 30% cache = 10M nodes × 3 avg levels × 0.3 = 9M entries
    /// - 100M+ vectors: 10% cache = 100M nodes × 3 avg levels × 0.1 = 30M entries
    /// - Memory per entry: ~50 bytes (16 neighbors × 4 bytes + overhead)
    ///
    /// # Example
    /// ```ignore
    /// // 10M vectors, 30% cache, 3 levels avg
    /// let capacity = NonZeroUsize::new(10_000_000 * 3 * 30 / 100).unwrap();
    /// let cached = CachedStorage::new(Box::new(disk_storage), capacity);
    /// ```
    #[must_use]
    pub fn new(backend: Box<dyn NodeStorage>, capacity: NonZeroUsize) -> Self {
        Self {
            backend,
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Get cache statistics
    ///
    /// Returns (hits, misses, `hit_rate`)
    #[must_use]
    pub fn cache_stats(&self) -> (usize, usize, f64) {
        let hits = *self.hits.lock().unwrap();
        let misses = *self.misses.lock().unwrap();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, hit_rate)
    }

    /// Reset cache statistics
    pub fn reset_stats(&mut self) {
        *self.hits.lock().unwrap() = 0;
        *self.misses.lock().unwrap() = 0;
    }

    /// Clear cache (for testing)
    pub fn clear_cache(&mut self) {
        self.cache.lock().unwrap().clear();
    }

    /// Get cache size (current number of entries)
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get cache capacity (max entries)
    #[must_use]
    pub fn cache_capacity(&self) -> usize {
        self.cache.lock().unwrap().cap().get()
    }
}

impl NodeStorage for CachedStorage {
    fn read_neighbors(&self, node_id: NodeId, level: Level) -> Result<Vec<NodeId>> {
        let key = (node_id, level);

        // Try cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(neighbors) = cache.get(&key) {
                *self.hits.lock().unwrap() += 1;
                return Ok(neighbors.clone());
            }
        }

        // Cache miss - load from backend
        *self.misses.lock().unwrap() += 1;
        let neighbors = self.backend.read_neighbors(node_id, level)?;

        // Store in cache (read-through)
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(key, neighbors.clone());
        }

        Ok(neighbors)
    }

    fn write_neighbors(
        &mut self,
        node_id: NodeId,
        level: Level,
        neighbors: &[NodeId],
    ) -> Result<()> {
        // Write to backend first
        self.backend.write_neighbors(node_id, level, neighbors)?;

        // Update cache (write-through)
        let key = (node_id, level);
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, neighbors.to_vec());

        Ok(())
    }

    fn exists(&self, node_id: NodeId) -> bool {
        self.backend.exists(node_id)
    }

    fn num_levels(&self, node_id: NodeId) -> Result<usize> {
        self.backend.num_levels(node_id)
    }

    fn len(&self) -> usize {
        self.backend.len()
    }

    fn is_empty(&self) -> bool {
        self.backend.is_empty()
    }

    fn flush(&mut self) -> Result<()> {
        self.backend.flush()
    }

    fn memory_usage(&self) -> usize {
        // Backend memory + cache memory
        let backend_mem = self.backend.memory_usage();

        let cache_mem = {
            let cache = self.cache.lock().unwrap();
            let num_entries = cache.len();

            // Estimate memory per entry:
            // - CacheKey: 8 bytes (u32 + u32)
            // - Vec<NodeId>: ~50 bytes avg (16 neighbors × 4 bytes + overhead)
            num_entries * (8 + 50)
        };

        backend_mem + cache_mem
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::hnsw::node_storage::MemoryStorage;

    #[test]
    fn test_cached_storage_basic() {
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(100).unwrap();
        let mut cached = CachedStorage::new(backend, capacity);

        // Write some data
        cached.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        cached.write_neighbors(0, 1, &[4, 5]).unwrap();

        // Read back (should be cached)
        assert_eq!(cached.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(cached.read_neighbors(0, 1).unwrap(), vec![4, 5]);

        // Check stats
        let (hits, misses, hit_rate) = cached.cache_stats();
        assert_eq!(hits, 2); // Both reads hit cache
        assert_eq!(misses, 0);
        assert_eq!(hit_rate, 1.0);
    }

    #[test]
    fn test_cached_storage_cache_miss() {
        let mut backend = MemoryStorage::new(8);
        backend.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        let backend = Box::new(backend);
        let capacity = NonZeroUsize::new(100).unwrap();
        let cached = CachedStorage::new(backend, capacity);

        // First read is cache miss
        assert_eq!(cached.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);

        let (hits, misses, _) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Second read is cache hit
        assert_eq!(cached.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);

        let (hits, misses, hit_rate) = cached.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cached_storage_eviction() {
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(2).unwrap(); // Small cache
        let mut cached = CachedStorage::new(backend, capacity);

        // Write 3 entries (will evict first)
        cached.write_neighbors(0, 0, &[1, 2]).unwrap();
        cached.write_neighbors(1, 0, &[3, 4]).unwrap();
        cached.write_neighbors(2, 0, &[5, 6]).unwrap(); // Evicts (0,0)

        // Cache size should be capped at 2
        assert_eq!(cached.cache_size(), 2);

        // Read evicted entry (cache miss)
        cached.read_neighbors(0, 0).unwrap();
        let (_hits, misses, _) = cached.cache_stats();
        assert_eq!(misses, 1); // First read of evicted entry
    }

    #[test]
    fn test_cached_storage_exists() {
        let mut backend = MemoryStorage::new(8);
        backend.write_neighbors(0, 0, &[1]).unwrap();

        let backend = Box::new(backend);
        let capacity = NonZeroUsize::new(100).unwrap();
        let cached = CachedStorage::new(backend, capacity);

        assert!(cached.exists(0));
        assert!(!cached.exists(1));
    }

    #[test]
    fn test_cached_storage_num_levels() {
        let mut backend = MemoryStorage::new(8);
        backend.write_neighbors(0, 0, &[1]).unwrap();
        backend.write_neighbors(0, 1, &[2]).unwrap();
        backend.write_neighbors(0, 2, &[3]).unwrap();

        let backend = Box::new(backend);
        let capacity = NonZeroUsize::new(100).unwrap();
        let cached = CachedStorage::new(backend, capacity);

        assert_eq!(cached.num_levels(0).unwrap(), 3);
    }

    #[test]
    fn test_cached_storage_len() {
        let mut backend = MemoryStorage::new(8);
        backend.write_neighbors(0, 0, &[1]).unwrap();
        backend.write_neighbors(1, 0, &[2]).unwrap();

        let backend = Box::new(backend);
        let capacity = NonZeroUsize::new(100).unwrap();
        let cached = CachedStorage::new(backend, capacity);

        assert_eq!(cached.len(), 2);
        assert!(!cached.is_empty());
    }

    #[test]
    fn test_cached_storage_clear_cache() {
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(100).unwrap();
        let mut cached = CachedStorage::new(backend, capacity);

        // Write some data
        cached.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        assert_eq!(cached.cache_size(), 1);

        // Clear cache
        cached.clear_cache();

        assert_eq!(cached.cache_size(), 0);

        // Next read is cache miss
        cached.read_neighbors(0, 0).unwrap();
        let (hits, misses, _) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_cached_storage_reset_stats() {
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(100).unwrap();
        let mut cached = CachedStorage::new(backend, capacity);

        // Generate some stats
        cached.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        cached.read_neighbors(0, 0).unwrap();

        let (hits, misses, _) = cached.cache_stats();
        assert!(hits > 0 || misses > 0);

        // Reset stats
        cached.reset_stats();

        let (hits, misses, _) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn test_cached_storage_memory_usage() {
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(100).unwrap();
        let mut cached = CachedStorage::new(backend, capacity);

        let initial = cached.memory_usage();

        // Add some data
        cached.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        cached.write_neighbors(1, 0, &[4, 5, 6]).unwrap();

        let after = cached.memory_usage();
        assert!(after > initial);
    }

    #[test]
    fn test_node_storage_trait() {
        // Test that CachedStorage implements NodeStorage trait
        let backend = Box::new(MemoryStorage::new(8));
        let capacity = NonZeroUsize::new(100).unwrap();
        let mut storage: Box<dyn NodeStorage> = Box::new(CachedStorage::new(backend, capacity));

        storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();
        let neighbors = storage.read_neighbors(0, 0).unwrap();

        assert_eq!(neighbors, vec![1, 2, 3]);
        assert!(storage.exists(0));
        assert_eq!(storage.len(), 1);
    }
}
