//! Integration tests for storage layers
//!
//! Tests the complete workflow:
//! 1. Build index with `MemoryStorage`
//! 2. Serialize to disk with `DiskStorage`
//! 3. Load from disk and verify queries
//! 4. Test with `CachedStorage` for performance

#[cfg(test)]
mod tests {
    use crate::hnsw::cached_storage::CachedStorage;
    use crate::hnsw::disk_storage::DiskStorage;
    use crate::hnsw::node_storage::{MemoryStorage, NodeStorage};
    use std::num::NonZeroUsize;
    use tempfile::TempDir;

    /// Test basic storage layer workflow
    #[test]
    fn test_storage_workflow_basic() {
        // Step 1: Build graph in memory
        let mut memory_storage = MemoryStorage::new(8);

        // Simulate a small HNSW graph:
        // Node 0: entry point (3 levels)
        //   - Level 2: neighbors [1]
        //   - Level 1: neighbors [1, 2]
        //   - Level 0: neighbors [1, 2, 3]
        memory_storage.write_neighbors(0, 2, &[1]).unwrap();
        memory_storage.write_neighbors(0, 1, &[1, 2]).unwrap();
        memory_storage.write_neighbors(0, 0, &[1, 2, 3]).unwrap();

        // Node 1: 2 levels
        memory_storage.write_neighbors(1, 1, &[0, 2]).unwrap();
        memory_storage.write_neighbors(1, 0, &[0, 2, 3]).unwrap();

        // Node 2: 1 level
        memory_storage.write_neighbors(2, 0, &[0, 1, 3]).unwrap();

        // Node 3: 1 level
        memory_storage.write_neighbors(3, 0, &[0, 1, 2]).unwrap();

        // Verify memory storage works
        assert_eq!(memory_storage.len(), 4);
        assert_eq!(memory_storage.num_levels(0).unwrap(), 3);
        assert_eq!(memory_storage.num_levels(1).unwrap(), 2);
        assert_eq!(memory_storage.num_levels(2).unwrap(), 1);

        // Step 2: Serialize to disk
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("test_graph.bin");

        // Collect nodes from memory storage
        let nodes: Vec<Vec<Vec<u32>>> = (0..4)
            .map(|node_id| {
                let num_levels = memory_storage.num_levels(node_id as u32).unwrap();
                (0..num_levels)
                    .map(|level| {
                        memory_storage
                            .read_neighbors(node_id as u32, level as u8)
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        DiskStorage::create(&storage_path, &nodes, 2, 16).unwrap();

        // Step 3: Load from disk
        let disk_storage = DiskStorage::open(&storage_path, false).unwrap();

        // Verify disk storage matches memory storage
        assert_eq!(disk_storage.len(), 4);
        assert_eq!(disk_storage.num_levels(0).unwrap(), 3);
        assert_eq!(disk_storage.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(disk_storage.read_neighbors(0, 1).unwrap(), vec![1, 2]);
        assert_eq!(disk_storage.read_neighbors(0, 2).unwrap(), vec![1]);

        // Step 4: Wrap with cache
        let capacity = NonZeroUsize::new(100).unwrap();
        let cached_storage = CachedStorage::new(Box::new(disk_storage), capacity);

        // First read is cache miss
        assert_eq!(cached_storage.read_neighbors(1, 0).unwrap(), vec![0, 2, 3]);

        let (hits, misses, _) = cached_storage.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);

        // Second read is cache hit
        assert_eq!(cached_storage.read_neighbors(1, 0).unwrap(), vec![0, 2, 3]);

        let (hits, misses, hit_rate) = cached_storage.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.5).abs() < 0.01);
    }

    /// Test storage with 10K nodes
    #[test]
    fn test_storage_10k_nodes() {
        // Build a synthetic graph with 10K nodes
        let num_nodes = 10_000;
        let max_level = 5;
        let m = 16; // Neighbors per level

        let mut memory_storage = MemoryStorage::new(8);

        // Build graph: each node has random neighbors
        for node_id in 0..num_nodes {
            let num_levels = (node_id % (max_level + 1)) + 1; // 1-6 levels

            for level in 0..num_levels {
                // Generate neighbors (circular connections for simplicity)
                let neighbors: Vec<u32> = (1..=m)
                    .map(|offset| ((node_id + offset) % num_nodes) as u32)
                    .collect();

                memory_storage
                    .write_neighbors(node_id as u32, level as u8, &neighbors)
                    .unwrap();
            }
        }

        assert_eq!(memory_storage.len(), num_nodes);

        // Serialize to disk
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("test_10k.bin");

        // Collect all nodes
        let nodes: Vec<Vec<Vec<u32>>> = (0..num_nodes)
            .map(|node_id| {
                let num_levels = memory_storage.num_levels(node_id as u32).unwrap();
                (0..num_levels)
                    .map(|level| {
                        memory_storage
                            .read_neighbors(node_id as u32, level as u8)
                            .unwrap()
                    })
                    .collect()
            })
            .collect();

        DiskStorage::create(&storage_path, &nodes, max_level as u32, m as u32).unwrap();

        // Load from disk
        let disk_storage = DiskStorage::open(&storage_path, false).unwrap();

        assert_eq!(disk_storage.len(), num_nodes);

        // Verify random samples
        for sample in [0, 100, 1000, 5000, 9999] {
            let mem_neighbors = memory_storage.read_neighbors(sample, 0).unwrap();
            let disk_neighbors = disk_storage.read_neighbors(sample, 0).unwrap();
            assert_eq!(mem_neighbors, disk_neighbors);
        }

        // Test with cache (30% capacity = 10K nodes × 3 avg levels × 0.3 = 9K entries)
        let cache_capacity = NonZeroUsize::new(9_000).unwrap();
        let mut cached_storage = CachedStorage::new(Box::new(disk_storage), cache_capacity);

        // Read all nodes at level 0 (simulate search queries)
        for node_id in 0..100 {
            let _neighbors = cached_storage.read_neighbors(node_id as u32, 0).unwrap();
        }

        // After 100 reads, cache should have all entries (cache size 9K > 100)
        let (hits, misses, _hit_rate) = cached_storage.cache_stats();
        assert_eq!(misses, 100); // First 100 are misses
        assert_eq!(hits, 0); // No hits yet (first reads)

        // Read same nodes again
        for node_id in 0..100 {
            let _neighbors = cached_storage.read_neighbors(node_id as u32, 0).unwrap();
        }

        let (hits, misses, hit_rate) = cached_storage.cache_stats();
        assert_eq!(misses, 100); // Still 100 misses
        assert_eq!(hits, 100); // Now 100 hits
        assert!((hit_rate - 0.5).abs() < 0.01); // 50% hit rate

        // Verify cache stats
        assert_eq!(cached_storage.cache_size(), 100); // 100 entries cached
        assert_eq!(cached_storage.cache_capacity(), 9_000);

        // Clear cache and verify reset
        cached_storage.clear_cache();
        assert_eq!(cached_storage.cache_size(), 0);
    }

    /// Test disk storage persistence (save/load)
    #[test]
    fn test_disk_storage_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("persist_test.bin");

        // Create initial storage
        let nodes = vec![
            vec![vec![1, 2, 3]],                      // Node 0: 1 level
            vec![vec![0, 2], vec![0]],                // Node 1: 2 levels
            vec![vec![0, 1, 3], vec![0, 1], vec![0]], // Node 2: 3 levels
        ];

        DiskStorage::create(&storage_path, &nodes, 2, 16).unwrap();

        // Verify file exists
        assert!(storage_path.exists());

        // Load and verify
        let loaded = DiskStorage::open(&storage_path, false).unwrap();

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.num_levels(0).unwrap(), 1);
        assert_eq!(loaded.num_levels(1).unwrap(), 2);
        assert_eq!(loaded.num_levels(2).unwrap(), 3);

        assert_eq!(loaded.read_neighbors(0, 0).unwrap(), vec![1, 2, 3]);
        assert_eq!(loaded.read_neighbors(1, 0).unwrap(), vec![0, 2]);
        assert_eq!(loaded.read_neighbors(1, 1).unwrap(), vec![0]);
        assert_eq!(loaded.read_neighbors(2, 0).unwrap(), vec![0, 1, 3]);
        assert_eq!(loaded.read_neighbors(2, 1).unwrap(), vec![0, 1]);
        assert_eq!(loaded.read_neighbors(2, 2).unwrap(), vec![0]);
    }

    /// Test cache performance with different sizes
    #[test]
    fn test_cache_performance() {
        // Build a small graph
        let mut memory_storage = MemoryStorage::new(8);

        for node_id in 0..100 {
            memory_storage
                .write_neighbors(node_id, 0, &[1, 2, 3])
                .unwrap();
        }

        // Test with small cache (evictions expected)
        let small_capacity = NonZeroUsize::new(10).unwrap();
        let small_cache = CachedStorage::new(Box::new(memory_storage.clone()), small_capacity);

        // Read 50 nodes (will evict)
        for node_id in 0..50 {
            let _neighbors = small_cache.read_neighbors(node_id, 0).unwrap();
        }

        let (_, misses, _) = small_cache.cache_stats();
        assert_eq!(misses, 50); // All misses (first reads)

        // Read first 10 again (should hit cache - no evictions yet since we read 50 sequentially)
        for node_id in 40..50 {
            let _neighbors = small_cache.read_neighbors(node_id, 0).unwrap();
        }

        let (hits, _, _) = small_cache.cache_stats();
        assert!(hits > 0); // Some hits (recent nodes still in cache)

        // Test with large cache (no evictions)
        let large_capacity = NonZeroUsize::new(1000).unwrap();
        let large_cache = CachedStorage::new(Box::new(memory_storage), large_capacity);

        // Read all 100 nodes
        for node_id in 0..100 {
            let _neighbors = large_cache.read_neighbors(node_id, 0).unwrap();
        }

        // Read all again (100% hit rate expected)
        for node_id in 0..100 {
            let _neighbors = large_cache.read_neighbors(node_id, 0).unwrap();
        }

        let (hits, misses, hit_rate) = large_cache.cache_stats();
        assert_eq!(misses, 100); // First 100 reads
        assert_eq!(hits, 100); // Second 100 reads
        assert!((hit_rate - 0.5).abs() < 0.01); // 50% hit rate
    }

    /// Test memory usage tracking
    #[test]
    fn test_memory_usage_tracking() {
        // Memory storage
        let mut memory_storage = MemoryStorage::new(8);
        let initial_mem = memory_storage.memory_usage();

        memory_storage
            .write_neighbors(0, 0, &[1, 2, 3, 4, 5])
            .unwrap();
        let after_mem = memory_storage.memory_usage();

        assert!(after_mem > initial_mem);

        // Disk storage (minimal memory - just mmap handle)
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().join("mem_test.bin");

        let nodes = vec![vec![vec![1, 2, 3]]];
        DiskStorage::create(&storage_path, &nodes, 0, 16).unwrap();

        let disk_storage = DiskStorage::open(&storage_path, false).unwrap();
        let disk_mem = disk_storage.memory_usage();

        // Disk storage should use minimal memory (just metadata, not data)
        assert!(disk_mem < 1024); // <1KB for metadata only

        // Cached storage (backend + cache) - use MemoryStorage for writes
        let capacity = NonZeroUsize::new(100).unwrap();
        let writable_backend = MemoryStorage::new(8);
        let mut cached_storage = CachedStorage::new(Box::new(writable_backend), capacity);

        let initial_cache_mem = cached_storage.memory_usage();

        // Add some entries to cache (via writes)
        for i in 0..10 {
            cached_storage.write_neighbors(i, 0, &[1, 2, 3]).unwrap();
        }

        let after_cache_mem = cached_storage.memory_usage();
        assert!(after_cache_mem > initial_cache_mem);
    }
}
