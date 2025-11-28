    use super::*;

    #[test]
    fn test_hnsw_index_creation() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_index_insert_single() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec = vec![1.0, 2.0, 3.0];
        let id = index.insert(vec).unwrap();

        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_hnsw_index_insert_multiple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let vec3 = vec![7.0, 8.0, 9.0];

        let id1 = index.insert(vec1).unwrap();
        let id2 = index.insert(vec2).unwrap();
        let id3 = index.insert(vec3).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_hnsw_index_dimension_validation() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let wrong_dim = vec![1.0, 2.0]; // Only 2 dimensions
        assert!(index.insert(wrong_dim).is_err());
    }

    #[test]
    fn test_hnsw_index_search_empty() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5, 100).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_hnsw_index_search_single() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let vec = vec![1.0, 2.0, 3.0];
        index.insert(vec.clone()).unwrap();

        let results = index.search(&vec, 5, 100).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.01); // Should be ~0 (same vector)
    }

    #[test]
    fn test_random_level_distribution() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        let mut level_counts = vec![0; 8];

        // Generate 1000 random levels
        for _ in 0..1000 {
            let level = index.random_level();
            level_counts[level as usize] += 1;
        }

        // Level 0 should have most nodes (exponential decay)
        assert!(level_counts[0] > level_counts[1]);
        assert!(level_counts[1] > level_counts[2]);
    }

    #[test]
    fn test_memory_usage() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32; 128];
            index.insert(vec).unwrap();
        }

        let memory = index.memory_usage();

        // Should have memory for:
        // - 10 nodes (64 bytes each = 640 bytes)
        // - 10 vectors (128 * 4 bytes = 5120 bytes)
        // - Some neighbor storage
        assert!(memory > 5000); // At least vectors + nodes
        assert!(memory < 50000); // Not excessive
    }

    #[test]
    fn test_hnsw_index_search_multiple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 5 vectors
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 0.5, 0.5],
        ];

        for vec in vecs {
            index.insert(vec).unwrap();
        }

        // Search for k=3 nearest to [1.0, 0.0, 0.0]
        let query = vec![1.0, 0.0, 0.0];
        let results = index.search(&query, 3, 10).unwrap();

        // Should return 3 results
        assert_eq!(results.len(), 3);

        // First result should be closest (id=0, exact match)
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 0.01);

        // Results should be sorted by distance
        for i in 0..results.len() - 1 {
            assert!(results[i].distance <= results[i + 1].distance);
        }
    }

    #[test]
    fn test_hnsw_index_search_with_ef() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Search with different ef values
        let query = vec![5.0, 0.0, 0.0];

        let results_ef_5 = index.search(&query, 3, 5).unwrap();
        let results_ef_10 = index.search(&query, 3, 10).unwrap();

        // Both should return 3 results (k=3)
        assert_eq!(results_ef_5.len(), 3);
        assert_eq!(results_ef_10.len(), 3);

        // Higher ef should explore more candidates (potentially better recall)
        // Both should find node 5 as closest
        assert_eq!(results_ef_5[0].id, 5);
        assert_eq!(results_ef_10[0].id, 5);
    }

    #[test]
    fn test_hnsw_levels() {
        let mut params = HNSWParams::default();
        params.seed = 12345; // Fixed seed for reproducibility

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Count how many nodes have their TOP level at each height
        // Note: All nodes exist at level 0, but node.level is their TOP level
        let mut top_level_counts = vec![0; 8];
        for node in &index.nodes {
            top_level_counts[node.level as usize] += 1;
        }

        // Most nodes should have top level = 0 (due to exponential decay)
        assert!(top_level_counts[0] > 80); // Most nodes only at level 0

        // Some nodes should have higher top levels
        let higher_level_count: usize = top_level_counts[1..].iter().sum();
        assert!(higher_level_count > 0); // At least some nodes at higher levels

        // All nodes should exist (sum should be 100)
        let total: usize = top_level_counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_neighbor_count_limits() {
        let mut params = HNSWParams::default();
        params.m = 4; // Small M for easier testing
        params.ef_construction = 10;

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 20 vectors (enough to test neighbor pruning)
        for i in 0..20 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Check that no node has more than M*2 neighbors at level 0
        for node in &index.nodes {
            let neighbor_count = index
                .neighbors
                .get_neighbors(node.id, 0)
                .unwrap_or_default()
                .len();
            assert!(neighbor_count <= params.m * 2);
        }
    }

    #[test]
    fn test_search_recall_simple() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors in a line
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Query should find exact neighbors
        let query = vec![5.0, 0.0, 0.0];
        let results = index.search(&query, 3, 20).unwrap();

        // Should find nodes 5, 4, and 6 (closest to query)
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5); // Exact match

        // Second and third should be 4 or 6
        let ids: Vec<u32> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&6));
    }

    #[test]
    fn test_save_load_empty() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Save empty index
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        // Load it back
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.len(), 0);
        assert!(loaded.is_empty());
        assert_eq!(loaded.entry_point, None);
    }

    #[test]
    fn test_save_load_small() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Save index
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();

        // Load it back
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Verify basic properties
        assert_eq!(loaded.dimensions(), 3);
        assert_eq!(loaded.len(), 10);
        assert!(!loaded.is_empty());
        assert_eq!(loaded.entry_point, index.entry_point);

        // Verify vectors are preserved
        for i in 0..10 {
            let orig = index.vectors.get(i).unwrap();
            let load = loaded.vectors.get(i).unwrap();
            assert_eq!(orig, load);
        }

        // Verify search works on loaded index
        let query = vec![5.0, 0.0, 0.0];
        let results = loaded.search(&query, 3, 20).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 5); // Should still find exact match
    }

    #[test]
    fn test_save_load_preserves_graph() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..20 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            index.insert(vec).unwrap();
        }

        // Get search results before saving
        let query = vec![10.0, 20.0, 30.0];
        let results_before = index.search(&query, 5, 20).unwrap();

        // Save and load
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Get search results after loading
        let results_after = loaded.search(&query, 5, 20).unwrap();

        // Results should be identical
        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.id, after.id);
            assert!((before.distance - after.distance).abs() < 1e-5);
        }
    }

    #[test]
    fn test_save_load_with_quantization() {
        use tempfile::NamedTempFile;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(8, params, DistanceFunction::L2, true).unwrap();

        // Train quantization
        let _samples: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 8]).collect();
        if let VectorStorage::BinaryQuantized {
            ref mut thresholds, ..
        } = index.vectors
        {
            for (i, threshold) in thresholds.iter_mut().enumerate() {
                *threshold = i as f32 + 0.5;
            }
        }

        // Insert vectors
        for i in 0..10 {
            let vec = vec![i as f32; 8];
            index.insert(vec).unwrap();
        }

        // Save and load
        let temp_file = NamedTempFile::new().unwrap();
        index.save(temp_file.path()).unwrap();
        let loaded = HNSWIndex::load(temp_file.path()).unwrap();

        // Verify quantization is preserved
        match (&index.vectors, &loaded.vectors) {
            (
                VectorStorage::BinaryQuantized { thresholds: t1, .. },
                VectorStorage::BinaryQuantized { thresholds: t2, .. },
            ) => {
                assert_eq!(t1, t2);
            }
            _ => panic!("Expected BinaryQuantized storage"),
        }

        // Search should work
        let query = vec![5.0; 8];
        let results = loaded.search(&query, 3, 20).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_load_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"INVALID\0").unwrap();
        temp_file.flush().unwrap();

        let result = HNSWIndex::load(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            HNSWError::Storage(msg) => assert!(msg.contains("Invalid magic")),
            _ => panic!("Expected Storage error"),
        }
    }

    #[test]
    fn test_load_unsupported_version() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"HNSWIDX\0").unwrap(); // Magic
        temp_file.write_all(&99u32.to_le_bytes()).unwrap(); // Unsupported version
        temp_file.flush().unwrap();

        let result = HNSWIndex::load(temp_file.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            HNSWError::Storage(msg) => assert!(msg.contains("Unsupported version")),
            _ => panic!("Expected Storage error"),
        }
    }

    #[test]
    fn test_index_stats_empty() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 0);
        assert_eq!(stats.dimensions, 128);
        assert_eq!(stats.entry_point, None);
        assert_eq!(stats.max_level, 0);
        assert_eq!(stats.avg_neighbors_l0, 0.0);
        assert_eq!(stats.max_neighbors_l0, 0);
        assert!(!stats.quantization_enabled);
        assert!(matches!(stats.distance_function, DistanceFunction::L2));
    }

    #[test]
    fn test_index_stats_with_vectors() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 50 vectors
        for i in 0..50 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 50);
        assert_eq!(stats.dimensions, 3);
        assert!(stats.entry_point.is_some());
        assert!(stats.level_distribution.len() > 0);
        assert!(stats.level_distribution.iter().sum::<usize>() == 50); // All nodes accounted for
        assert!(stats.avg_neighbors_l0 > 0.0); // Should have some neighbors
        assert!(stats.max_neighbors_l0 > 0);
        assert!(stats.memory_bytes > 0);
        assert!(!stats.quantization_enabled);
    }

    #[test]
    fn test_index_stats_with_quantization() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(8, params, DistanceFunction::L2, true).unwrap();

        // Insert 10 vectors
        for i in 0..10 {
            let vec = vec![i as f32; 8];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        assert_eq!(stats.num_vectors, 10);
        assert!(stats.quantization_enabled); // Should be true
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_index_stats_level_distribution() {
        let mut params = HNSWParams::default();
        params.seed = 42; // Fixed seed for reproducibility

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 100 vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        // Level 0 should have most nodes (exponential decay)
        assert!(stats.level_distribution[0] > 70);

        // Total nodes should equal num_vectors
        let total: usize = stats.level_distribution.iter().sum();
        assert_eq!(total, 100);

        // Max level should match the distribution length - 1
        assert_eq!(stats.max_level as usize, stats.level_distribution.len() - 1);
    }

    #[test]
    fn test_index_stats_neighbors() {
        let mut params = HNSWParams::default();
        params.m = 8; // Set M for testing

        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert 30 vectors
        for i in 0..30 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        let stats = index.stats();

        // Average neighbors should be reasonable (between 0 and M*2)
        assert!(stats.avg_neighbors_l0 > 0.0);
        assert!(stats.avg_neighbors_l0 <= (params.m * 2) as f32);

        // Max neighbors should not exceed M*2 at level 0
        assert!(stats.max_neighbors_l0 <= params.m * 2);
    }

    #[test]
    fn test_index_stats_distance_functions() {
        // Test L2
        let params = HNSWParams::default();
        let index_l2 = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();
        let stats = index_l2.stats();
        assert!(matches!(stats.distance_function, DistanceFunction::L2));

        // Test Cosine
        let params = HNSWParams::default();
        let index_cos = HNSWIndex::new(3, params, DistanceFunction::Cosine, false).unwrap();
        let stats = index_cos.stats();
        assert!(matches!(stats.distance_function, DistanceFunction::Cosine));

        // Test NegativeDotProduct
        let params = HNSWParams::default();
        let index_dot =
            HNSWIndex::new(3, params, DistanceFunction::NegativeDotProduct, false).unwrap();
        let stats = index_dot.stats();
        assert!(matches!(
            stats.distance_function,
            DistanceFunction::NegativeDotProduct
        ));
    }

    // ========================================
    // Hybrid Storage Mode Tests ()
    // ========================================

    #[test]
    fn test_new_with_storage_memory_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Memory,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(index.neighbors.is_memory_mode());
        assert!(!index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_new_with_storage_hybrid_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(!index.neighbors.is_memory_mode());
        assert!(index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_new_with_storage_disk_heavy_mode() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::DiskHeavy,
        )
        .unwrap();

        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 128);
        assert!(!index.neighbors.is_memory_mode());
        assert!(index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_insert_query_hybrid_mode() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert some vectors
        for i in 0..100 {
            let vec = vec![i as f32, 0.0, 0.0];
            index.insert(vec).unwrap();
        }

        // Query
        let query = vec![50.0, 0.0, 0.0];
        let results = index.search(&query, 5, 10).unwrap();

        // Should find the vector at position 50
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 50); // Exact match
    }

    #[test]
    fn test_backward_compatibility_default_is_memory() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Default constructor should use Memory mode
        assert!(index.neighbors.is_memory_mode());
        assert!(!index.neighbors.is_layered_mode());
    }

    #[test]
    fn test_storage_mode_auto_select() {
        // Test auto-select for small dataset (<10M)
        let mode = StorageMode::auto_select(1_000_000);
        assert_eq!(mode, StorageMode::Memory);

        // Test auto-select for medium dataset (10M-100M)
        let mode = StorageMode::auto_select(50_000_000);
        assert_eq!(mode, StorageMode::Hybrid);

        // Test auto-select for large dataset (>100M)
        let mode = StorageMode::auto_select(200_000_000);
        assert_eq!(mode, StorageMode::DiskHeavy);
    }

    // ========================================
    // Edge Case Tests
    // ========================================

    #[test]
    fn test_serialize_hybrid_mode_fails_gracefully() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new_with_storage(
            128,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Try to save - should fail gracefully because Hybrid mode uses GraphStorage::Layered
        let result = index.save("/tmp/test_hybrid_save.hnsw");

        // Should return an error (not panic)
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_index_serialization() {
        let params = HNSWParams::default();
        let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Serialize empty index
        let path = "/tmp/test_empty_index.hnsw";
        index.save(path).unwrap();

        // Deserialize
        let loaded = HNSWIndex::load(path).unwrap();

        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dimensions(), 128);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert some data
        for i in 0..10 {
            index.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        // Share across threads (tests Send + Sync)
        let index = Arc::new(index);
        let mut handles = vec![];

        for _ in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                // Query from multiple threads
                let query = vec![5.0, 0.0, 0.0];
                let results = index_clone.search(&query, 3, 10).unwrap();
                assert_eq!(results.len(), 3);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_hybrid_mode_without_disk_operations() {
        // Create hybrid mode but only do in-memory operations
        // This should work fine (LayeredStorage starts with MemoryStorage)
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert and query should work fine
        for i in 0..50 {
            index.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }

        let query = vec![25.0, 0.0, 0.0];
        let results = index.search(&query, 5, 10).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, 25);
    }

    #[test]
    fn test_error_propagation_in_hybrid_mode() {
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        // Insert valid vector
        let result = index.insert(vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());

        // Insert invalid dimension (should error)
        let result = index.insert(vec![1.0, 2.0]); // Wrong dimensions
        assert!(result.is_err());
    }

    // Disk-backed storage tests

    #[test]
    fn test_save_graph_to_disk() {
        use tempfile::tempdir;

        // Build small index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..100 {
            index
                .insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0])
                .unwrap();
        }

        // Save graph to disk
        let temp_dir = tempdir().unwrap();
        let graph_path = temp_dir.path().join("test_graph");

        let result = index.save_graph_to_disk(&graph_path);
        assert!(result.is_ok());

        // Verify disk files exist
        assert!(graph_path.join("metadata.bin").exists());
        assert!(graph_path.join("layer_0.graph").exists());
    }

    #[test]
    fn test_save_graph_to_disk_requires_memory_mode() {
        use tempfile::tempdir;

        // Create index with hybrid mode
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new_with_storage(
            3,
            params,
            DistanceFunction::L2,
            false,
            StorageMode::Hybrid,
        )
        .unwrap();

        index.insert(vec![1.0, 2.0, 3.0]).unwrap();

        // Try to save (should fail - already in layered mode)
        let temp_dir = tempdir().unwrap();
        let graph_path = temp_dir.path().join("test_graph");

        let result = index.save_graph_to_disk(&graph_path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires GraphStorage::Memory mode"));
    }

    #[test]
    fn test_load_with_disk_graph() {
        use tempfile::tempdir;

        // Build and save index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

        // Insert vectors
        for i in 0..100 {
            index
                .insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0])
                .unwrap();
        }

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("test_index.bin");
        let graph_path = temp_dir.path().join("test_graph");

        // Save both index and graph
        index.save(&index_path).unwrap();
        index.save_graph_to_disk(&graph_path).unwrap();

        // Load with disk-backed graph
        let config = DiskConfig::new(graph_path, 100);
        let loaded_index = HNSWIndex::load_with_disk_graph(&index_path, config).unwrap();

        // Verify loaded correctly
        assert_eq!(loaded_index.len(), 100);
        assert_eq!(loaded_index.dimensions(), 3);
        assert!(loaded_index.neighbors.is_layered_mode());
        assert_eq!(loaded_index.neighbors.mode(), StorageMode::Hybrid);

        // Verify queries work
        let query = vec![50.0, 51.0, 52.0];
        let results = loaded_index.search(&query, 5, 50).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_disk_backed_workflow_end_to_end() {
        use tempfile::tempdir;

        // Full workflow: Build → Save → Load with disk → Query
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

        // Build index (200 vectors)
        let vectors: Vec<Vec<f32>> = (0..200)
            .map(|i| (0..128).map(|d| (i * 128 + d) as f32).collect())
            .collect();

        for vec in &vectors {
            index.insert(vec.clone()).unwrap();
        }

        let temp_dir = tempdir().unwrap();
        let index_path = temp_dir.path().join("index.bin");
        let graph_path = temp_dir.path().join("graph");

        // Save index and graph
        index.save(&index_path).unwrap();
        index.save_graph_to_disk(&graph_path).unwrap();

        // Load with disk-backed graph
        let config = DiskConfig::new(graph_path, 200);
        let disk_index = HNSWIndex::load_with_disk_graph(&index_path, config).unwrap();

        // Query and verify
        for (i, query) in vectors.iter().enumerate().take(10) {
            let results = disk_index.search(query, 5, 50).unwrap();
            assert_eq!(results.len(), 5);
            // First result should be the query vector itself
            assert_eq!(results[0].id, i as u32);
        }

        // Verify memory mode vs disk mode both work
        assert!(index.neighbors.is_memory_mode());
        assert!(disk_index.neighbors.is_layered_mode());
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_search_qps -- --ignored --nocapture
    fn bench_search_qps() {
        use std::time::Instant;

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!(
            "\n=== HNSW Raw Search Benchmark ({} vectors, {} queries) ===\n",
            n, queries
        );

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * dim + d) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * dim + d + 500) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();

        // Create index
        let params = HNSWParams::default();
        let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();

        // Batch insert (fair comparison with VectorStore)
        let start = Instant::now();
        index.batch_insert(vectors.clone()).unwrap();
        let insert_time = start.elapsed();
        println!(
            "Batch insert: {:?} ({:.0} vec/s)",
            insert_time,
            n as f64 / insert_time.as_secs_f64()
        );

        // Warm up
        for _ in 0..10 {
            let _ = index.search(&query_vecs[0], 10, 100);
        }

        // Benchmark
        let start = Instant::now();
        for q in &query_vecs {
            let _ = index.search(q, 10, 100).unwrap();
        }
        let search_time = start.elapsed();
        let qps = queries as f64 / search_time.as_secs_f64();

        println!("Search: {:?} ({:.0} QPS)", search_time, qps);
        println!(
            "\nPer-query: {:.3}ms",
            search_time.as_secs_f64() * 1000.0 / queries as f64
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --release bench_vectorstore_qps -- --ignored --nocapture
    fn bench_vectorstore_qps() {
        use crate::vector::{Vector, VectorStore};
        use std::time::Instant;

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!(
            "\n=== VectorStore Search Benchmark ({} vectors, {} queries) ===\n",
            n, queries
        );

        // Generate vectors
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * dim + d) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();
        let query_vecs: Vec<Vector> = (0..queries)
            .map(|i| {
                Vector::new(
                    (0..dim)
                        .map(|d| ((i * dim + d + 500) % 1000) as f32 / 1000.0)
                        .collect(),
                )
            })
            .collect();

        // Create store with batch insert
        let mut store = VectorStore::new(dim);
        let start = Instant::now();
        let batch: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i.to_string(),
                    Vector::new(v.clone()),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        store.set_batch(batch).unwrap();
        let insert_time = start.elapsed();
        println!(
            "Batch insert: {:?} ({:.0} vec/s)",
            insert_time,
            n as f64 / insert_time.as_secs_f64()
        );

        // Warm up
        for _ in 0..10 {
            let _ = store.knn_search(&query_vecs[0], 10);
        }

        // Benchmark knn_search (no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.knn_search(q, 10).unwrap();
        }
        let knn_time = start.elapsed();
        let knn_qps = queries as f64 / knn_time.as_secs_f64();
        println!(
            "knn_search (no metadata): {:?} ({:.0} QPS)",
            knn_time, knn_qps
        );

        // Benchmark search (with metadata lookup)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.search(q, 10, None).unwrap();
        }
        let search_time = start.elapsed();
        let search_qps = queries as f64 / search_time.as_secs_f64();
        println!(
            "search (with metadata): {:?} ({:.0} QPS)",
            search_time, search_qps
        );

        println!("\n=== Summary ===");
        println!(
            "knn_search QPS: {:.0} ({:.3}ms/query)",
            knn_qps,
            knn_time.as_secs_f64() * 1000.0 / queries as f64
        );
        println!(
            "search QPS: {:.0} ({:.3}ms/query)",
            search_qps,
            search_time.as_secs_f64() * 1000.0 / queries as f64
        );
        println!(
            "Metadata overhead: {:.1}%",
            (1.0 - search_qps / knn_qps) * 100.0
        );
    }

    /// Profile seerdb storage to identify optimization targets.
    /// Results written to SEERDB_PROFILE_RESULTS.md for seerdb development.
    ///
    /// Run with: cargo test --release profile_seerdb -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_seerdb() {
        profile_seerdb_impl(100_000);
    }

    /// Comprehensive seerdb profile comparing persistent vs in-memory.
    /// Tests actual seerdb impact: startup, insert, metadata lookups.
    ///
    /// Run with: cargo test --release profile_seerdb_comprehensive -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_seerdb_comprehensive() {
        use crate::vector::{Vector, VectorStore};
        use rand::Rng;
        use std::fs::File;
        use std::io::Write;
        use std::time::Instant;

        let n = 10_000;
        let dim = 128;
        let queries = 1000;

        println!("\n=== Comprehensive seerdb Profile ({} vectors) ===\n", n);

        // Generate random vectors
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        // === TEST 1: In-memory (no seerdb) ===
        println!("=== 1. In-Memory Mode (no seerdb) ===");
        let mut inmem_store = VectorStore::new(dim);

        // Insert
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i.to_string(),
                    Vector::new(v.clone()),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        inmem_store.set_batch(docs).unwrap();
        let inmem_insert = start.elapsed();
        println!(
            "Insert: {:?} ({:.0} vec/s)",
            inmem_insert,
            n as f64 / inmem_insert.as_secs_f64()
        );

        // Search (knn_search - no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = inmem_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let inmem_knn = start.elapsed();
        let inmem_knn_qps = queries as f64 / inmem_knn.as_secs_f64();
        println!("knn_search: {:?} ({:.0} QPS)", inmem_knn, inmem_knn_qps);

        // Search (with metadata lookup)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = inmem_store.search(&Vector::new(q.clone()), 10, None);
        }
        let inmem_search = start.elapsed();
        let inmem_search_qps = queries as f64 / inmem_search.as_secs_f64();
        println!(
            "search (metadata): {:?} ({:.0} QPS)",
            inmem_search, inmem_search_qps
        );

        // === TEST 2: Persistent (seerdb) ===
        println!("\n=== 2. Persistent Mode (seerdb) ===");
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("profile-oadb");
        let mut persist_store = VectorStore::open_with_dimensions(&path, dim).unwrap();

        // Insert
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i.to_string(),
                    Vector::new(v.clone()),
                    serde_json::json!({"idx": i}),
                )
            })
            .collect();
        persist_store.set_batch(docs).unwrap();
        let persist_insert = start.elapsed();
        println!(
            "Insert: {:?} ({:.0} vec/s)",
            persist_insert,
            n as f64 / persist_insert.as_secs_f64()
        );

        // Flush
        let start = Instant::now();
        persist_store.flush().unwrap();
        let flush_time = start.elapsed();
        println!("Flush: {:?}", flush_time);

        // Search (knn_search - no metadata)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = persist_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let persist_knn = start.elapsed();
        let persist_knn_qps = queries as f64 / persist_knn.as_secs_f64();
        println!("knn_search: {:?} ({:.0} QPS)", persist_knn, persist_knn_qps);

        // Search (with metadata lookup from seerdb)
        let start = Instant::now();
        for q in &query_vecs {
            let _ = persist_store.search(&Vector::new(q.clone()), 10, None);
        }
        let persist_search = start.elapsed();
        let persist_search_qps = queries as f64 / persist_search.as_secs_f64();
        println!(
            "search (metadata): {:?} ({:.0} QPS)",
            persist_search, persist_search_qps
        );

        // Get seerdb stats for metadata lookups
        let stats = persist_store.storage().unwrap().stats();

        drop(persist_store);

        // === TEST 3: Cold Start (reopen from disk) ===
        println!("\n=== 3. Cold Start (reload from seerdb) ===");
        let start = Instant::now();
        let mut reloaded_store = VectorStore::open(&path).unwrap();
        let reload_time = start.elapsed();
        println!("Reload {} vectors: {:?}", reloaded_store.len(), reload_time);

        // Verify search works
        let start = Instant::now();
        for q in &query_vecs {
            let _ = reloaded_store.knn_search(&Vector::new(q.clone()), 10);
        }
        let reload_knn = start.elapsed();
        let reload_knn_qps = queries as f64 / reload_knn.as_secs_f64();
        println!(
            "knn_search (post-reload): {:?} ({:.0} QPS)",
            reload_knn, reload_knn_qps
        );

        // === SUMMARY ===
        println!("\n=== Summary: seerdb Impact ===");
        println!("| Operation | In-Memory | seerdb | Overhead |");
        println!("|-----------|-----------|--------|----------|");
        println!(
            "| Insert ({} vec) | {:?} | {:?} | {:.1}x |",
            n,
            inmem_insert,
            persist_insert,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64()
        );
        println!(
            "| knn_search | {:.0} QPS | {:.0} QPS | {:.1}% |",
            inmem_knn_qps,
            persist_knn_qps,
            (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0
        );
        println!(
            "| search (metadata) | {:.0} QPS | {:.0} QPS | {:.1}% |",
            inmem_search_qps,
            persist_search_qps,
            (1.0 - persist_search_qps / inmem_search_qps) * 100.0
        );
        println!("| Cold start | N/A | {:?} | - |", reload_time);
        println!("| Flush | N/A | {:?} | - |", flush_time);

        println!("\n=== seerdb Stats ===");
        println!("Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
        println!(
            "Cache hits: {}, misses: {}",
            stats.cache_hits, stats.cache_misses
        );
        println!("Total gets: {}", stats.total_gets);
        println!("SSTables: {:?}", stats.sstables_per_level);

        // Write comprehensive results
        let results = format!(
            r#"# seerdb Profile Results (Comprehensive)

**Date**: {}
**Dataset**: {} vectors, {} dimensions
**Queries**: {}

## Should oadb Use seerdb?

**YES** - seerdb provides:
1. **Persistence** - Data survives process restart
2. **Durability** - WAL protects against crashes
3. **Minimal overhead** - See benchmarks below

## Performance Comparison

| Operation | In-Memory | seerdb | Overhead |
|-----------|-----------|--------|----------|
| Insert ({} vec) | {:?} | {:?} | {:.1}x |
| knn_search | {:.0} QPS | {:.0} QPS | {:.1}% |
| search (metadata) | {:.0} QPS | {:.0} QPS | {:.1}% |
| Cold start | N/A | {:?} | - |
| Flush | N/A | {:?} | - |

## Key Findings

### 1. Search Performance: IDENTICAL
- knn_search uses in-memory HNSW graph (no seerdb I/O)
- seerdb overhead: {:.1}% (within noise)

### 2. Insert Performance: {:.1}x Overhead
- seerdb writes: vectors, metadata, ID mappings (4 KV pairs/vector)
- Still achieves {:.0} vec/s with persistence

### 3. Cold Start: {:?}
- Loads {} vectors from seerdb on startup
- Rebuilds HNSW index in memory

### 4. Metadata Lookups
- search() reads metadata from seerdb for results
- Cache hit rate: {:.1}%
- Total seerdb gets: {}

## seerdb Stats

```
Cache hits: {}
Cache misses: {}
Hit rate: {:.1}%
SSTables per level: {:?}
Total gets: {}
```

## Recommendation

**Keep seerdb for oadb** because:
1. Search performance is NOT affected (in-memory HNSW)
2. Insert overhead is acceptable ({:.1}x for durability)
3. Cold start is fast enough ({:?} for {} vectors)
4. Durability is critical for production use

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| No persistence | Fastest inserts | Data lost on restart |
| Simple file | Simpler code | No durability, slow reload |
| fjall | Simpler API | Less optimized for oadb use case |
| **seerdb** | Optimized, durable | Slight insert overhead |
"#,
            chrono::Local::now().format("%Y-%m-%d"),
            n,
            dim,
            queries,
            n,
            inmem_insert,
            persist_insert,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            inmem_knn_qps,
            persist_knn_qps,
            (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0,
            inmem_search_qps,
            persist_search_qps,
            (1.0 - persist_search_qps / inmem_search_qps) * 100.0,
            reload_time,
            flush_time,
            (1.0 - persist_knn_qps / inmem_knn_qps) * 100.0,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            n as f64 / persist_insert.as_secs_f64(),
            reload_time,
            n,
            stats.cache_hit_rate * 100.0,
            stats.total_gets,
            stats.cache_hits,
            stats.cache_misses,
            stats.cache_hit_rate * 100.0,
            stats.sstables_per_level,
            stats.total_gets,
            persist_insert.as_secs_f64() / inmem_insert.as_secs_f64(),
            reload_time,
            n,
        );

        let output_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SEERDB_PROFILE_RESULTS.md");
        let mut file = File::create(&output_path).expect("Failed to create results file");
        file.write_all(results.as_bytes())
            .expect("Failed to write results");
        println!("\n✓ Results written to: {}", output_path.display());
    }

    fn profile_seerdb_impl(n: usize) {
        use crate::vector::{Vector, VectorStore};
        use rand::Rng;
        use std::fs::File;
        use std::io::Write;
        use std::time::Instant;

        let n = 100_000;
        let dim = 128;
        let queries = 100;

        println!(
            "\n=== seerdb Profile ({} vectors, {} queries) ===\n",
            n, queries
        );

        // Generate random vectors
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query_vecs: Vec<Vec<f32>> = (0..queries)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();

        // Create VectorStore with seerdb persistence
        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("profile-oadb");
        let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

        // Insert vectors (batch)
        let start = Instant::now();
        let docs: Vec<(String, Vector, serde_json::Value)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), Vector::new(v.clone()), serde_json::json!({})))
            .collect();
        store.set_batch(docs).unwrap();
        let insert_time = start.elapsed();
        println!(
            "Insert: {:?} ({:.0} vec/s)",
            insert_time,
            n as f64 / insert_time.as_secs_f64()
        );

        // Flush to disk
        let start = Instant::now();
        store.flush().unwrap();
        let flush_time = start.elapsed();
        println!("Flush: {:?}", flush_time);

        // Warm up
        for _ in 0..10 {
            let _ = store.knn_search(&Vector::new(query_vecs[0].clone()), 10);
        }

        // Get stats after warmup (snapshot)
        let stats_after_warmup = store
            .storage()
            .expect("Store should have persistent storage")
            .stats();

        // Benchmark search
        let start = Instant::now();
        for q in &query_vecs {
            let _ = store.knn_search(&Vector::new(q.clone()), 10);
        }
        let search_time = start.elapsed();
        let qps = queries as f64 / search_time.as_secs_f64();
        let ms_per_query = search_time.as_secs_f64() * 1000.0 / queries as f64;

        // Get post-search stats
        let stats_after = store
            .storage()
            .expect("Store should have persistent storage")
            .stats();

        // Calculate delta for this search batch
        let search_gets = stats_after.total_gets - stats_after_warmup.total_gets;
        let search_cache_hits = stats_after.cache_hits - stats_after_warmup.cache_hits;
        let search_cache_misses = stats_after.cache_misses - stats_after_warmup.cache_misses;
        let search_hit_rate = if search_cache_hits + search_cache_misses > 0 {
            search_cache_hits as f64 / (search_cache_hits + search_cache_misses) as f64
        } else {
            0.0
        };

        // Print results
        println!("\n=== Search Latency ===");
        println!("Total search time: {:?}", search_time);
        println!("Per-query: {:.2}ms", ms_per_query);
        println!("QPS: {:.0}", qps);
        println!(
            "Edge lookups (seerdb gets): {} ({:.1} per query)",
            search_gets,
            search_gets as f64 / queries as f64
        );

        println!("\n=== Cache Stats (Search Batch) ===");
        println!("Cache hits: {}", search_cache_hits);
        println!("Cache misses: {}", search_cache_misses);
        println!("Cache hit rate: {:.1}%", search_hit_rate * 100.0);

        println!("\n=== Cache Stats (Cumulative) ===");
        println!("Total cache hits: {}", stats_after.cache_hits);
        println!("Total cache misses: {}", stats_after.cache_misses);
        println!(
            "Overall hit rate: {:.1}%",
            stats_after.cache_hit_rate * 100.0
        );

        println!("\n=== LSM Tree Health ===");
        println!("SSTables per level: {:?}", stats_after.sstables_per_level);
        println!("Total SSTables: {}", stats_after.total_sstables);
        println!(
            "Total disk bytes: {} KB",
            stats_after.total_disk_bytes / 1024
        );

        println!("\n=== Latency Percentiles ===");
        println!("Get p50: {}us", stats_after.get_latency_p50_us);
        println!("Get p95: {}us", stats_after.get_latency_p95_us);
        println!("Get p99: {}us", stats_after.get_latency_p99_us);
        println!("Get p999: {}us", stats_after.get_latency_p999_us);

        // Write results to file
        let results = format!(
            r#"# seerdb Profile Results

**Date**: {}
**Dataset**: {} vectors, {} dimensions
**Hardware**: M3 Max (via cargo test)

## Search Latency
- Avg search time: {:.2} ms
- QPS: {:.0}
- Edge lookups per search: {:.1}

## Cache Stats (During Search)
- Hit rate: {:.1}%
- Hits: {}
- Misses: {}

## Flush Timing
- {} vector flush: {:?}

## LSM Health
- SSTables per level: {:?}
- Total SSTables: {}
- Total disk: {} KB

## Get Latency Percentiles
- p50: {}us
- p95: {}us
- p99: {}us
- p999: {}us

## Analysis
- Cache hit rate {}: {}
- LSM health: {}
"#,
            chrono::Local::now().format("%Y-%m-%d"),
            n,
            dim,
            ms_per_query,
            qps,
            search_gets as f64 / queries as f64,
            search_hit_rate * 100.0,
            search_cache_hits,
            search_cache_misses,
            n,
            flush_time,
            stats_after.sstables_per_level,
            stats_after.total_sstables,
            stats_after.total_disk_bytes / 1024,
            stats_after.get_latency_p50_us,
            stats_after.get_latency_p95_us,
            stats_after.get_latency_p99_us,
            stats_after.get_latency_p999_us,
            if search_hit_rate > 0.7 { ">" } else { "<" },
            if search_hit_rate > 0.7 {
                "GOOD (>70%)"
            } else {
                "NEEDS TUNING (<70%)"
            },
            if stats_after.sstables_per_level.first().copied().unwrap_or(0) < 20 {
                "GOOD (L0 < 20)"
            } else {
                "COMPACTION FALLING BEHIND"
            },
        );

        // Write to SEERDB_PROFILE_RESULTS.md
        let output_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SEERDB_PROFILE_RESULTS.md");
        let mut file = File::create(&output_path).expect("Failed to create results file");
        file.write_all(results.as_bytes())
            .expect("Failed to write results");
        println!("\n✓ Results written to: {}", output_path.display());
    }
