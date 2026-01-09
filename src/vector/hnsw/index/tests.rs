#![allow(clippy::float_cmp, clippy::field_reassign_with_default)]

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
    let id = index.insert(&vec).unwrap();

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

    let id1 = index.insert(&vec1).unwrap();
    let id2 = index.insert(&vec2).unwrap();
    let id3 = index.insert(&vec3).unwrap();

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
    assert!(index.insert(&wrong_dim).is_err());
}

#[test]
fn test_hnsw_index_insert_nan_rejected() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // NaN should be rejected
    let vec_nan = vec![1.0, f32::NAN, 3.0];
    let result = index.insert(&vec_nan);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HNSWError::InvalidVector));
}

#[test]
fn test_hnsw_index_insert_infinity_rejected() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // Positive infinity should be rejected
    let vec_inf = vec![1.0, f32::INFINITY, 3.0];
    let result = index.insert(&vec_inf);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HNSWError::InvalidVector));

    // Negative infinity should be rejected
    let vec_neg_inf = vec![1.0, f32::NEG_INFINITY, 3.0];
    let result = index.insert(&vec_neg_inf);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HNSWError::InvalidVector));
}

#[test]
fn test_hnsw_index_search_nan_rejected() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // Insert a valid vector first
    index.insert(&[1.0, 2.0, 3.0]).unwrap();

    // Search with NaN should be rejected
    let query_nan = vec![1.0, f32::NAN, 3.0];
    let result = index.search(&query_nan, 5, 100);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HNSWError::InvalidVector));
}

#[test]
fn test_hnsw_index_search_infinity_rejected() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // Insert a valid vector first
    index.insert(&[1.0, 2.0, 3.0]).unwrap();

    // Search with infinity should be rejected
    let query_inf = vec![1.0, f32::INFINITY, 3.0];
    let result = index.search(&query_inf, 5, 100);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HNSWError::InvalidVector));
}

#[test]
fn test_hnsw_index_search_invalid_params_ef_less_than_k() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // Insert vectors
    index.insert(&[1.0, 2.0, 3.0]).unwrap();
    index.insert(&[4.0, 5.0, 6.0]).unwrap();

    // Search with ef < k should fail
    let query = vec![1.0, 2.0, 3.0];
    let result = index.search(&query, 10, 5); // k=10, ef=5
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        HNSWError::InvalidSearchParams { k: 10, ef: 5 }
    ));
}

#[test]
fn test_hnsw_index_search_invalid_params_ef_zero() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    index.insert(&[1.0, 2.0, 3.0]).unwrap();

    // Search with ef=0 should fail
    let query = vec![1.0, 2.0, 3.0];
    let result = index.search(&query, 1, 0); // k=1, ef=0
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        HNSWError::InvalidSearchParams { k: 1, ef: 0 }
    ));
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
    index.insert(&vec).unwrap();

    let results = index.search(&vec, 5, 100).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
    assert!(results[0].distance < 0.01); // Should be ~0 (same vector)
}

#[test]
fn test_random_level_distribution() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    let mut level_counts = [0; 8];

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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
    }

    // Count how many nodes have their TOP level at each height
    // Note: All nodes exist at level 0, but node.level is their TOP level
    let mut top_level_counts = [0; 8];
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
        index.insert(&vec).unwrap();
    }

    // Check that no node has more than M*2 neighbors at level 0
    for node in &index.nodes {
        let neighbor_count = index.neighbors.get_neighbors(node.id, 0).len();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
    }

    let stats = index.stats();

    assert_eq!(stats.num_vectors, 50);
    assert_eq!(stats.dimensions, 3);
    assert!(stats.entry_point.is_some());
    assert!(!stats.level_distribution.is_empty());
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
        index.insert(&vec).unwrap();
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
    let index_dot = HNSWIndex::new(3, params, DistanceFunction::NegativeDotProduct, false).unwrap();
    let stats = index_dot.stats();
    assert!(matches!(
        stats.distance_function,
        DistanceFunction::NegativeDotProduct
    ));
}

// ========================================
// Edge Case Tests
// ========================================

#[test]
fn test_empty_index_serialization() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test_empty_index.hnsw");

    let params = HNSWParams::default();
    let index = HNSWIndex::new(128, params, DistanceFunction::L2, false).unwrap();

    // Serialize empty index
    index.save(&path).unwrap();

    // Deserialize
    let loaded = HNSWIndex::load(&path).unwrap();

    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.dimensions(), 128);
}

#[test]
fn test_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(3, params, DistanceFunction::L2, false).unwrap();

    // Insert some data
    for i in 0..10 {
        index.insert(&[i as f32, 0.0, 0.0]).unwrap();
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
#[ignore = "benchmark - run with: cargo test --release bench_search_qps -- --ignored --nocapture"]
fn bench_search_qps() {
    use std::time::Instant;

    let n = 10_000;
    let dim = 128;
    let queries = 1000;

    println!("\n=== HNSW Raw Search Benchmark ({n} vectors, {queries} queries) ===\n");

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

    println!("Search: {search_time:?} ({qps:.0} QPS)");
    println!(
        "\nPer-query: {:.3}ms",
        search_time.as_secs_f64() * 1000.0 / queries as f64
    );
}

#[test]
#[ignore = "benchmark - run with: cargo test --release bench_vectorstore_qps -- --ignored --nocapture"]
fn bench_vectorstore_qps() {
    use crate::vector::{Vector, VectorStore};
    use std::time::Instant;

    let n = 10_000;
    let dim = 128;
    let queries = 1000;

    println!("\n=== VectorStore Search Benchmark ({n} vectors, {queries} queries) ===\n");

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
    println!("knn_search (no metadata): {knn_time:?} ({knn_qps:.0} QPS)");

    // Benchmark search (with metadata lookup)
    let start = Instant::now();
    for q in &query_vecs {
        let _ = store.search(q, 10, None).unwrap();
    }
    let search_time = start.elapsed();
    let search_qps = queries as f64 / search_time.as_secs_f64();
    println!("search (with metadata): {search_time:?} ({search_qps:.0} QPS)");

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

/// Profile persistent storage to identify optimization targets.
///
/// Run with: cargo test --release `profile_persistence` -- --ignored --nocapture
#[test]
#[ignore = "profiling - run manually with --ignored"]
fn profile_persistence() {
    profile_persistence_impl(100_000);
}

/// Comprehensive persistence profile comparing persistent vs in-memory.
/// Tests actual disk I/O impact: startup, insert, metadata lookups.
///
/// Run with: cargo test --release `profile_persistence_comprehensive` -- --ignored --nocapture
#[test]
#[ignore = "profiling - run manually with --ignored"]
fn profile_persistence_comprehensive() {
    use crate::vector::{Vector, VectorStore};
    use rand::Rng;
    use std::fs::File;
    use std::io::Write;
    use std::time::Instant;

    let n = 10_000;
    let dim = 128;
    let queries = 1000;

    println!("\n=== Comprehensive Persistence Profile ({n} vectors) ===\n");

    // Generate random vectors
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let query_vecs: Vec<Vec<f32>> = (0..queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // === TEST 1: In-memory (no persistence) ===
    println!("=== 1. In-Memory Mode (no persistence) ===");
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
    println!("knn_search: {inmem_knn:?} ({inmem_knn_qps:.0} QPS)");

    // Search (with metadata lookup)
    let start = Instant::now();
    for q in &query_vecs {
        let _ = inmem_store.search(&Vector::new(q.clone()), 10, None);
    }
    let inmem_search = start.elapsed();
    let inmem_search_qps = queries as f64 / inmem_search.as_secs_f64();
    println!("search (metadata): {inmem_search:?} ({inmem_search_qps:.0} QPS)");

    // === TEST 2: Persistent (disk) ===
    println!("\n=== 2. Persistent Mode (disk) ===");
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
    println!("Flush: {flush_time:?}");

    // Search (knn_search - no metadata)
    let start = Instant::now();
    for q in &query_vecs {
        let _ = persist_store.knn_search(&Vector::new(q.clone()), 10);
    }
    let persist_knn = start.elapsed();
    let persist_knn_qps = queries as f64 / persist_knn.as_secs_f64();
    println!("knn_search: {persist_knn:?} ({persist_knn_qps:.0} QPS)");

    // Search (with metadata lookup from disk)
    let start = Instant::now();
    for q in &query_vecs {
        let _ = persist_store.search(&Vector::new(q.clone()), 10, None);
    }
    let persist_search = start.elapsed();
    let persist_search_qps = queries as f64 / persist_search.as_secs_f64();
    println!("search (metadata): {persist_search:?} ({persist_search_qps:.0} QPS)");

    drop(persist_store);

    // === TEST 3: Cold Start (reopen from disk) ===
    println!("\n=== 3. Cold Start (reload from disk) ===");
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
    println!("knn_search (post-reload): {reload_knn:?} ({reload_knn_qps:.0} QPS)");

    // === SUMMARY ===
    println!("\n=== Summary: Persistence Impact ===");
    println!("| Operation | In-Memory | Persistent | Overhead |");
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
    println!("| Cold start | N/A | {reload_time:?} | - |");
    println!("| Flush | N/A | {flush_time:?} | - |");

    // Write results
    let results = format!(
        r"# Persistence Profile Results

**Date**: {}
**Dataset**: {} vectors, {} dimensions
**Queries**: {}

## Performance Comparison

| Operation | In-Memory | Persistent | Overhead |
|-----------|-----------|------------|----------|
| Insert ({} vec) | {:?} | {:?} | {:.1}x |
| knn_search | {:.0} QPS | {:.0} QPS | {:.1}% |
| search (metadata) | {:.0} QPS | {:.0} QPS | {:.1}% |
| Cold start | N/A | {:?} | - |
| Flush | N/A | {:?} | - |
",
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
    );

    let output_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("PERSISTENCE_PROFILE_RESULTS.md");
    let mut file = File::create(&output_path).expect("Failed to create results file");
    file.write_all(results.as_bytes())
        .expect("Failed to write results");
    println!("\n✓ Results written to: {}", output_path.display());
}

fn profile_persistence_impl(_n: usize) {
    use crate::vector::{Vector, VectorStore};
    use rand::Rng;
    use std::time::Instant;

    let n = 10_000;
    let dim = 128;
    let queries = 100;

    println!("\n=== Persistence Profile ({n} vectors, {queries} queries) ===\n");

    // Generate random vectors
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let query_vecs: Vec<Vec<f32>> = (0..queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Create VectorStore with persistence
    let tmpdir = tempfile::tempdir().unwrap();
    let path = tmpdir.path().join("profile-omen");
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
    println!("Flush: {flush_time:?}");

    // Warm up
    for _ in 0..10 {
        let _ = store.knn_search(&Vector::new(query_vecs[0].clone()), 10);
    }

    // Benchmark search
    let start = Instant::now();
    for q in &query_vecs {
        let _ = store.knn_search(&Vector::new(q.clone()), 10);
    }
    let search_time = start.elapsed();
    let qps = queries as f64 / search_time.as_secs_f64();
    let ms_per_query = search_time.as_secs_f64() * 1000.0 / queries as f64;

    println!("\n=== Search Latency ===");
    println!("Total search time: {search_time:?}");
    println!("Per-query: {ms_per_query:.2}ms");
    println!("QPS: {qps:.0}");
}

#[test]
fn test_asymmetric_hnsw_search() {
    use crate::compression::RaBitQParams;

    let params = HNSWParams::default();
    let rabitq = RaBitQParams::bits4();
    let mut index =
        HNSWIndex::new_with_asymmetric(32, params, DistanceFunction::L2, rabitq).unwrap();

    assert!(index.is_asymmetric());
    assert!(index.is_empty());

    // Insert some vectors with unique patterns (use i directly, not i%32)
    let num_vectors = 32; // Keep under dimension to ensure unique patterns
    for i in 0..num_vectors {
        let mut vec = vec![0.0f32; 32];
        vec[i] = 1.0;
        vec[(i + 1) % 32] = 0.5;
        index.insert(&vec).unwrap();
    }

    assert_eq!(index.len(), num_vectors);

    // Search for nearest neighbor
    let mut query = vec![0.0f32; 32];
    query[0] = 1.0;
    query[1] = 0.5;

    let results = index.search(&query, 10, 50).unwrap();

    // Should find the matching vector (id 0) as closest
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 0, "Expected vector 0 to be closest");

    // Distance to self should be very small (just quantization error)
    assert!(
        results[0].distance < 0.5,
        "Distance to self should be small: {}",
        results[0].distance
    );
}

#[test]
fn test_asymmetric_only_supports_l2() {
    use crate::compression::RaBitQParams;

    let params = HNSWParams::default();
    let rabitq = RaBitQParams::bits4();

    // L2 should work
    let result = HNSWIndex::new_with_asymmetric(32, params, DistanceFunction::L2, rabitq.clone());
    assert!(result.is_ok());

    // Cosine should fail
    let result =
        HNSWIndex::new_with_asymmetric(32, params, DistanceFunction::Cosine, rabitq.clone());
    assert!(result.is_err());

    // NegativeDotProduct should fail
    let result =
        HNSWIndex::new_with_asymmetric(32, params, DistanceFunction::NegativeDotProduct, rabitq);
    assert!(result.is_err());
}

#[test]
fn test_binary_quantization_search() {
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new_with_binary(64, params, DistanceFunction::L2).unwrap();

    assert!(index.is_asymmetric());
    assert!(index.is_empty());

    // Train thresholds from sample vectors
    let samples: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            let mut v = vec![0.0f32; 64];
            // Create vectors with clear patterns above/below zero
            for j in 0..64 {
                v[j] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
            v
        })
        .collect();
    index.train_quantizer(&samples).unwrap();

    // Insert vectors with distinct binary patterns
    let num_vectors = 100;
    for i in 0..num_vectors {
        let mut vec = vec![0.0f32; 64];
        // Create vectors where bits are set based on i
        for j in 0..64 {
            vec[j] = if (i >> (j % 7)) & 1 == 1 { 1.0 } else { -1.0 };
        }
        index.insert(&vec).unwrap();
    }

    assert_eq!(index.len(), num_vectors);

    // Search for vector similar to vector 0
    let mut query = vec![0.0f32; 64];
    for j in 0..64 {
        query[j] = if j % 7 == 0 { 1.0 } else { -1.0 };
    }

    let results = index.search(&query, 10, 50).unwrap();

    // Should find results
    assert!(!results.is_empty(), "Binary search should return results");
    assert!(results.len() <= 10, "Should return at most k results");

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(
            results[i - 1].distance <= results[i].distance,
            "Results should be sorted by distance"
        );
    }
}

#[test]
fn test_binary_quantization_recall() {
    use rand::prelude::*;

    let dim = 128;
    let n_vectors = 500;
    let n_queries = 20;
    let k = 10;
    let ef = 100;

    // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Create binary index
    let params = HNSWParams::default();
    let mut binary_index = HNSWIndex::new_with_binary(dim, params, DistanceFunction::L2).unwrap();

    // Create full precision index for ground truth
    let mut fp_index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();

    // Generate random vectors
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Train binary quantization
    let sample: Vec<Vec<f32>> = vectors.iter().take(100).cloned().collect();
    binary_index.train_quantizer(&sample).unwrap();

    // Insert into both indexes
    for vec in &vectors {
        binary_index.insert(vec).unwrap();
        fp_index.insert(vec).unwrap();
    }

    // Generate queries and measure recall
    let mut total_recall = 0.0;
    for _ in 0..n_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Get ground truth from full precision index
        let fp_results = fp_index.search(&query, k, ef).unwrap();
        let ground_truth: std::collections::HashSet<u32> =
            fp_results.iter().map(|r| r.id).collect();

        // Get binary quantization results
        let binary_results = binary_index.search(&query, k, ef).unwrap();
        let binary_ids: std::collections::HashSet<u32> =
            binary_results.iter().map(|r| r.id).collect();

        // Calculate recall
        let matches = ground_truth.intersection(&binary_ids).count();
        total_recall += matches as f64 / k as f64;
    }

    let avg_recall = total_recall / n_queries as f64;

    // Binary quantization should achieve at least 70% recall with rescore
    // (hamming is used for graph traversal, L2 for final ranking)
    assert!(
        avg_recall >= 0.70,
        "Binary quantization recall too low: {:.2}% (expected >= 70%)",
        avg_recall * 100.0
    );
}

/// Test SQ8 recall at the HNSW level (small scale - 1K vectors)
///
/// This test verifies that L2 decomposition gives the same results as direct L2 distance.
/// We compare search results with SQ8 quantization to expected results.
#[test]
fn test_sq8_recall_regression() {
    use rand::prelude::*;

    let dim = 128;
    let n_vectors = 1000;
    let n_queries = 50;
    let k = 10;
    let ef = 100;

    // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate vectors in typical SIFT-like range
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect();

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect();

    // Create SQ8 index and insert vectors
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new_with_sq8(dim, params, DistanceFunction::L2).unwrap();

    for vec in &vectors {
        index.insert(vec).unwrap();
    }

    // Compute ground truth using brute force L2
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i as u32, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(id, _)| *id).collect()
        })
        .collect();

    // Search with SQ8 and compute recall
    let mut total_recall = 0.0;
    for (i, query) in queries.iter().enumerate() {
        let results = index.search(query, k, ef).unwrap();
        let result_ids: std::collections::HashSet<u32> = results.iter().map(|r| r.id).collect();
        let gt_ids: std::collections::HashSet<u32> = ground_truth[i].iter().copied().collect();
        let intersection = result_ids.intersection(&gt_ids).count();
        total_recall += intersection as f32 / k as f32;
    }
    let mean_recall = total_recall / n_queries as f32;

    println!("SQ8 HNSW recall@{k} (1K vectors): {:.4}", mean_recall);

    // SQ8 should achieve at least 90% recall
    assert!(
        mean_recall >= 0.90,
        "SQ8 recall too low: {:.4} (expected >= 0.90)",
        mean_recall
    );
}

/// Test SQ8 recall at larger scale (10K vectors - matches CI validation)
/// Run with: cargo test test_sq8_recall_10k -- --ignored
#[test]
#[ignore = "slow test (~10 min), run explicitly"]
fn test_sq8_recall_10k() {
    use rand::prelude::*;

    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;
    let ef = 100;

    // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate vectors in typical SIFT-like range
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect();

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect();

    // Create SQ8 index and insert vectors
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new_with_sq8(dim, params, DistanceFunction::L2).unwrap();

    for vec in &vectors {
        index.insert(vec).unwrap();
    }

    // Compute ground truth using brute force L2
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist: f32 = query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i as u32, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(id, _)| *id).collect()
        })
        .collect();

    // Search with SQ8 and compute recall
    let mut total_recall = 0.0;
    for (i, query) in queries.iter().enumerate() {
        let results = index.search(query, k, ef).unwrap();
        let result_ids: std::collections::HashSet<u32> = results.iter().map(|r| r.id).collect();
        let gt_ids: std::collections::HashSet<u32> = ground_truth[i].iter().copied().collect();
        let intersection = result_ids.intersection(&gt_ids).count();
        total_recall += intersection as f32 / k as f32;
    }
    let mean_recall = total_recall / n_queries as f32;

    println!("SQ8 HNSW recall@{k} (10K vectors): {:.4}", mean_recall);

    // At 10K scale, allow lower recall (CI threshold is 0.88)
    assert!(
        mean_recall >= 0.85,
        "SQ8 recall too low: {:.4} (expected >= 0.85)",
        mean_recall
    );
}

/// Test that SQ8 L2 decomposition gives the same distances as direct asymmetric L2
#[test]
fn test_sq8_distance_consistency() {
    use rand::prelude::*;

    let dim = 128;
    let n_vectors = 500;

    // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Create SQ8 index and insert vectors
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new_with_sq8(dim, params, DistanceFunction::L2).unwrap();

    for vec in &vectors {
        index.insert(vec).unwrap();
    }

    // Generate query
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Search and get results with distances
    let results = index.search(&query, 10, 100).unwrap();

    // Verify that returned distances are reasonable
    for result in &results {
        // Distance should be positive
        assert!(result.distance >= 0.0, "Distance should be non-negative");
        // Distance should not be extremely large (sanity check)
        assert!(
            result.distance < 1000.0,
            "Distance too large: {}",
            result.distance
        );
    }

    // Results should be sorted by distance (closest first)
    for window in results.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "Results not sorted by distance"
        );
    }
}

/// Test that L2 decomposition and direct asymmetric L2 give same rankings
#[test]
fn test_sq8_distance_methods_match() {
    use crate::distance::norm_squared;
    use rand::prelude::*;

    let dim = 128;
    let n_vectors = 300; // Enough to trigger quantization (>256)

    let mut rng = StdRng::seed_from_u64(42);

    // Generate SIFT-like vectors (values in 0-255 range)
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect();

    // Create SQ8 index
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new_with_sq8(dim, params, DistanceFunction::L2).unwrap();

    for vec in &vectors {
        index.insert(vec).unwrap();
    }

    // Generate query
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect();
    let query_norm = norm_squared(&query);

    // Compare L2 decomposition vs direct asymmetric L2 for all vectors
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;

    for id in 0..n_vectors as u32 {
        // Method 1: L2 decomposition (what current search uses)
        let decomp_dist = index
            .vectors
            .distance_l2_decomposed(&query, query_norm, id)
            .unwrap();

        // Method 2: Direct asymmetric L2 (what old asymmetric path used)
        let direct_dist = index.vectors.distance_asymmetric_l2(&query, id).unwrap();

        let abs_diff = (decomp_dist - direct_dist).abs();
        let rel_diff = abs_diff / direct_dist.max(1e-10);

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);
    }

    println!("Max absolute difference: {max_abs_diff}");
    println!("Max relative difference: {max_rel_diff}");

    // The two methods should give nearly identical results
    assert!(
        max_rel_diff < 1e-4,
        "L2 decomposition differs from direct L2: max_rel_diff = {max_rel_diff}"
    );
}
