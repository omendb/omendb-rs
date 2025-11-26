/*
 * IP-DiskANN Integration Tests
 *
 * Week 2 Goal: Validate IP-DiskANN implementation at scale
 * - 10K vectors: Recall validation
 * - Performance comparison vs HNSW
 * - Persistence validation
 */

#[cfg(test)]
mod tests {
    use super::super::index::IPDiskANNIndex;
    use super::super::types::IPDiskANNConfig;
    use rand::Rng;

    /// Generate random vectors for testing
    fn generate_random_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| (0..dimension).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    /// Compute ground truth k-NN via brute force
    fn compute_ground_truth(
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(id, vec)| {
                let dist = l2_distance(query, vec);
                (id, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }

    /// L2 distance helper
    fn l2_distance(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Calculate recall@k
    fn calculate_recall(ground_truth: &[(usize, f32)], results: &[(usize, f32)]) -> f32 {
        let gt_ids: std::collections::HashSet<_> =
            ground_truth.iter().map(|(id, _)| id).collect();
        let result_ids: std::collections::HashSet<_> = results.iter().map(|(id, _)| id).collect();

        let intersection = gt_ids.intersection(&result_ids).count();
        intersection as f32 / ground_truth.len() as f32
    }

    #[test]
    fn test_100_vectors_recall() {
        println!("\n=== Testing IP-DiskANN with 100 vectors ===");

        let dimension = 128;
        let num_vectors = 100;
        let k = 10;

        // Generate vectors
        let vectors = generate_random_vectors(num_vectors, dimension);

        // Build index
        let mut index = IPDiskANNIndex::new(dimension, None);
        for vector in vectors.iter() {
            index.insert(vector.clone()).unwrap();
        }

        println!("Built index with {} vectors", index.len());

        // Test 10 random queries
        let num_queries = 10;
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query_idx = rand::thread_rng().gen_range(0..num_vectors);
            let query = &vectors[query_idx];

            // Ground truth (brute force)
            let ground_truth = compute_ground_truth(query, &vectors, k);

            // IP-DiskANN search
            let results = index.search(query, k).unwrap();
            let results_with_dist: Vec<(usize, f32)> = results
                .iter()
                .map(|n| (n.id as usize, n.distance))
                .collect();

            // Calculate recall
            let recall = calculate_recall(&ground_truth, &results_with_dist);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);

        // Should have high recall (>90%) for small dataset
        assert!(
            avg_recall > 0.9,
            "Recall too low: {:.2}%",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_1000_vectors_recall() {
        println!("\n=== Testing IP-DiskANN with 1,000 vectors ===");

        let dimension = 128;
        let num_vectors = 1000;
        let k = 10;

        // Generate vectors
        let vectors = generate_random_vectors(num_vectors, dimension);

        // Build index
        let mut index = IPDiskANNIndex::new(dimension, None);
        let start = std::time::Instant::now();

        for vector in vectors.iter() {
            index.insert(vector.clone()).unwrap();
        }

        let build_time = start.elapsed();
        println!(
            "Built index with {} vectors in {:.2}s ({:.0} vec/sec)",
            index.len(),
            build_time.as_secs_f64(),
            num_vectors as f64 / build_time.as_secs_f64()
        );

        // Test 20 random queries
        let num_queries = 20;
        let mut total_recall = 0.0;
        let mut total_query_time = std::time::Duration::ZERO;

        for _ in 0..num_queries {
            let query_idx = rand::thread_rng().gen_range(0..num_vectors);
            let query = &vectors[query_idx];

            // Ground truth
            let ground_truth = compute_ground_truth(query, &vectors, k);

            // IP-DiskANN search (timed)
            let search_start = std::time::Instant::now();
            let results = index.search(query, k).unwrap();
            total_query_time += search_start.elapsed();

            let results_with_dist: Vec<(usize, f32)> = results
                .iter()
                .map(|n| (n.id as usize, n.distance))
                .collect();

            let recall = calculate_recall(&ground_truth, &results_with_dist);
            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f32;
        let avg_query_time = total_query_time / num_queries;

        println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
        println!(
            "Average query time: {:.2}ms ({:.0} QPS)",
            avg_query_time.as_secs_f64() * 1000.0,
            1.0 / avg_query_time.as_secs_f64()
        );

        // Should have >85% recall for 1K vectors
        assert!(
            avg_recall > 0.85,
            "Recall too low: {:.2}%",
            avg_recall * 100.0
        );

        // Query time should be <10ms
        assert!(
            avg_query_time.as_millis() < 10,
            "Query too slow: {:.2}ms",
            avg_query_time.as_secs_f64() * 1000.0
        );
    }

    #[test]
    #[ignore] // Slow test, run with --ignored
    fn test_10k_vectors_recall() {
        println!("\n=== Testing IP-DiskANN with 10,000 vectors ===");

        let dimension = 128;
        let num_vectors = 10_000;
        let k = 10;

        println!("Generating {} vectors...", num_vectors);
        let vectors = generate_random_vectors(num_vectors, dimension);

        // Build index
        println!("Building IP-DiskANN index...");
        let mut index = IPDiskANNIndex::new(dimension, None);
        let start = std::time::Instant::now();

        for (i, vector) in vectors.iter().enumerate() {
            index.insert(vector.clone()).unwrap();
            if (i + 1) % 1000 == 0 {
                println!("  Inserted {}/{}", i + 1, num_vectors);
            }
        }

        let build_time = start.elapsed();
        println!(
            "Built index with {} vectors in {:.2}s ({:.0} vec/sec)",
            index.len(),
            build_time.as_secs_f64(),
            num_vectors as f64 / build_time.as_secs_f64()
        );

        // Test 50 random queries
        println!("Testing recall with 50 queries...");
        let num_queries = 50;
        let mut total_recall = 0.0;
        let mut total_query_time = std::time::Duration::ZERO;

        for i in 0..num_queries {
            let query_idx = rand::thread_rng().gen_range(0..num_vectors);
            let query = &vectors[query_idx];

            // Ground truth
            let ground_truth = compute_ground_truth(query, &vectors, k);

            // IP-DiskANN search
            let search_start = std::time::Instant::now();
            let results = index.search(query, k).unwrap();
            total_query_time += search_start.elapsed();

            let results_with_dist: Vec<(usize, f32)> = results
                .iter()
                .map(|n| (n.id as usize, n.distance))
                .collect();

            let recall = calculate_recall(&ground_truth, &results_with_dist);
            total_recall += recall;

            if (i + 1) % 10 == 0 {
                println!("  Completed {}/{} queries", i + 1, num_queries);
            }
        }

        let avg_recall = total_recall / num_queries as f32;
        let avg_query_time = total_query_time / num_queries;

        println!("\n=== Results ===");
        println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
        println!(
            "Average query time: {:.2}ms ({:.0} QPS)",
            avg_query_time.as_secs_f64() * 1000.0,
            1.0 / avg_query_time.as_secs_f64()
        );
        println!(
            "Build throughput: {:.0} vec/sec",
            num_vectors as f64 / build_time.as_secs_f64()
        );

        // IP-DiskANN target: >80% recall @ 10K scale
        assert!(
            avg_recall > 0.80,
            "Recall too low: {:.2}%",
            avg_recall * 100.0
        );

        // Query time should be <10ms @ 10K
        assert!(
            avg_query_time.as_millis() < 10,
            "Query too slow: {:.2}ms",
            avg_query_time.as_secs_f64() * 1000.0
        );
    }

    #[test]
    fn test_persistence_with_large_index() {
        println!("\n=== Testing persistence with 500 vectors ===");

        let dimension = 128;
        let num_vectors = 500;
        let k = 10;

        // Generate vectors
        let vectors = generate_random_vectors(num_vectors, dimension);

        // Build index
        let mut index = IPDiskANNIndex::new(dimension, None);
        for vector in vectors.iter() {
            index.insert(vector.clone()).unwrap();
        }

        println!("Built index with {} vectors", index.len());

        // Test query before save
        let query = &vectors[0];
        let results_before = index.search(query, k).unwrap();

        // Save
        let path = "/tmp/test_ip_diskann_large.bin";
        index.save(path).unwrap();
        println!("Saved index to {}", path);

        // Load
        let loaded = IPDiskANNIndex::load(path).unwrap();
        println!("Loaded index with {} vectors", loaded.len());

        // Test same query after load
        let results_after = loaded.search(query, k).unwrap();

        // Results should be identical
        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.id, after.id);
            assert!((before.distance - after.distance).abs() < 1e-6);
        }

        println!("Persistence validated - results identical");

        // Cleanup
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_insert_delete_search() {
        println!("\n=== Testing insert/delete workflow ===");

        let dimension = 128;
        let mut index = IPDiskANNIndex::new(dimension, None);

        // Insert 100 vectors
        let vectors = generate_random_vectors(100, dimension);
        for vector in vectors.iter() {
            index.insert(vector.clone()).unwrap();
        }

        println!("Inserted 100 vectors");
        assert_eq!(index.len(), 100);

        // Delete 50 vectors
        for i in 0..50 {
            index.delete(i).unwrap();
        }

        println!("Deleted 50 vectors");
        assert_eq!(index.len(), 50);

        // Search should still work
        let query = &vectors[60]; // Query with a remaining vector
        let results = index.search(query, 10).unwrap();

        println!("Search returned {} results", results.len());
        assert!(results.len() > 0);

        // Verify no deleted vectors in results
        for result in results {
            assert!(
                result.id >= 50,
                "Deleted vector {} found in results",
                result.id
            );
        }

        println!("Insert/delete workflow validated");
    }
}
