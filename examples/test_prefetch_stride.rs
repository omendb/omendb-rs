//! Test different prefetch strides for HNSW
use omendb_core::vector::hnsw::{DistanceFunction, HNSWIndex, HNSWParams};
use rand::Rng;
use std::time::Instant;

fn main() {
    let dim = 768;
    let n_vectors = 10_000;
    let n_queries = 5_000;
    let k = 10;

    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();
    index.batch_insert(vectors).unwrap();

    // Warmup
    for q in queries.iter().take(100) {
        let _ = index.search(q, k, 100);
    }

    // Test different ef values to see scaling
    for ef in [50, 100, 200, 400] {
        let start = Instant::now();
        for q in &queries {
            let _ = index.search(q, k, ef);
        }
        let elapsed = start.elapsed();
        let qps = n_queries as f64 / elapsed.as_secs_f64();
        let avg_us = elapsed.as_micros() as f64 / n_queries as f64;
        println!("ef={:3}: {:5.0} QPS, {:5.0} µs avg", ef, qps, avg_us);
    }
}
