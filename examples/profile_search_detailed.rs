//! Detailed search profiling for samply
//! Run: samply record ./target/release/examples/profile_search_detailed

use omendb::vector::hnsw::{DistanceFunction, HNSWIndex, HNSWParams};
use rand::Rng;
use std::time::Instant;

fn main() {
    let dim = 768;
    let n_vectors = 10_000;
    let n_queries = 10_000; // More queries for better profiling
    let k = 10;
    let ef = 100;

    eprintln!("Generating {} vectors of {}D...", n_vectors, dim);
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    eprintln!("Building index with batch_insert...");
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();
    index.batch_insert(vectors).unwrap();

    // Warmup
    eprintln!("Warming up...");
    for q in queries.iter().take(100) {
        let _ = index.search(q, k, ef);
    }

    eprintln!("Running {} queries for profiling...", n_queries);
    let start = Instant::now();
    for q in &queries {
        let _ = index.search(q, k, ef);
    }
    let elapsed = start.elapsed();
    eprintln!("QPS: {:.0}", n_queries as f64 / elapsed.as_secs_f64());
}
