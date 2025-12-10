//! Search profiling benchmark
//!
//! Run with: cargo build --release --example profile_search
//! Then: perf record -g ./target/release/examples/profile_search
//!       perf report

use omendb::vector::hnsw::{DistanceFunction, HNSWIndex, HNSWParams};
use rand::Rng;
use std::time::Instant;

fn main() {
    // Use 128D for target optimization (aim: 7K QPS)
    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 1000;
    let k = 10;
    let ef = 100;

    println!("Generating {} vectors of {}D...", n_vectors, dim);
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    println!("Building index...");
    // Use default M=16 (industry standard)
    let params = HNSWParams::default();
    let mut index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();

    let build_start = Instant::now();
    for vec in &vectors {
        index.insert(vec).unwrap();
    }
    let build_time = build_start.elapsed();
    println!(
        "Build: {:.0} vec/s",
        n_vectors as f64 / build_time.as_secs_f64()
    );

    // Warmup
    println!("Warming up...");
    for q in queries.iter().take(50) {
        let _ = index.search(q, k, ef);
    }

    // Benchmark
    println!("Benchmarking {} queries...", n_queries);
    let search_start = Instant::now();
    for q in &queries {
        let _ = index.search(q, k, ef);
    }
    let search_time = search_start.elapsed();

    let qps = n_queries as f64 / search_time.as_secs_f64();
    let latency_ms = search_time.as_secs_f64() * 1000.0 / n_queries as f64;

    println!("Search: {:.0} QPS ({:.2}ms avg)", qps, latency_ms);

    // Run more iterations for profiling
    println!("\nRunning 5 more rounds for profiling...");
    for round in 1..=5 {
        let start = Instant::now();
        for q in &queries {
            let _ = index.search(q, k, ef);
        }
        let elapsed = start.elapsed();
        println!(
            "Round {}: {:.0} QPS",
            round,
            n_queries as f64 / elapsed.as_secs_f64()
        );
    }
}
