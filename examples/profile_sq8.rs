//! SQ8 search profiling benchmark
//!
//! Run with: cargo build --release --example profile_sq8
//! Then: samply record ./target/release/examples/profile_sq8

use omendb_core::vector::hnsw::{DistanceFunction, HNSWIndex, HNSWParams};
use rand::Rng;
use std::time::Instant;

fn main() {
    let dim = 768; // Common embedding dimension
    let n_vectors = 10_000;
    let n_queries = 1000;
    let k = 10;
    let ef = 100;

    println!("=== Quantization Benchmark ===");
    println!("Vectors: {n_vectors}, Dim: {dim}, Queries: {n_queries}, k: {k}, ef: {ef}");
    println!();

    println!("=== SQ8 Profiling Benchmark ===");
    println!("Vectors: {n_vectors}, Dim: {dim}, Queries: {n_queries}");
    println!();

    // Generate random vectors
    println!("Generating vectors...");
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Build SQ8 index (lazy training: trains on first 256 vectors automatically)
    println!("Building SQ8 index...");
    let params = HNSWParams::default();
    let mut sq8_index = HNSWIndex::new_with_sq8(dim, params, DistanceFunction::L2).unwrap();

    let build_start = Instant::now();
    for vec in &vectors {
        sq8_index.insert(vec).unwrap();
    }
    let sq8_build_time = build_start.elapsed();
    println!(
        "SQ8 Build: {:.0} vec/s ({:.2}s)",
        n_vectors as f64 / sq8_build_time.as_secs_f64(),
        sq8_build_time.as_secs_f64()
    );

    // Build f32 index for comparison
    println!("Building f32 index...");
    let mut f32_index = HNSWIndex::new(dim, params, DistanceFunction::L2, false).unwrap();
    let build_start = Instant::now();
    for vec in &vectors {
        f32_index.insert(vec).unwrap();
    }
    let f32_build_time = build_start.elapsed();
    println!(
        "f32 Build: {:.0} vec/s ({:.2}s)",
        n_vectors as f64 / f32_build_time.as_secs_f64(),
        f32_build_time.as_secs_f64()
    );
    println!();

    // Warmup
    println!("Warming up...");
    for q in queries.iter().take(50) {
        let _ = sq8_index.search(q, k, ef);
        let _ = f32_index.search(q, k, ef);
    }

    // Benchmark SQ8 (with rescore - default)
    println!("Benchmarking SQ8 (with rescore)...");
    let search_start = Instant::now();
    for q in &queries {
        let _ = sq8_index.search(q, k, ef);
    }
    let sq8_search_time = search_start.elapsed();
    let sq8_qps = n_queries as f64 / sq8_search_time.as_secs_f64();

    // Benchmark SQ8 (no rescore - quantized distance only)
    println!("Benchmarking SQ8 (no rescore)...");
    let search_start = Instant::now();
    for q in &queries {
        let _ = sq8_index.search_asymmetric(q, k, ef);
    }
    let sq8_no_rescore_time = search_start.elapsed();
    let sq8_no_rescore_qps = n_queries as f64 / sq8_no_rescore_time.as_secs_f64();

    // Benchmark f32
    println!("Benchmarking f32...");
    let search_start = Instant::now();
    for q in &queries {
        let _ = f32_index.search(q, k, ef);
    }
    let f32_search_time = search_start.elapsed();
    let f32_qps = n_queries as f64 / f32_search_time.as_secs_f64();

    println!();
    println!("=== Results ===");
    println!(
        "SQ8 (rescore):    {:.0} QPS ({:.2}ms avg) - {:.2}x vs f32",
        sq8_qps,
        sq8_search_time.as_secs_f64() * 1000.0 / n_queries as f64,
        sq8_qps / f32_qps
    );
    println!(
        "SQ8 (no rescore): {:.0} QPS ({:.2}ms avg) - {:.2}x vs f32",
        sq8_no_rescore_qps,
        sq8_no_rescore_time.as_secs_f64() * 1000.0 / n_queries as f64,
        sq8_no_rescore_qps / f32_qps
    );
    println!(
        "f32:              {:.0} QPS ({:.2}ms avg)",
        f32_qps,
        f32_search_time.as_secs_f64() * 1000.0 / n_queries as f64
    );

    // Memory comparison
    println!();
    println!("=== Memory ===");
    let f32_mem = n_vectors * dim * 4; // 4 bytes per f32
    let sq8_mem = n_vectors * dim * 5; // 1 byte quantized + 4 bytes original
    println!("f32: {:.1} MB", f32_mem as f64 / 1_000_000.0);
    println!(
        "SQ8: {:.1} MB (quantized + original for rescore)",
        sq8_mem as f64 / 1_000_000.0
    );

    // Extended profiling runs
    println!();
    println!("=== Extended Profiling (5 rounds each) ===");
    for round in 1..=5 {
        let start = Instant::now();
        for q in &queries {
            let _ = sq8_index.search(q, k, ef);
        }
        let sq8_elapsed = start.elapsed();

        let start = Instant::now();
        for q in &queries {
            let _ = f32_index.search(q, k, ef);
        }
        let f32_elapsed = start.elapsed();

        println!(
            "Round {}: SQ8 {:.0} QPS, f32 {:.0} QPS, speedup {:.2}x",
            round,
            n_queries as f64 / sq8_elapsed.as_secs_f64(),
            n_queries as f64 / f32_elapsed.as_secs_f64(),
            (n_queries as f64 / sq8_elapsed.as_secs_f64())
                / (n_queries as f64 / f32_elapsed.as_secs_f64())
        );
    }
}
