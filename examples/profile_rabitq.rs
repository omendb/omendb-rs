//! RaBitQ/FastScan profiling benchmark
//!
//! Run with: cargo build --release --example profile_rabitq
//! Then: samply record ./target/release/examples/profile_rabitq

use omendb::vector::store::VectorStoreOptions;
use omendb::vector::types::Vector;
use omendb::vector::QuantizationMode;
use rand::Rng;
use serde_json::json;
use std::time::Instant;

fn main() {
    let dim = 768; // Common embedding dimension
    let n_vectors = 10_000;
    let n_queries = 500;
    let k = 10;

    println!("=== RaBitQ/FastScan Profiling ===");
    println!(
        "Vectors: {}, Dims: {}, Queries: {}",
        n_vectors, dim, n_queries
    );

    // Generate data
    println!("\nGenerating vectors...");
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vector> = (0..n_queries)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()))
        .collect();

    // Build RaBitQ index
    println!("Building RaBitQ index...");
    let mut store = VectorStoreOptions::new()
        .dimensions(dim)
        .quantization(QuantizationMode::rabitq())
        .ef_search(100)
        .build()
        .expect("Failed to build store");

    let build_start = Instant::now();
    for (i, vec) in vectors.iter().enumerate() {
        store
            .set(format!("v{}", i), Vector::new(vec.clone()), json!({}))
            .unwrap();
    }
    let build_time = build_start.elapsed();
    println!(
        "Build: {:.0} vec/s ({:.1}s)",
        n_vectors as f64 / build_time.as_secs_f64(),
        build_time.as_secs_f64()
    );

    // Warmup
    println!("\nWarming up...");
    for q in queries.iter().take(20) {
        let _ = store.search(q, k, None);
    }

    // Benchmark
    println!("Benchmarking {} queries...", n_queries);
    let search_start = Instant::now();
    for q in &queries {
        let _ = store.search(q, k, None);
    }
    let search_time = search_start.elapsed();

    let qps = n_queries as f64 / search_time.as_secs_f64();
    let latency_ms = search_time.as_secs_f64() * 1000.0 / n_queries as f64;
    println!("Search: {:.0} QPS ({:.2}ms avg)", qps, latency_ms);

    // Run more iterations for profiling
    println!("\nRunning profiling iterations...");
    for round in 1..=10 {
        let start = Instant::now();
        for q in &queries {
            let _ = store.search(q, k, None);
        }
        let elapsed = start.elapsed();
        println!(
            "Round {}: {:.0} QPS",
            round,
            n_queries as f64 / elapsed.as_secs_f64()
        );
    }
}
