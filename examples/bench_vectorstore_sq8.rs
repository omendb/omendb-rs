//! VectorStore SQ8 benchmark (same path as Python)
//!
//! Run with: cargo run --release --example bench_vectorstore_sq8

use omendb_core::vector::{QuantizationMode, Vector, VectorStoreOptions};
use rand::Rng;
use std::time::Instant;

fn main() {
    let dim = 768;
    let n_vectors = 10_000;
    let n_queries = 1000;
    let k = 10;

    println!("=== VectorStore SQ8 vs FP32 Benchmark ===");
    println!("Vectors: {n_vectors}, Dim: {dim}, Queries: {n_queries}, k: {k}");
    println!();

    // Generate random vectors
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Build FP32 store
    println!("Building FP32 store...");
    let mut fp32_store = VectorStoreOptions::default()
        .dimensions(dim)
        .build()
        .unwrap();
    for (i, v) in vectors.iter().enumerate() {
        fp32_store
            .set(
                format!("v{i}"),
                Vector::new(v.clone()),
                serde_json::json!({}),
            )
            .unwrap();
    }
    fp32_store.ensure_index_ready().unwrap();

    // Build SQ8 store
    println!("Building SQ8 store...");
    let mut sq8_store = VectorStoreOptions::default()
        .dimensions(dim)
        .quantization(QuantizationMode::SQ8)
        .build()
        .unwrap();
    for (i, v) in vectors.iter().enumerate() {
        sq8_store
            .set(
                format!("v{i}"),
                Vector::new(v.clone()),
                serde_json::json!({}),
            )
            .unwrap();
    }
    sq8_store.ensure_index_ready().unwrap();

    // Build SQ8 no-rescore store
    println!("Building SQ8 (no rescore) store...");
    let mut sq8_nr_store = VectorStoreOptions::default()
        .dimensions(dim)
        .quantization(QuantizationMode::SQ8)
        .rescore(false)
        .build()
        .unwrap();
    for (i, v) in vectors.iter().enumerate() {
        sq8_nr_store
            .set(
                format!("v{i}"),
                Vector::new(v.clone()),
                serde_json::json!({}),
            )
            .unwrap();
    }
    sq8_nr_store.ensure_index_ready().unwrap();

    println!();

    // Warmup
    for q in queries.iter().take(50) {
        let _ = fp32_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
        let _ = sq8_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
        let _ = sq8_nr_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }

    // Benchmark FP32
    println!("Benchmarking FP32 (VectorStore.knn_search_ef)...");
    let start = Instant::now();
    for q in &queries {
        let _ = fp32_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let fp32_time = start.elapsed();
    let fp32_qps = n_queries as f64 / fp32_time.as_secs_f64();

    // Benchmark SQ8 with rescore
    println!("Benchmarking SQ8 (VectorStore.knn_search_ef, rescore=true)...");
    let start = Instant::now();
    for q in &queries {
        let _ = sq8_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let sq8_time = start.elapsed();
    let sq8_qps = n_queries as f64 / sq8_time.as_secs_f64();

    // Benchmark SQ8 no rescore
    println!("Benchmarking SQ8 (VectorStore.knn_search_ef, rescore=false)...");
    let start = Instant::now();
    for q in &queries {
        let _ = sq8_nr_store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let sq8_nr_time = start.elapsed();
    let sq8_nr_qps = n_queries as f64 / sq8_nr_time.as_secs_f64();

    println!();
    println!("=== Results ===");
    println!(
        "FP32:             {:.0} QPS ({:.2}ms avg)",
        fp32_qps,
        fp32_time.as_secs_f64() * 1000.0 / n_queries as f64
    );
    println!(
        "SQ8 (rescore):    {:.0} QPS ({:.2}ms avg) - {:.2}x vs FP32",
        sq8_qps,
        sq8_time.as_secs_f64() * 1000.0 / n_queries as f64,
        sq8_qps / fp32_qps
    );
    println!(
        "SQ8 (no rescore): {:.0} QPS ({:.2}ms avg) - {:.2}x vs FP32",
        sq8_nr_qps,
        sq8_nr_time.as_secs_f64() * 1000.0 / n_queries as f64,
        sq8_nr_qps / fp32_qps
    );
}
