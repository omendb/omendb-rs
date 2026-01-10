//! Benchmark SQ8 with RwLock wrapper (like Python bindings)
//!
//! Run with: cargo run --release --example bench_rwlock_sq8

use omendb_core::vector::{QuantizationMode, Vector, VectorStoreOptions};
use parking_lot::RwLock;
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let dim = 768;
    let n_vectors = 10_000;
    let n_queries = 1000;
    let k = 10;

    println!("=== RwLock Wrapper SQ8 Benchmark ===");
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

    // Build SQ8 store wrapped in Arc<RwLock<>> like Python
    println!("Building SQ8 store (wrapped in Arc<RwLock<>>)...");
    let store = {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .quantization(QuantizationMode::SQ8)
            .build()
            .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(
                    format!("v{i}"),
                    Vector::new(v.clone()),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.ensure_index_ready().unwrap();
        store
    };

    // Direct access (like Rust benchmark)
    println!("\n1. Direct access (no wrapper):");
    for q in queries.iter().take(50) {
        let _ = store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let start = Instant::now();
    for q in &queries {
        let _ = store.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let direct_time = start.elapsed();
    let direct_qps = n_queries as f64 / direct_time.as_secs_f64();
    println!(
        "   {:.0} QPS ({:.2}ms avg)",
        direct_qps,
        direct_time.as_secs_f64() * 1000.0 / n_queries as f64
    );

    // Wrapped in Arc<RwLock<>> (like Python)
    let wrapped_store = Arc::new(RwLock::new(store));

    println!("\n2. Arc<RwLock<>> access (like Python):");
    for q in queries.iter().take(50) {
        let inner = wrapped_store.read();
        let _ = inner.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let start = Instant::now();
    for q in &queries {
        let inner = wrapped_store.read();
        let _ = inner.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let wrapped_time = start.elapsed();
    let wrapped_qps = n_queries as f64 / wrapped_time.as_secs_f64();
    println!(
        "   {:.0} QPS ({:.2}ms avg)",
        wrapped_qps,
        wrapped_time.as_secs_f64() * 1000.0 / n_queries as f64
    );

    // Wrapped with Arc::clone each call (exactly like Python)
    println!("\n3. Arc clone + RwLock read each call:");
    for q in queries.iter().take(50) {
        let store_arc = Arc::clone(&wrapped_store);
        let inner = store_arc.read();
        let _ = inner.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let start = Instant::now();
    for q in &queries {
        let store_arc = Arc::clone(&wrapped_store);
        let inner = store_arc.read();
        let _ = inner.knn_search_ef(&Vector::new(q.clone()), k, 100);
    }
    let clone_time = start.elapsed();
    let clone_qps = n_queries as f64 / clone_time.as_secs_f64();
    println!(
        "   {:.0} QPS ({:.2}ms avg)",
        clone_qps,
        clone_time.as_secs_f64() * 1000.0 / n_queries as f64
    );

    println!("\n=== Summary ===");
    println!("Direct:      {:.0} QPS (baseline)", direct_qps);
    println!(
        "RwLock:      {:.0} QPS ({:.2}x)",
        wrapped_qps,
        wrapped_qps / direct_qps
    );
    println!(
        "Arc+RwLock:  {:.0} QPS ({:.2}x)",
        clone_qps,
        clone_qps / direct_qps
    );
}
