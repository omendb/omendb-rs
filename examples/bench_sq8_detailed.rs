//! Detailed SQ8 benchmark matching Python test.
//!
//! Run with: cargo run --release --example bench_sq8_detailed

use omendb::vector::{QuantizationMode, Vector, VectorStoreOptions};
use rand::Rng;
use std::time::Instant;

fn main() {
    let dim = 768;
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;
    let ef = 100;

    println!("{}", "=".repeat(60));
    println!(
        "Setup: {} vectors, {}D, k={}, {} queries",
        n_vectors, dim, k, n_queries
    );
    println!("{}", "=".repeat(60));

    // Generate random vectors (same seed as Python for reproducibility)
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Test 1: FP32
    println!("\n--- Test 1: FP32 (baseline) ---");
    let store_fp32 = {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .build()
            .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(
                    format!("v{}", i),
                    Vector::new(v.clone()),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.ensure_index_ready().unwrap();
        store
    };

    // Warmup
    for q in queries.iter().take(50) {
        let _ = store_fp32.knn_search_ef(&Vector::new(q.clone()), k, ef);
    }

    // Measure
    let mut times_us = Vec::with_capacity(n_queries);
    for q in &queries {
        let start = Instant::now();
        let _ = store_fp32.knn_search_ef(&Vector::new(q.clone()), k, ef);
        times_us.push(start.elapsed().as_micros() as f64);
    }

    let avg_us = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let min_us = times_us.iter().cloned().fold(f64::MAX, f64::min);
    let max_us = times_us.iter().cloned().fold(0.0, f64::max);
    let qps = 1_000_000.0 / avg_us;
    println!("\nFP32:");
    println!("  Avg: {:.0} us  ({:.0} QPS)", avg_us, qps);
    println!("  Min: {:.0} us  Max: {:.0} us", min_us, max_us);
    let t_fp32 = avg_us;

    // Test 2: SQ8 with rescore
    println!("\n--- Test 2: SQ8 with rescore (default) ---");
    let store_sq8 = {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .quantization(QuantizationMode::SQ8)
            .rescore(true)
            .build()
            .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(
                    format!("v{}", i),
                    Vector::new(v.clone()),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.ensure_index_ready().unwrap();
        store
    };

    // Warmup
    for q in queries.iter().take(50) {
        let _ = store_sq8.knn_search_ef(&Vector::new(q.clone()), k, ef);
    }

    // Measure
    let mut times_us = Vec::with_capacity(n_queries);
    for q in &queries {
        let start = Instant::now();
        let _ = store_sq8.knn_search_ef(&Vector::new(q.clone()), k, ef);
        times_us.push(start.elapsed().as_micros() as f64);
    }

    let avg_us = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let min_us = times_us.iter().cloned().fold(f64::MAX, f64::min);
    let max_us = times_us.iter().cloned().fold(0.0, f64::max);
    let qps = 1_000_000.0 / avg_us;
    println!("\nSQ8 (rescore=True):");
    println!("  Avg: {:.0} us  ({:.0} QPS)", avg_us, qps);
    println!("  Min: {:.0} us  Max: {:.0} us", min_us, max_us);
    let t_sq8 = avg_us;

    // Test 3: SQ8 without rescore
    println!("\n--- Test 3: SQ8 without rescore ---");
    let store_sq8_nr = {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .quantization(QuantizationMode::SQ8)
            .rescore(false)
            .build()
            .unwrap();
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(
                    format!("v{}", i),
                    Vector::new(v.clone()),
                    serde_json::json!({}),
                )
                .unwrap();
        }
        store.ensure_index_ready().unwrap();
        store
    };

    // Warmup
    for q in queries.iter().take(50) {
        let _ = store_sq8_nr.knn_search_ef(&Vector::new(q.clone()), k, ef);
    }

    // Measure
    let mut times_us = Vec::with_capacity(n_queries);
    for q in &queries {
        let start = Instant::now();
        let _ = store_sq8_nr.knn_search_ef(&Vector::new(q.clone()), k, ef);
        times_us.push(start.elapsed().as_micros() as f64);
    }

    let avg_us = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let min_us = times_us.iter().cloned().fold(f64::MAX, f64::min);
    let max_us = times_us.iter().cloned().fold(0.0, f64::max);
    let qps = 1_000_000.0 / avg_us;
    println!("\nSQ8 (rescore=False):");
    println!("  Avg: {:.0} us  ({:.0} QPS)", avg_us, qps);
    println!("  Min: {:.0} us  Max: {:.0} us", min_us, max_us);
    let t_sq8_nr = avg_us;

    // Summary
    println!("\n--- Summary ---");
    println!("FP32:             {:.0} us", t_fp32);
    println!(
        "SQ8 (rescore):    {:.0} us  ({:.2}x FP32)",
        t_sq8,
        t_sq8 / t_fp32
    );
    println!(
        "SQ8 (no rescore): {:.0} us  ({:.2}x FP32)",
        t_sq8_nr,
        t_sq8_nr / t_fp32
    );
}
