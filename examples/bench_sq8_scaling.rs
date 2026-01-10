//! Benchmark SQ8 and RaBitQ scaling with vector count.
//!
//! Run with: cargo run --release --example bench_sq8_scaling

use omendb_core::vector::{QuantizationMode, RaBitQParams, Vector, VectorStoreOptions};
use rand::Rng;
use std::time::Instant;

fn main() {
    // First: test with same config as Python to compare
    println!("=== 10K vectors, 768D ===");
    let n_vectors = 10000;
    let dim = 768;
    let k = 10;
    let ef = 100;
    let n_queries = 100;

    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // FP32
    {
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

        for q in queries.iter().take(50) {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let start = Instant::now();
        for q in &queries {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let us = start.elapsed().as_micros() as f64 / n_queries as f64;
        println!("FP32:    {:.0} us", us);
    }

    // SQ8
    {
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

        for q in queries.iter().take(50) {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let start = Instant::now();
        for q in &queries {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let us = start.elapsed().as_micros() as f64 / n_queries as f64;
        println!("SQ8:     {:.0} us", us);
    }

    // RaBitQ
    {
        let mut store = VectorStoreOptions::default()
            .dimensions(dim)
            .quantization(QuantizationMode::RaBitQ(RaBitQParams::default()))
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

        for q in queries.iter().take(50) {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let start = Instant::now();
        for q in &queries {
            let _ = store.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let us = start.elapsed().as_micros() as f64 / n_queries as f64;
        println!("RaBitQ:  {:.0} us", us);
    }

    println!("\n=== Scaling test ===");
    let configs = [(1000, 128), (1000, 768), (5000, 768), (10000, 768)];

    for (n_vectors, dim) in configs {
        let ef = 100;
        let k = 10;
        let n_queries = 100;

        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        // FP32
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

        let start = Instant::now();
        for q in &queries {
            let _ = store_fp32.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let fp32_us = start.elapsed().as_micros() as f64 / n_queries as f64;

        // SQ8
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

        let start = Instant::now();
        for q in &queries {
            let _ = store_sq8.knn_search_ef(&Vector::new(q.clone()), k, ef);
        }
        let sq8_us = start.elapsed().as_micros() as f64 / n_queries as f64;

        println!(
            "{:5}x{:4}: FP32={:5.0}us  SQ8={:5.0}us  ratio={:.2}x",
            n_vectors,
            dim,
            fp32_us,
            sq8_us,
            sq8_us / fp32_us
        );
    }
}
