/// Benchmark full HNSW search with SQ8 vs FP32
use omendb::vector::QuantizationMode;
use omendb::{Vector, VectorStore};
use rand::Rng;
use std::time::Instant;

fn main() {
    for dim in [128, 768, 1536] {
        bench_dim(dim);
        println!();
    }
}

fn bench_dim(dim: usize) {
    let n_vectors = 10000;
    let n_queries = 100;

    println!("HNSW Search Benchmark: {} vectors, {}D", n_vectors, dim);
    println!("{}", "=".repeat(60));

    let mut rng = rand::thread_rng();

    // Create random vectors
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    // Create query
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let query_vec = Vector::new(query.clone());

    // Create FP32 store
    let mut fp32_store = VectorStore::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        fp32_store
            .set(
                format!("{i}"),
                Vector::new(v.clone()),
                serde_json::json!({}),
            )
            .unwrap();
    }

    // Create SQ8 store
    let mut sq8_store = VectorStore::new_with_quantization(dim, QuantizationMode::SQ8);
    for (i, v) in vectors.iter().enumerate() {
        sq8_store
            .set(
                format!("{i}"),
                Vector::new(v.clone()),
                serde_json::json!({}),
            )
            .unwrap();
    }

    // Warmup
    for _ in 0..10 {
        let _ = fp32_store.knn_search(&query_vec, 10);
        let _ = sq8_store.knn_search(&query_vec, 10);
    }

    // Benchmark FP32
    let start = Instant::now();
    for _ in 0..n_queries {
        let _ = fp32_store.knn_search(&query_vec, 10);
    }
    let fp32_us = start.elapsed().as_micros() as f64 / n_queries as f64;

    // Benchmark SQ8
    let start = Instant::now();
    for _ in 0..n_queries {
        let _ = sq8_store.knn_search(&query_vec, 10);
    }
    let sq8_us = start.elapsed().as_micros() as f64 / n_queries as f64;

    println!("FP32 search: {:.1} µs/query", fp32_us);
    println!("SQ8 search:  {:.1} µs/query", sq8_us);
    println!("Ratio:       {:.2}x", sq8_us / fp32_us);
}
