//! Quick test: single vs batch performance
use omendb_core::vector::{Vector, VectorStore};
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

fn main() {
    let dim = 128;
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;

    println!("Creating store with {n_vectors} vectors at {dim}D...");
    let vectors = generate_vectors(n_vectors, dim);
    let queries = generate_vectors(n_queries, dim);

    let mut store = VectorStore::new(dim);
    for v in &vectors {
        store.insert(v.clone()).expect("insert");
    }
    store.ensure_index_ready().expect("ready");

    // Warmup
    for q in &queries {
        let _ = store.knn_search_readonly(q, k, None);
    }
    let _ = store.search_batch(&queries, k, None);

    // Single-query benchmark
    let start = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        for q in &queries {
            let _ = store.knn_search_readonly(q, k, None);
        }
    }
    let single_elapsed = start.elapsed();
    let single_total = (iters * n_queries) as f64;
    let single_qps = single_total / single_elapsed.as_secs_f64();

    // Batch benchmark
    let start = Instant::now();
    for _ in 0..iters {
        let _ = store.search_batch(&queries, k, None);
    }
    let batch_elapsed = start.elapsed();
    let batch_total = (iters * n_queries) as f64;
    let batch_qps = batch_total / batch_elapsed.as_secs_f64();

    println!("\n=== 128D Results (10K vectors, k=10) ===");
    println!(
        "Single-query: {:.0} QPS ({:.2}ms per 100q)",
        single_qps,
        single_elapsed.as_secs_f64() * 1000.0 / iters as f64
    );
    println!(
        "Batch (parallel): {:.0} QPS ({:.2}ms per 100q)",
        batch_qps,
        batch_elapsed.as_secs_f64() * 1000.0 / iters as f64
    );
    println!("Batch speedup: {:.1}x", batch_qps / single_qps);
}
