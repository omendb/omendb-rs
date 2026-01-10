//! Full performance test: all dimensions, single and batch
use omendb_core::vector::{Vector, VectorStore};
use rand::Rng;
use std::time::Instant;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

fn bench_dimension(dim: usize) {
    let n_vectors = 10_000;
    let n_queries = 100;
    let k = 10;

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
    let iters = 5;
    let start = Instant::now();
    for _ in 0..iters {
        for q in &queries {
            let _ = store.knn_search_readonly(q, k, None);
        }
    }
    let single_elapsed = start.elapsed();
    let single_qps = (iters * n_queries) as f64 / single_elapsed.as_secs_f64();

    // Batch benchmark
    let start = Instant::now();
    for _ in 0..iters {
        let _ = store.search_batch(&queries, k, None);
    }
    let batch_elapsed = start.elapsed();
    let batch_qps = (iters * n_queries) as f64 / batch_elapsed.as_secs_f64();

    println!(
        "| {}D | {:.0} QPS | {:.0} QPS | {:.1}x |",
        dim,
        single_qps,
        batch_qps,
        batch_qps / single_qps
    );
}

fn main() {
    println!("\n=== OmenDB Performance (10K vectors, k=10, M3 Max) ===\n");
    println!("| Dimension | Single | Batch | Speedup |");
    println!("|-----------|--------|-------|---------|");

    bench_dimension(128);
    bench_dimension(768);
    bench_dimension(1536);
}
