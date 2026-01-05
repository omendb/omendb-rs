//! Search benchmarks to isolate hot path performance.
//!
//! Run: cargo bench --bench search_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb_core::vector::{Vector, VectorStore};
use rand::Rng;
use serde_json::json;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_qps");
    group.sample_size(20); // Fewer samples for faster benchmark

    for (n_vectors, dim) in [(10_000, 128), (10_000, 768), (10_000, 1536)] {
        let vectors = generate_vectors(n_vectors, dim);
        let queries = generate_vectors(100, dim);

        // Create and populate store
        let mut store = VectorStore::new(dim);
        for v in &vectors {
            store.insert(v.clone()).expect("insert");
        }

        // Ensure index is ready before benchmarking
        store.ensure_index_ready().expect("index ready");

        // Benchmark readonly path (fast - no &mut self overhead)
        group.bench_with_input(
            BenchmarkId::new("knn_search", format!("{n_vectors}x{dim}D")),
            &(n_vectors, dim),
            |b, _| {
                b.iter(|| {
                    for q in &queries {
                        black_box(store.knn_search_readonly(q, 10, None).expect("search"));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_search_ef_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_ef");
    group.sample_size(20);

    let dim = 768;
    let n = 10_000;
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    let mut store = VectorStore::new(dim);
    for v in &vectors {
        store.insert(v.clone()).expect("insert");
    }

    for ef in [64, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("768D", format!("ef={ef}")),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &queries {
                        black_box(store.knn_search_with_ef(q, 10, Some(ef)).expect("search"));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark with metadata (same path as Python)
fn bench_search_with_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_with_metadata");
    group.sample_size(20);

    let dim = 768;
    let n = 10_000;
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    let mut store = VectorStore::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        // Insert with ID and metadata like Python does
        store
            .insert_with_metadata(format!("d{i}"), v.clone(), json!({"cat": i % 10}))
            .expect("insert");
    }

    // Test with metadata path (search_with_ef returns metadata)
    group.bench_function("768D_ef64_with_metadata", |b| {
        b.iter(|| {
            for q in &queries {
                // This matches Python: search_with_ef returns (index, distance, metadata)
                black_box(store.search_with_ef(q, 10, None, Some(64)).expect("search"));
            }
        })
    });

    // Compare: without metadata
    group.bench_function("768D_ef64_no_metadata", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(store.knn_search_with_ef(q, 10, Some(64)).expect("search"));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_search_ef_comparison,
    bench_search_with_metadata
);
criterion_main!(benches);
