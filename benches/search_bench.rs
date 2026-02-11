//! Search benchmarks to isolate hot path performance.
//!
//! Run: cargo bench --bench search_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb::vector::{Vector, VectorStore};
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
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(format!("v{i}"), v.clone(), json!({}))
                .expect("set");
        }

        // Ensure index is ready before benchmarking
        store.ensure_index_ready().expect("index ready");

        // Benchmark search path
        group.bench_with_input(
            BenchmarkId::new("knn_search", format!("{n_vectors}x{dim}D")),
            &(n_vectors, dim),
            |b, _| {
                b.iter(|| {
                    for q in &queries {
                        black_box(store.search(q, 10, None).expect("search"));
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
    for (i, v) in vectors.iter().enumerate() {
        store
            .set(format!("v{i}"), v.clone(), json!({}))
            .expect("set");
    }

    for ef in [64, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("768D", format!("ef={ef}")),
            &ef,
            |b, &ef| {
                b.iter(|| {
                    for q in &queries {
                        black_box(
                            store
                                .search_with_options(q, 10, None, Some(ef), None)
                                .expect("search"),
                        );
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
        store
            .set(format!("d{i}"), v.clone(), json!({"cat": i % 10}))
            .expect("set");
    }

    // Test with metadata path
    group.bench_function("768D_ef64_with_metadata", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(
                    store
                        .search_with_options(q, 10, None, Some(64), None)
                        .expect("search"),
                );
            }
        })
    });

    // Compare: without ef override
    group.bench_function("768D_default_ef", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(store.search(q, 10, None).expect("search"));
            }
        })
    });

    group.finish();
}

/// Isolate metadata cloning cost: no metadata vs large metadata per record.
fn bench_metadata_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_overhead");
    group.sample_size(20);

    let dim = 128;
    let n = 10_000;
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(100, dim);

    // Store with empty metadata
    let mut store_empty = VectorStore::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        store_empty
            .set(format!("v{i}"), v.clone(), json!({}))
            .expect("set");
    }
    store_empty.ensure_index_ready().expect("index ready");

    // Store with ~1KB metadata per record
    let mut store_heavy = VectorStore::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        let metadata = json!({
            "title": format!("Document number {} with a reasonably long title for testing purposes", i),
            "category": format!("cat_{}", i % 100),
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
            "nested": {
                "field1": "value1",
                "field2": i as f64,
                "field3": true,
                "array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            },
            "description": "A longer text field that simulates real-world metadata with enough content to matter for cloning costs in benchmark measurements",
            "score": i as f64 * 0.1,
            "timestamp": 1700000000 + i,
        });
        store_heavy
            .set(format!("v{i}"), v.clone(), metadata)
            .expect("set");
    }
    store_heavy.ensure_index_ready().expect("index ready");

    group.bench_function("no_metadata", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(store_empty.search(q, 10, None).expect("search"));
            }
        })
    });

    group.bench_function("heavy_metadata", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(store_heavy.search(q, 10, None).expect("search"));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_search_ef_comparison,
    bench_search_with_metadata,
    bench_metadata_overhead
);
criterion_main!(benches);
