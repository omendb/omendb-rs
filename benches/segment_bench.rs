//! Multi-segment search benchmark.
//!
//! Tests search performance across multiple frozen segments.
//! Uses reduced segment capacity (25K) so 50K/100K vectors create 2-4 frozen segments.
//!
//! Run: cargo bench --bench segment_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use omendb::vector::{Vector, VectorStore};
use rand::Rng;
use serde_json::json;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.r#gen::<f32>()).collect()))
        .collect()
}

fn bench_multi_segment_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_segment");
    group.sample_size(10);

    let dim = 128;
    let queries = generate_vectors(50, dim);

    for n_vectors in [50_000, 100_000] {
        let vectors = generate_vectors(n_vectors, dim);

        // Use 25K segment capacity so 50K = 2 frozen, 100K = 4 frozen
        let mut store = VectorStore::new(dim).with_segment_capacity(25_000);
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(&format!("v{i}"), v.clone(), json!({"cat": i % 10}))
                .expect("set");
        }
        store.ensure_index_ready().expect("index ready");

        group.bench_with_input(
            BenchmarkId::new("search", format!("{n_vectors}_128D")),
            &n_vectors,
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

/// Compare single-segment vs multi-segment for the same vector count.
/// 50K tests below-threshold (2 frozen, sequential), 100K tests above-threshold (4 frozen, parallel).
fn bench_single_vs_multi_segment(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_comparison");
    group.sample_size(10);

    let dim = 128;
    let queries = generate_vectors(50, dim);

    for n_vectors in [50_000, 100_000] {
        let vectors = generate_vectors(n_vectors, dim);

        // Single segment (capacity > n_vectors)
        let mut single = VectorStore::new(dim).with_segment_capacity(200_000);
        for (i, v) in vectors.iter().enumerate() {
            single
                .set(&format!("v{i}"), v.clone(), json!({}))
                .expect("set");
        }
        single.ensure_index_ready().expect("index ready");

        // Multi segment (capacity = 25K, so 50K = 2 frozen, 100K = 4 frozen)
        let mut multi = VectorStore::new(dim).with_segment_capacity(25_000);
        for (i, v) in vectors.iter().enumerate() {
            multi
                .set(&format!("v{i}"), v.clone(), json!({}))
                .expect("set");
        }
        multi.ensure_index_ready().expect("index ready");

        let label = format!("{}", n_vectors / 1000);

        group.bench_function(format!("single_segment_{label}K"), |b| {
            b.iter(|| {
                for q in &queries {
                    black_box(single.search(q, 10, None).expect("search"));
                }
            })
        });

        group.bench_function(format!("multi_segment_{label}K"), |b| {
            b.iter(|| {
                for q in &queries {
                    black_box(multi.search(q, 10, None).expect("search"));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multi_segment_search,
    bench_single_vs_multi_segment
);
criterion_main!(benches);
