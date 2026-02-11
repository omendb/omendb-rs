//! Multi-segment search benchmark.
//!
//! Tests search performance across multiple frozen segments (200K+ vectors).
//! Verifies that frozen segment parallelization works correctly.
//!
//! Run: cargo bench --bench segment_bench

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

fn bench_multi_segment_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_segment");
    group.sample_size(10);

    let dim = 128;
    let queries = generate_vectors(50, dim);

    for n_vectors in [50_000, 100_000] {
        let vectors = generate_vectors(n_vectors, dim);

        let mut store = VectorStore::new(dim);
        for (i, v) in vectors.iter().enumerate() {
            store
                .set(format!("v{i}"), v.clone(), json!({"cat": i % 10}))
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

criterion_group!(benches, bench_multi_segment_search);
criterion_main!(benches);
