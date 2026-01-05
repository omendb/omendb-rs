//! Micro-benchmarks for distance calculations.
//!
//! Run: cargo bench --bench distance_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb_core::distance::{dot_product, l2_distance_squared};
use rand::Rng;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance_squared");

    for dim in [128, 384, 768, 1536] {
        let vectors = generate_vectors(1000, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for v in &vectors {
                    black_box(l2_distance_squared(&query, v));
                }
            })
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [128, 384, 768, 1536] {
        let vectors = generate_vectors(1000, dim);
        let query = generate_vectors(1, dim).pop().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for v in &vectors {
                    black_box(dot_product(&query, v));
                }
            })
        });
    }

    group.finish();
}

/// Compare with sequential (non-random) access
fn bench_l2_sequential_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_sequential_1000");

    for dim in [128, 768, 1536] {
        // Contiguous memory layout
        let n = 1000;
        let flat: Vec<f32> = generate_vectors(n, dim).into_iter().flatten().collect();
        let query = generate_vectors(1, dim).pop().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| {
                for i in 0..n {
                    let start = i * dim;
                    let end = start + dim;
                    black_box(l2_distance_squared(&query, &flat[start..end]));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_dot_product,
    bench_l2_sequential_access
);
criterion_main!(benches);
