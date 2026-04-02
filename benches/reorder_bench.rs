//! BFS Reorder benchmark.
//!
//! Compares search performance with and without BFS graph reordering.
//! BFS reordering improves spatial locality by placing frequently-accessed
//! nodes close together in memory.
//!
//! Run: cargo bench --bench reorder_bench

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use omendb::types::Metric;
use omendb::vector::hnsw::{HNSWIndex, HNSWParams};
use rand::Rng;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

fn bench_reorder_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_reorder");
    group.sample_size(10);

    let dim = 128;
    let n_vectors = 50_000;
    let vectors = generate_vectors(n_vectors, dim);
    let queries = generate_vectors(100, dim);
    let params = HNSWParams::default();

    // 1. Build unoptimized index
    let unoptimized =
        HNSWIndex::build_parallel(dim, params.clone(), Metric::L2, false, vectors.clone())
            .expect("build unoptimized");

    // 2. Build optimized index (copy unoptimized, then reorder)
    // We can't clone HNSWIndex easily, so we build it again but optimize it.
    let mut optimized = HNSWIndex::build_parallel(dim, params, Metric::L2, false, vectors)
        .expect("build optimized");
    optimized.optimize_cache_locality().expect("optimize");

    let k = 10;
    let ef = 100;

    group.bench_function(BenchmarkId::new("search", "unoptimized"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(unoptimized.search(q, k, ef).expect("search"));
            }
        })
    });

    group.bench_function(BenchmarkId::new("search", "optimized"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(optimized.search(q, k, ef).expect("search"));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_reorder_impact);
criterion_main!(benches);
