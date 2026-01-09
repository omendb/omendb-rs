//! Filtered search benchmarks to validate ACORN-1 implementation.
//!
//! Tests different selectivity levels (1%, 10%, 50%) to verify the fix.
//! Run: cargo bench --bench filtered_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb_core::vector::store::MetadataFilter;
use omendb_core::vector::{Vector, VectorStore};
use rand::Rng;
use serde_json::json;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

/// Benchmark filtered search at different selectivity levels
fn bench_filtered_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_search");
    group.sample_size(20);

    let dim = 128;
    let n = 10_000;
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(50, dim);

    let mut store = VectorStore::new(dim);

    // Insert with category metadata (10 categories, ~1000 vectors each)
    for (i, v) in vectors.iter().enumerate() {
        store
            .insert_with_metadata(
                format!("d{i}"),
                v.clone(),
                json!({
                    "category": i % 10,
                    "group": i % 100,
                    "rare": i % 1000,
                }),
            )
            .expect("insert");
    }

    store.ensure_index_ready().expect("index ready");

    // 10% selectivity (1 of 10 categories)
    let filter_10pct = MetadataFilter::Eq("category".to_string(), json!(0));

    // 1% selectivity (1 of 100 groups)
    let filter_1pct = MetadataFilter::Eq("group".to_string(), json!(0));

    // 0.1% selectivity (1 of 1000 rare values)
    let filter_01pct = MetadataFilter::Eq("rare".to_string(), json!(0));

    // Baseline: unfiltered search
    group.bench_function(BenchmarkId::new("unfiltered", "100%"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(store.knn_search_readonly(q, 10, None).expect("search"));
            }
        })
    });

    // 10% selectivity - should use ACORN-1
    group.bench_function(BenchmarkId::new("filtered", "10%"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(
                    store
                        .knn_search_with_filter(q, 10, &filter_10pct)
                        .expect("search"),
                );
            }
        })
    });

    // 1% selectivity - ACORN-1 with more 2-hop expansion
    group.bench_function(BenchmarkId::new("filtered", "1%"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(
                    store
                        .knn_search_with_filter(q, 10, &filter_1pct)
                        .expect("search"),
                );
            }
        })
    });

    // 0.1% selectivity - very selective, tests 2-hop heavily
    group.bench_function(BenchmarkId::new("filtered", "0.1%"), |b| {
        b.iter(|| {
            for q in &queries {
                black_box(
                    store
                        .knn_search_with_filter(q, 10, &filter_01pct)
                        .expect("search"),
                );
            }
        })
    });

    group.finish();
}

/// Benchmark recall at different selectivity levels
fn bench_filtered_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_recall");
    group.sample_size(10);

    let dim = 128;
    let n = 10_000;
    let vectors = generate_vectors(n, dim);
    let queries = generate_vectors(20, dim);

    let mut store = VectorStore::new(dim);

    for (i, v) in vectors.iter().enumerate() {
        store
            .insert_with_metadata(
                format!("d{i}"),
                v.clone(),
                json!({
                    "category": i % 10,
                }),
            )
            .expect("insert");
    }

    store.ensure_index_ready().expect("index ready");

    let filter = MetadataFilter::Eq("category".to_string(), json!(0));

    // Measure recall: compare HNSW filtered results to brute force
    group.bench_function("recall_measurement", |b| {
        b.iter(|| {
            let mut total_recall = 0.0;
            for q in &queries {
                // Get HNSW results
                let hnsw_results = store
                    .knn_search_with_filter(q, 10, &filter)
                    .expect("search");

                // Get brute force ground truth
                let mut brute_force: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 10 == 0) // Same filter
                    .map(|(i, v)| {
                        let dist = q
                            .data
                            .iter()
                            .zip(&v.data)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>();
                        (i, dist)
                    })
                    .collect();
                brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                brute_force.truncate(10);

                // Calculate recall
                let ground_truth: std::collections::HashSet<_> =
                    brute_force.iter().map(|(i, _)| *i).collect();
                let found: std::collections::HashSet<_> =
                    hnsw_results.iter().map(|(i, _, _)| *i).collect();
                let recall = found.intersection(&ground_truth).count() as f64 / 10.0;
                total_recall += recall;
            }
            black_box(total_recall / queries.len() as f64)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_filtered_search, bench_filtered_recall);
criterion_main!(benches);
