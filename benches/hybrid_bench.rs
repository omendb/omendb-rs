//! Hybrid search benchmarks comparing vector, text, and combined search.
//!
//! Run: cargo bench --bench hybrid_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb_core::text::TextIndex;
use omendb_core::vector::{Vector, VectorStore, VectorStoreOptions};
use rand::Rng;
use serde_json::json;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| Vector::new((0..dim).map(|_| rng.gen::<f32>()).collect()))
        .collect()
}

fn generate_text_corpus(n: usize) -> Vec<String> {
    let words = [
        "database",
        "vector",
        "search",
        "query",
        "index",
        "storage",
        "memory",
        "performance",
        "fast",
        "efficient",
        "scalable",
        "distributed",
        "embedded",
        "machine",
        "learning",
        "neural",
        "network",
        "model",
        "training",
        "inference",
        "algorithm",
        "optimization",
        "parallel",
        "concurrent",
        "async",
        "thread",
        "rust",
        "python",
        "javascript",
        "typescript",
        "node",
        "server",
        "client",
    ];
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let n_words = rng.gen_range(5..15);
            (0..n_words)
                .map(|_| words[rng.gen_range(0..words.len())])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Benchmark text search alone (BM25 via tantivy)
fn bench_text_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_search");
    group.sample_size(20);

    for n in [1_000, 10_000] {
        let texts = generate_text_corpus(n);
        let queries = ["vector database", "machine learning", "rust performance"];

        let mut index = TextIndex::open_in_memory().unwrap();
        for (i, text) in texts.iter().enumerate() {
            index.index_document(&format!("doc{i}"), text).unwrap();
        }
        index.commit().unwrap();

        group.bench_with_input(BenchmarkId::new("bm25", format!("{n}_docs")), &n, |b, _| {
            b.iter(|| {
                for q in &queries {
                    black_box(index.search(q, 10).unwrap());
                }
            })
        });
    }

    group.finish();
}

/// Benchmark hybrid search with different alpha values
fn bench_hybrid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_search");
    group.sample_size(20);

    let n = 10_000;
    let dim = 384; // Common embedding dimension (e.g., all-MiniLM-L6-v2)

    let vectors = generate_vectors(n, dim);
    let texts = generate_text_corpus(n);
    let query_vectors = generate_vectors(10, dim);
    let query_texts = ["vector database", "machine learning", "rust performance"];

    // Create store with text search enabled
    let options = VectorStoreOptions::default()
        .dimensions(dim)
        .text_search(true);
    let mut store = options.build().unwrap();

    for (i, (vec, text)) in vectors.iter().zip(texts.iter()).enumerate() {
        store
            .set_with_text(format!("doc{i}"), vec.clone(), text, json!({}))
            .unwrap();
    }
    store.flush().unwrap();

    // Benchmark different alpha values
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0] {
        let alpha_label = if alpha == 0.0 {
            "text_only"
        } else if alpha == 1.0 {
            "vector_only"
        } else {
            &format!("alpha_{}", (alpha * 10.0) as i32)
        };

        group.bench_with_input(BenchmarkId::new("rrf", alpha_label), &alpha, |b, &alpha| {
            b.iter(|| {
                for (qv, qt) in query_vectors.iter().zip(query_texts.iter().cycle()) {
                    black_box(store.hybrid_search(qv, qt, 10, Some(alpha)).unwrap());
                }
            })
        });
    }

    group.finish();
}

/// Compare vector-only vs hybrid search overhead
fn bench_hybrid_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_overhead");
    group.sample_size(20);

    let n = 10_000;
    let dim = 768;

    let vectors = generate_vectors(n, dim);
    let texts = generate_text_corpus(n);
    let query_vectors = generate_vectors(100, dim);
    let query_text = "vector database search";

    // Vector-only store
    let mut vector_store = VectorStore::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        vector_store
            .insert_with_metadata(format!("doc{i}"), v.clone(), json!({}))
            .unwrap();
    }

    // Hybrid store
    let options = VectorStoreOptions::default()
        .dimensions(dim)
        .text_search(true);
    let mut hybrid_store = options.build().unwrap();
    for (i, (v, t)) in vectors.iter().zip(texts.iter()).enumerate() {
        hybrid_store
            .set_with_text(format!("doc{i}"), v.clone(), t, json!({}))
            .unwrap();
    }
    hybrid_store.flush().unwrap();

    group.bench_function("vector_only_768D", |b| {
        b.iter(|| {
            for q in &query_vectors {
                black_box(vector_store.knn_search(q, 10).unwrap());
            }
        })
    });

    group.bench_function("hybrid_768D", |b| {
        b.iter(|| {
            for q in &query_vectors {
                black_box(hybrid_store.hybrid_search(q, query_text, 10, None).unwrap());
            }
        })
    });

    group.finish();
}

/// Benchmark RRF fusion overhead (just the fusion, not the searches)
fn bench_rrf_fusion(c: &mut Criterion) {
    use omendb_core::text::{reciprocal_rank_fusion, weighted_reciprocal_rank_fusion};

    let mut group = c.benchmark_group("rrf_fusion");

    // Simulate search results of various sizes
    for n in [10, 100, 1000] {
        let vector_results: Vec<_> = (0..n)
            .map(|i| (format!("vec_{i}"), i as f32 * 0.1))
            .collect();
        let text_results: Vec<_> = (0..n)
            .map(|i| (format!("text_{i}"), 100.0 - i as f32))
            .collect();

        group.bench_with_input(BenchmarkId::new("unweighted", n), &n, |b, _| {
            b.iter(|| {
                black_box(reciprocal_rank_fusion(
                    vector_results.clone(),
                    text_results.clone(),
                    10,
                    60,
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("weighted", n), &n, |b, _| {
            b.iter(|| {
                black_box(weighted_reciprocal_rank_fusion(
                    vector_results.clone(),
                    text_results.clone(),
                    10,
                    60,
                    0.7,
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_text_search,
    bench_hybrid_search,
    bench_hybrid_overhead,
    bench_rrf_fusion
);
criterion_main!(benches);
