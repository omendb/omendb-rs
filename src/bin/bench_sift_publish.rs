use std::env;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use omendb::{MetadataFilter, Vector, VectorStore};
use serde_json::json;
use tempfile::TempDir;

#[derive(Debug)]
struct Args {
    vectors_file: String,
    queries_file: String,
    ground_truth_file: String,
    n_vectors: usize,
    n_queries: usize,
    dimensions: usize,
    k: usize,
    warmup: usize,
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn parse_required<T: std::str::FromStr>(args: &[String], flag: &str) -> T {
    value_after(args, flag)
        .unwrap_or_else(|| panic!("missing required {flag}"))
        .parse()
        .unwrap_or_else(|_| panic!("invalid value for {flag}"))
}

fn parse_args() -> Args {
    let raw: Vec<String> = env::args().collect();
    Args {
        vectors_file: value_after(&raw, "--vectors-file").expect("missing required --vectors-file"),
        queries_file: value_after(&raw, "--queries-file").expect("missing required --queries-file"),
        ground_truth_file: value_after(&raw, "--ground-truth-file")
            .expect("missing required --ground-truth-file"),
        n_vectors: parse_required(&raw, "--vectors"),
        n_queries: parse_required(&raw, "--queries"),
        dimensions: parse_required(&raw, "--dimensions"),
        k: value_after(&raw, "--k")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10),
        warmup: value_after(&raw, "--warmup")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10),
    }
}

fn read_f32_matrix(path: impl AsRef<Path>, rows: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let path = path.as_ref();
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = rows * dimensions * std::mem::size_of::<f32>();
    assert_eq!(
        bytes.len(),
        expected,
        "{} has {} bytes, expected {}",
        path.display(),
        bytes.len(),
        expected
    );

    bytes
        .chunks_exact(dimensions * std::mem::size_of::<f32>())
        .map(|row| {
            row.chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect()
}

fn read_i32_matrix(path: impl AsRef<Path>, rows: usize, k: usize) -> Vec<Vec<i32>> {
    let path = path.as_ref();
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let expected = rows * k * std::mem::size_of::<i32>();
    assert_eq!(
        bytes.len(),
        expected,
        "{} has {} bytes, expected {}",
        path.display(),
        bytes.len(),
        expected
    );

    bytes
        .chunks_exact(k * std::mem::size_of::<i32>())
        .map(|row| {
            row.chunks_exact(std::mem::size_of::<i32>())
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
        .collect()
}

fn benchmark_build(vectors: &[Vec<f32>], dimensions: usize) -> (VectorStore, serde_json::Value) {
    let tmp = TempDir::new().expect("tempdir");
    let store = VectorStore::open_with_dimensions(tmp.path().join("rust_publish.omen"), dimensions)
        .expect("open store");
    let batch: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(i, vector)| {
            (
                format!("d{i}"),
                Vector::new(vector.clone()),
                json!({"cat": i % 10}),
            )
        })
        .collect();

    let start = Instant::now();
    store.set_batch(batch).expect("set_batch");
    let elapsed = start.elapsed().as_secs_f64();

    // Keep the temp directory alive through the store by leaking it for process lifetime.
    std::mem::forget(tmp);

    (
        store,
        json!({
            "vectors": vectors.len(),
            "time_s": elapsed,
            "vec_per_s": vectors.len() as f64 / elapsed,
        }),
    )
}

fn benchmark_search(
    store: &VectorStore,
    queries: &[Vec<f32>],
    k: usize,
    warmup: usize,
) -> serde_json::Value {
    for query in queries.iter().take(warmup) {
        black_box(
            store
                .search(&Vector::new(query.clone()), k, None)
                .expect("warmup search"),
        );
    }

    let mut latencies = Vec::with_capacity(queries.len());
    let start = Instant::now();
    for query in queries {
        let query = Vector::new(query.clone());
        let query_start = Instant::now();
        black_box(store.search(&query, k, None).expect("search"));
        latencies.push(query_start.elapsed().as_secs_f64() * 1000.0);
    }
    let total = start.elapsed().as_secs_f64();
    latencies.sort_by(f64::total_cmp);

    json!({
        "queries": queries.len(),
        "time_s": total,
        "qps": queries.len() as f64 / total,
        "latency_avg_ms": latencies.iter().sum::<f64>() / latencies.len() as f64,
        "latency_p50_ms": latencies[latencies.len() / 2],
        "latency_p99_ms": latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)],
    })
}

fn benchmark_recall(
    store: &VectorStore,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<i32>],
    k: usize,
) -> serde_json::Value {
    let n_queries = queries.len().min(100);
    let mut total_recall = 0.0;

    for (query, expected) in queries.iter().zip(ground_truth.iter()).take(n_queries) {
        let results = store
            .search(&Vector::new(query.clone()), k, None)
            .expect("recall search");
        let returned: std::collections::HashSet<_> = results
            .iter()
            .filter_map(|result| result.id.strip_prefix('d')?.parse::<i32>().ok())
            .collect();
        let found = expected
            .iter()
            .take(k)
            .filter(|id| returned.contains(id))
            .count();
        total_recall += found as f64 / k as f64;
    }

    json!({
        "recall_at_k": total_recall / n_queries as f64,
        "k": k,
        "n_queries": n_queries,
    })
}

fn benchmark_filtered_search(
    store: &VectorStore,
    queries: &[Vec<f32>],
    k: usize,
    warmup: usize,
) -> serde_json::Value {
    let filter = MetadataFilter::Eq("cat".to_string(), json!(5));
    for query in queries.iter().take(warmup) {
        black_box(
            store
                .search(&Vector::new(query.clone()), k, Some(&filter))
                .expect("filtered warmup"),
        );
    }

    let start = Instant::now();
    for query in queries {
        black_box(
            store
                .search(&Vector::new(query.clone()), k, Some(&filter))
                .expect("filtered search"),
        );
    }
    let total = start.elapsed().as_secs_f64();

    json!({
        "queries": queries.len(),
        "time_s": total,
        "qps": queries.len() as f64 / total,
        "latency_ms": (total / queries.len() as f64) * 1000.0,
    })
}

fn benchmark_batch_search(
    store: &VectorStore,
    queries: &[Vec<f32>],
    k: usize,
) -> serde_json::Value {
    let query_vectors: Vec<_> = queries.iter().cloned().map(Vector::new).collect();
    let start = Instant::now();
    let results = store.search_batch(&query_vectors, k, None);
    for result in results {
        black_box(result.expect("batch search"));
    }
    let total = start.elapsed().as_secs_f64();

    json!({
        "queries": queries.len(),
        "time_s": total,
        "qps": queries.len() as f64 / total,
        "latency_ms": (total / queries.len() as f64) * 1000.0,
    })
}

fn main() {
    let args = parse_args();
    let vectors = read_f32_matrix(&args.vectors_file, args.n_vectors, args.dimensions);
    let queries = read_f32_matrix(&args.queries_file, args.n_queries, args.dimensions);
    let ground_truth = read_i32_matrix(&args.ground_truth_file, args.n_queries, args.k);

    let (store, build) = benchmark_build(&vectors, args.dimensions);
    let search = benchmark_search(&store, &queries, args.k, args.warmup);
    let recall = benchmark_recall(&store, &queries, &ground_truth, args.k);
    let filtered = benchmark_filtered_search(&store, &queries, args.k, args.warmup);
    let batch = benchmark_batch_search(&store, &queries, args.k);

    println!(
        "{}",
        json!({
            "config": {
                "n_vectors": args.n_vectors,
                "dimensions": args.dimensions,
                "n_queries": args.n_queries,
                "quantize_bits": 0,
                "dataset": format!("sift-{}k", args.n_vectors / 1000),
                "dataset_family": "sift",
                "api": "rust",
            },
            "build": build,
            "search": search,
            "recall": recall,
            "filtered": filtered,
            "batch": batch,
        })
    );
}
