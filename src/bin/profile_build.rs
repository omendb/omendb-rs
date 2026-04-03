/// Write-path profiling binary.
///
/// Ingests vectors into a persistent VectorStore so the full pipeline is exercised:
/// HNSW index construction, auto-checkpoint (every 10K entries), and segment merge.
/// Uses real SIFT embeddings for realistic graph traversal patterns.
///
/// Run:
///   samply record ./target/release/profile_build --file benchmarks/data/sift-100k.f32bin --dim 128
///   samply record ./target/release/profile_build --file benchmarks/data/sift-100k.f32bin --dim 128 --repeat 5
use std::env;
use std::fs;
use std::time::Instant;

use omendb::vector::{Vector, VectorStore};
use serde_json::json;
use tempfile::TempDir;

fn main() {
    let args: Vec<String> = env::args().collect();

    let file = args
        .windows(2)
        .find(|w| w[0] == "--file")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "benchmarks/data/sift-100k.f32bin".to_string());

    let dim: usize = args
        .windows(2)
        .find(|w| w[0] == "--dim")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(128);

    // --repeat N: cycle through the dataset N times (for longer profiles)
    let repeat: usize = args
        .windows(2)
        .find(|w| w[0] == "--repeat")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(1);

    // Load raw f32 binary (N * dim * 4 bytes, little-endian)
    eprintln!("loading {} (dim={})", file, dim);
    let bytes = fs::read(&file).unwrap_or_else(|e| panic!("read {file}: {e}"));
    let n_base = bytes.len() / (dim * 4);
    let n_vectors = n_base * repeat;
    eprintln!(
        "{} vectors x {} repeats = {} total",
        n_base, repeat, n_vectors
    );

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("store.omen");
    let store = VectorStore::open_with_dimensions(&path, dim).expect("open");

    let chunk_size = 10_000;
    eprintln!("ingesting in chunks of {}...", chunk_size);
    let t0 = Instant::now();
    let mut ingested = 0usize;

    'outer: for _ in 0..repeat {
        for chunk_start in (0..n_base).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n_base);
            let mut batch = Vec::with_capacity(chunk_end - chunk_start);
            for i in chunk_start..chunk_end {
                let v: Vec<f32> = floats[i * dim..(i + 1) * dim].to_vec();
                batch.push((format!("v{ingested}"), Vector::new(v), json!({})));
                ingested += 1;
            }

            store.set_batch(batch).expect("set_batch");

            if ingested.is_multiple_of(100_000) {
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  {}K  ({:.0} vec/s)",
                    ingested / 1000,
                    ingested as f64 / elapsed
                );
                if ingested >= n_vectors {
                    break 'outer;
                }
            }
        }
    }

    store.flush().expect("flush");

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "done: {:.0} vec/s  ({:.1}s total, {} vectors)",
        n_vectors as f64 / elapsed,
        elapsed,
        n_vectors,
    );
}
