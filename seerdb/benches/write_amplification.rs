//! Write amplification benchmark
//!
//! Measures the ratio of physical bytes written to logical bytes written.
//! Compares traditional LSM (no VLog) vs WiscKey (with VLog).
//!
//! Run with: cargo bench --bench write_amplification

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use seerdb::{DBOptions, SyncPolicy, DB};
use std::path::PathBuf;
use tempfile::tempdir;

// Config
const VALUE_SIZE: usize = 8192; // 8KB
const VLOG_THRESHOLD: usize = 4096; // 4KB

fn bench_write_amplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_amplification");

    // 100k writes * 8KB = ~800MB logical data
    // Smaller size for CI bench to keep it reasonably fast (~seconds)
    // But large enough to trigger compactions
    const NUM_OPS: usize = 20_000;

    group.throughput(Throughput::Bytes((NUM_OPS * VALUE_SIZE) as u64));

    // Benchmark 1: Traditional LSM (No VLog)
    group.bench_function("lsm_traditional", |b| {
        b.iter(|| run_workload(NUM_OPS, None))
    });

    // Benchmark 2: WiscKey (With VLog)
    group.bench_function("wisckey_vlog", |b| {
        b.iter(|| run_workload(NUM_OPS, Some(VLOG_THRESHOLD)))
    });

    group.finish();
}

fn run_workload(num_ops: usize, vlog_threshold: Option<usize>) -> f64 {
    let dir = tempdir().unwrap();
    let opts = DBOptions {
        data_dir: dir.path().to_path_buf(),
        // Small memtable to trigger flushes and compaction
        memtable_capacity: 4 * 1024 * 1024,
        wal_sync_policy: SyncPolicy::None,
        background_compaction: false, // Sync for deterministic results
        vlog_threshold,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Use a constant value to avoid generation overhead
    // (compressible, but seerdb doesn't compress by default yet)
    let value = vec![b'x'; VALUE_SIZE];

    for i in 0..num_ops {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), &value).unwrap();
    }

    // Force flush
    db.flush().unwrap();

    let stats = db.stats();

    let logical = stats.logical_bytes_written;
    let physical = stats.physical_bytes_written;

    let amp = if logical > 0 {
        physical as f64 / logical as f64
    } else {
        0.0
    };

    // Return amp to prevent optimization
    black_box(amp)
}

criterion_group!(benches, bench_write_amplification);
criterion_main!(benches);
