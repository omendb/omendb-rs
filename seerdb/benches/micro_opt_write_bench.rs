// Benchmark for write path micro-optimizations
// Measures per-write latency with inline hints on BlockBuilder, SSTableBuilder, BloomFilter, WAL

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use seerdb::{DBOptions, DB};
use std::sync::Arc;

fn bench_single_write_latency(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 16 * 1024 * 1024, // 16MB (avoid flushes)
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    let mut counter = 0;
    c.bench_function("single_write_memtable_hot", |b| {
        b.iter(|| {
            let key = format!("key_{:08}", counter);
            let value = format!("value_{:08}_data", counter);
            counter += 1;
            black_box(db.put(key.as_bytes(), value.as_bytes()).unwrap());
        });
    });
}

fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_throughput");
    group.sample_size(10);

    // Benchmark 10K writes (measures overall throughput)
    group.bench_function("write_10k_sequential", |b| {
        b.iter(|| {
            let temp_dir = tempfile::tempdir().unwrap();
            let opts = DBOptions {
                data_dir: temp_dir.path().to_path_buf(),
                memtable_capacity: 16 * 1024 * 1024, // 16MB
                ..Default::default()
            };
            let db = Arc::new(DB::open(opts).unwrap());

            for i in 0..10_000 {
                let key = format!("key_{:08}", i);
                let value = format!("value_{:08}_data", i);
                black_box(db.put(key.as_bytes(), value.as_bytes()).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_single_write_latency, bench_write_throughput);
criterion_main!(benches);
