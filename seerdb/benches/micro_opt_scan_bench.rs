// Benchmark for range scan micro-optimizations
// Measures impact of inline hints on iterator hot path

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::{DBOptions, DB};
use std::sync::Arc;

fn bench_prefix_scans(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 4 * 1024 * 1024,
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Insert graph-like data: user:000001:edges:000001, user:000001:edges:000002, ...
    // 10K nodes, 50 edges per node = 500K total entries
    let num_nodes = 10_000;
    let edges_per_node = 50;

    for src in 0..num_nodes {
        for dst in 0..edges_per_node {
            let key = format!("user:{:06}:edges:{:06}", src, dst);
            let value = format!("weight_{}", dst);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
    }

    db.flush().unwrap();

    let mut group = c.benchmark_group("prefix_scans");

    // Benchmark single prefix scan (cold cache)
    group.bench_function("single_prefix_scan_50_keys", |b| {
        b.iter(|| {
            let prefix = format!("user:{:06}:edges:", 5000);
            let count: usize = db
                .prefix(black_box(prefix.as_bytes()))
                .unwrap()
                .filter_map(|r| r.ok())
                .count();
            black_box(count);
        });
    });

    // Benchmark hot cache prefix scans (same key repeatedly)
    group.bench_function("hot_cache_prefix_scan_50_keys", |b| {
        let prefix = format!("user:{:06}:edges:", 5000);
        b.iter(|| {
            let count: usize = db
                .prefix(black_box(prefix.as_bytes()))
                .unwrap()
                .filter_map(|r| r.ok())
                .count();
            black_box(count);
        });
    });

    // Benchmark throughput: many random prefix scans
    group.bench_function("random_prefix_scans_1k", |b| {
        use rand::Rng;
        b.iter(|| {
            let mut rng = rand::thread_rng();
            for _ in 0..1000 {
                let src = rng.gen_range(0..num_nodes);
                let prefix = format!("user:{:06}:edges:", src);
                let count: usize = db
                    .prefix(prefix.as_bytes())
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .count();
                black_box(count);
            }
        });
    });

    group.finish();
}

fn bench_range_scans(c: &mut Criterion) {
    let temp_dir = tempfile::tempdir().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 4 * 1024 * 1024,
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Insert sequential data
    for i in 0..100_000 {
        let key = format!("key_{:08}", i);
        let value = format!("value_{:08}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    db.flush().unwrap();

    let mut group = c.benchmark_group("range_scans");

    for range_size in [100, 1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("range_scan", range_size),
            &range_size,
            |b, &size| {
                b.iter(|| {
                    let start = format!("key_{:08}", 50000);
                    let end = format!("key_{:08}", 50000 + size);
                    let count: usize = db
                        .range(black_box(start.as_bytes()), Some(black_box(end.as_bytes())))
                        .unwrap()
                        .filter_map(|r| r.ok())
                        .count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_prefix_scans, bench_range_scans);
criterion_main!(benches);
