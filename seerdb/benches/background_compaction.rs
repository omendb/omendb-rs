// Background compaction benchmark - measure throughput improvement
// Compares synchronous vs asynchronous compaction

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use seerdb::{DBOptions, SyncPolicy, DB};
use std::time::Duration;
use tempfile::tempdir;

fn benchmark_sync_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync_compaction");

    for num_writes in [1000, 5000, 10000] {
        group.throughput(Throughput::Elements(num_writes));

        group.bench_with_input(
            BenchmarkId::new("synchronous", num_writes),
            &num_writes,
            |b, &num_writes| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let options = DBOptions {
                            data_dir: dir.path().to_path_buf(),
                            memtable_capacity: 1024, // Small to trigger flushes
                            wal_sync_policy: SyncPolicy::None, // Fast for benchmark
                            background_compaction: false, // Synchronous
                            ..Default::default()
                        };
                        (DB::open(options).unwrap(), dir)
                    },
                    |(db, _dir)| {
                        for i in 0..num_writes {
                            let key = format!("key_{:06}", i);
                            let value = format!("value_{:06}", i);
                            db.put(key.as_bytes(), value.as_bytes()).unwrap();
                            black_box(());
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

fn benchmark_async_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_compaction");

    for num_writes in [1000, 5000, 10000] {
        group.throughput(Throughput::Elements(num_writes));

        group.bench_with_input(
            BenchmarkId::new("asynchronous", num_writes),
            &num_writes,
            |b, &num_writes| {
                b.iter_with_setup(
                    || {
                        let dir = tempdir().unwrap();
                        let options = DBOptions {
                            data_dir: dir.path().to_path_buf(),
                            memtable_capacity: 1024, // Small to trigger flushes
                            wal_sync_policy: SyncPolicy::None, // Fast for benchmark
                            background_compaction: true, // Asynchronous
                            ..Default::default()
                        };
                        (DB::open(options).unwrap(), dir)
                    },
                    |(db, _dir)| {
                        for i in 0..num_writes {
                            let key = format!("key_{:06}", i);
                            let value = format!("value_{:06}", i);
                            black_box(db.put(key.as_bytes(), value.as_bytes()).unwrap());
                        }
                        // DB dropped here, graceful shutdown waits for compactions
                    },
                );
            },
        );
    }

    group.finish();
}

fn benchmark_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction_comparison");
    group.throughput(Throughput::Elements(5000));

    // Synchronous compaction
    group.bench_function("sync_5k_writes", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let options = DBOptions {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 1024,
                    wal_sync_policy: SyncPolicy::None,
                    background_compaction: false,
                    ..Default::default()
                };
                (DB::open(options).unwrap(), dir)
            },
            |(db, _dir)| {
                for i in 0..5000 {
                    let key = format!("key_{:06}", i);
                    let value = format!("value_{:06}", i);
                    db.put(key.as_bytes(), value.as_bytes()).unwrap();
                    black_box(());
                }
            },
        );
    });

    // Asynchronous compaction
    group.bench_function("async_5k_writes", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let options = DBOptions {
                    data_dir: dir.path().to_path_buf(),
                    memtable_capacity: 1024,
                    wal_sync_policy: SyncPolicy::None,
                    background_compaction: true,
                    ..Default::default()
                };
                (DB::open(options).unwrap(), dir)
            },
            |(db, _dir)| {
                for i in 0..5000 {
                    let key = format!("key_{:06}", i);
                    let value = format!("value_{:06}", i);
                    db.put(key.as_bytes(), value.as_bytes()).unwrap();
                    black_box(());
                }
            },
        );
    });

    group.finish();
}

criterion_group! {
    name = background_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .sample_size(20);
    targets = benchmark_sync_compaction, benchmark_async_compaction, benchmark_comparison
}

criterion_main!(background_benches);
