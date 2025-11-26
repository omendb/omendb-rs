//! Transaction Benchmark
//!
//! Measures transaction throughput under various conditions:
//! - No contention (disjoint keys)
//! - High contention (same key)
//! - Compare with raw put/get overhead
//!
//! Run with: cargo bench --bench transaction_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use seerdb::{DBError, DBOptions, SyncPolicy, DB};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

const OPS_PER_ITERATION: u64 = 1_000;

/// Benchmark: Transaction commits with no contention (each thread has its own keys)
fn bench_transaction_no_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_no_contention");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    for threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("disjoint_keys", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let options = DBOptions {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: 128 * 1024 * 1024,
                        wal_sync_policy: SyncPolicy::None,
                        background_compaction: true,
                        background_flush: true,
                        ..Default::default()
                    };
                    let db = Arc::new(DB::open(options).unwrap());

                    let ops_per_thread = OPS_PER_ITERATION / num_threads as u64;
                    let committed = Arc::new(AtomicU64::new(0));

                    let handles: Vec<_> = (0..num_threads)
                        .map(|t_idx| {
                            let db = db.clone();
                            let committed = committed.clone();
                            thread::spawn(move || {
                                for i in 0..ops_per_thread {
                                    // Each thread has its own key space - no conflicts
                                    let key = format!("t{}_k{}", t_idx, i);

                                    let mut txn = db.begin_transaction();
                                    txn.put(key.as_bytes(), b"value").unwrap();

                                    if txn.commit().is_ok() {
                                        committed.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(committed.load(Ordering::Relaxed))
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Transaction commits with high contention (all threads compete for same key)
fn bench_transaction_high_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_high_contention");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    for threads in [2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("same_key_retry", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let options = DBOptions {
                        data_dir: dir.path().to_path_buf(),
                        memtable_capacity: 128 * 1024 * 1024,
                        wal_sync_policy: SyncPolicy::None,
                        background_compaction: true,
                        background_flush: true,
                        ..Default::default()
                    };
                    let db = Arc::new(DB::open(options).unwrap());

                    // Initialize counter
                    db.put(b"counter", b"0").unwrap();

                    let ops_per_thread = OPS_PER_ITERATION / num_threads as u64;
                    let committed = Arc::new(AtomicU64::new(0));
                    let conflicts = Arc::new(AtomicU64::new(0));

                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let db = db.clone();
                            let committed = committed.clone();
                            let conflicts = conflicts.clone();
                            thread::spawn(move || {
                                let mut local_committed = 0u64;
                                while local_committed < ops_per_thread {
                                    let mut txn = db.begin_transaction();

                                    // Read-modify-write pattern
                                    let current = txn.get(b"counter").unwrap();
                                    let value: i64 = current
                                        .map(|b| String::from_utf8_lossy(&b).parse().unwrap_or(0))
                                        .unwrap_or(0);
                                    let new_value = (value + 1).to_string();
                                    txn.put(b"counter", new_value.as_bytes()).unwrap();

                                    match txn.commit() {
                                        Ok(()) => {
                                            local_committed += 1;
                                            committed.fetch_add(1, Ordering::Relaxed);
                                        }
                                        Err(DBError::TransactionConflict(_)) => {
                                            conflicts.fetch_add(1, Ordering::Relaxed);
                                            // Retry loop continues
                                        }
                                        Err(e) => panic!("Unexpected error: {:?}", e),
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box((
                        committed.load(Ordering::Relaxed),
                        conflicts.load(Ordering::Relaxed),
                    ))
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Compare raw put vs transaction put (overhead measurement)
fn bench_transaction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_overhead");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    // Raw put (no transaction)
    group.bench_function("raw_put", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let options = DBOptions {
                data_dir: dir.path().to_path_buf(),
                memtable_capacity: 128 * 1024 * 1024,
                wal_sync_policy: SyncPolicy::None,
                background_compaction: true,
                background_flush: true,
                ..Default::default()
            };
            let db = DB::open(options).unwrap();

            for i in 0..OPS_PER_ITERATION {
                let key = format!("key_{}", i);
                db.put(key.as_bytes(), b"value").unwrap();
            }

            black_box(db.stats().total_puts)
        });
    });

    // Transaction put (single op per txn)
    group.bench_function("txn_put_single", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let options = DBOptions {
                data_dir: dir.path().to_path_buf(),
                memtable_capacity: 128 * 1024 * 1024,
                wal_sync_policy: SyncPolicy::None,
                background_compaction: true,
                background_flush: true,
                ..Default::default()
            };
            let db = DB::open(options).unwrap();

            for i in 0..OPS_PER_ITERATION {
                let key = format!("key_{}", i);
                let mut txn = db.begin_transaction();
                txn.put(key.as_bytes(), b"value").unwrap();
                txn.commit().unwrap();
            }

            black_box(db.stats().total_puts)
        });
    });

    // Transaction put (batched - 10 ops per txn)
    group.bench_function("txn_put_batch_10", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let options = DBOptions {
                data_dir: dir.path().to_path_buf(),
                memtable_capacity: 128 * 1024 * 1024,
                wal_sync_policy: SyncPolicy::None,
                background_compaction: true,
                background_flush: true,
                ..Default::default()
            };
            let db = DB::open(options).unwrap();

            let batch_size = 10;
            for batch_start in (0..OPS_PER_ITERATION).step_by(batch_size) {
                let mut txn = db.begin_transaction();
                for i in batch_start..(batch_start + batch_size as u64).min(OPS_PER_ITERATION) {
                    let key = format!("key_{}", i);
                    txn.put(key.as_bytes(), b"value").unwrap();
                }
                txn.commit().unwrap();
            }

            black_box(db.stats().total_puts)
        });
    });

    group.finish();
}

/// Benchmark: Read-heavy transaction workload
fn bench_transaction_read_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_read_heavy");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    group.bench_function("10_reads_1_write", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let options = DBOptions {
                data_dir: dir.path().to_path_buf(),
                memtable_capacity: 128 * 1024 * 1024,
                wal_sync_policy: SyncPolicy::None,
                background_compaction: true,
                background_flush: true,
                ..Default::default()
            };
            let db = DB::open(options).unwrap();

            // Pre-populate keys
            for i in 0..100 {
                let key = format!("key_{}", i);
                db.put(key.as_bytes(), b"initial").unwrap();
            }

            let mut committed = 0u64;
            for i in 0..OPS_PER_ITERATION {
                let mut txn = db.begin_transaction();

                // Read 10 keys (adds to read-set)
                for j in 0..10 {
                    let key = format!("key_{}", (i * 10 + j) % 100);
                    let _ = txn.get(key.as_bytes());
                }

                // Write 1 key
                let write_key = format!("key_{}", i % 100);
                txn.put(write_key.as_bytes(), b"updated").unwrap();

                if txn.commit().is_ok() {
                    committed += 1;
                }
            }

            black_box(committed)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transaction_no_contention,
    bench_transaction_high_contention,
    bench_transaction_overhead,
    bench_transaction_read_heavy,
);
criterion_main!(benches);
