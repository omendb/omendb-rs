//! Mixed workload benchmark (Read/Write/Scan)
//!
//! Simulates a realistic concurrent workload:
//! - 50% Get
//! - 40% Put
//! - 10% Range Scan (short range)
//!
//! Run with: cargo bench --bench mixed_workload

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use seerdb::{DBOptions, SyncPolicy, DB};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

const OPS_PER_ITERATION: u64 = 10_000; // Total ops per measurement loop

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    group.throughput(Throughput::Elements(OPS_PER_ITERATION));

    // Pre-generate some keys to read
    const PRELOAD_KEYS: usize = 10_000;

    for threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("50get_40put_10scan", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    // Setup (included in measurement, so keep it fast or increase ops)
                    // To avoid setup cost in measurement, we could use iter_batched,
                    // but DB requires cleanup/Drop which iter_batched handles but
                    // multithreading makes complex.
                    // For 10k ops, setup (dir creation) is negligible (~ms vs ~100ms for ops).

                    let dir = tempdir().unwrap();
                    let options = DBOptions {
                        data_dir: dir.path().to_path_buf(),
                        // Large memtable to focus on concurrency overhead, not flush storm
                        memtable_capacity: 128 * 1024 * 1024,
                        wal_sync_policy: SyncPolicy::None,
                        background_compaction: true,
                        background_flush: true,
                        ..Default::default()
                    };
                    let db = Arc::new(DB::open(options).unwrap());

                    // Pre-load data (fast, single thread)
                    // We use a fixed set of keys for read hits
                    for i in 0..PRELOAD_KEYS {
                        let key = format!("key_{:08}", i);
                        // Small values to stress index/locking more than I/O
                        db.put(key.as_bytes(), b"value_init").unwrap();
                    }

                    let ops_per_thread = OPS_PER_ITERATION / num_threads as u64;
                    let total_ops = Arc::new(AtomicU64::new(0));

                    let handles: Vec<_> = (0..num_threads)
                        .map(|t_idx| {
                            let db = db.clone();
                            let total_ops = total_ops.clone();
                            thread::spawn(move || {
                                // Use thread-local RNG
                                let mut rng = rand::thread_rng();
                                let mut local_ops = 0;

                                for _ in 0..ops_per_thread {
                                    let op_type = rng.gen_range(0..100);
                                    let key_id = rng.gen_range(0..PRELOAD_KEYS * 2); // 50% hit rate for new keys
                                    let key = format!("key_{:08}", key_id);

                                    if op_type < 50 {
                                        // 50% Get
                                        let _ = black_box(db.get(key.as_bytes()));
                                    } else if op_type < 90 {
                                        // 40% Put
                                        let value = format!("val_{}_{}", t_idx, local_ops);
                                        let _ = black_box(db.put(key.as_bytes(), value.as_bytes()));
                                    } else {
                                        // 10% Scan (short range ~10 items)
                                        let end_key = format!("key_{:08}", key_id + 10);
                                        if let Ok(iter) =
                                            db.range(key.as_bytes(), Some(end_key.as_bytes()))
                                        {
                                            for item in iter {
                                                let _ = black_box(item);
                                            }
                                        }
                                    }
                                    local_ops += 1;
                                }
                                total_ops.fetch_add(local_ops, Ordering::Relaxed);
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(total_ops.load(Ordering::Relaxed));
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mixed_workload);
criterion_main!(benches);
