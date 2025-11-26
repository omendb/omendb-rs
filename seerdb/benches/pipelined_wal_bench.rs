//! Benchmark for pipelined WAL optimizations
//!
//! Compares:
//! - Pipelining enabled vs disabled
//! - Different thread counts (1, 2, 4, 8)
//!
//! Run with: cargo bench --bench pipelined_wal_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use seerdb::wal::{PipelineConfig, PipelinedWAL, Record, SyncPolicy, WAL};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

const NUM_WRITES: usize = 1_000;
const VALUE_SIZE: usize = 100;

fn create_test_record(key: &[u8], value: &[u8]) -> Record {
    Record::Put {
        key: bytes::Bytes::copy_from_slice(key),
        value: bytes::Bytes::copy_from_slice(value),
    }
}

fn bench_pipelined_wal(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipelined_wal");
    group.throughput(Throughput::Elements(NUM_WRITES as u64));

    let value = vec![0u8; VALUE_SIZE];

    for threads in [1, 2, 4, 8] {
        // Benchmark with pipelining enabled
        group.bench_with_input(
            BenchmarkId::new("pipelined", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let wal_path = dir.path().join("test.wal");
                    let wal = Arc::new(Mutex::new(
                        WAL::create(&wal_path, SyncPolicy::None).unwrap(),
                    ));

                    let config = PipelineConfig {
                        enable_pipelining: true,
                        ..Default::default()
                    };
                    let pwal = Arc::new(PipelinedWAL::with_config(wal, config));

                    let counter = Arc::new(AtomicU64::new(0));
                    let writes_per_thread = NUM_WRITES / num_threads;

                    let handles: Vec<_> = (0..num_threads)
                        .map(|t| {
                            let pwal = pwal.clone();
                            let value = value.clone();
                            let counter = counter.clone();
                            thread::spawn(move || {
                                for i in 0..writes_per_thread {
                                    let key = format!("key_{:06}_{:06}", t, i);
                                    let record = create_test_record(key.as_bytes(), &value);
                                    pwal.put(record, |records| {
                                        // Simulate memtable write
                                        counter.fetch_add(records.len() as u64, Ordering::Relaxed);
                                    })
                                    .unwrap();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(counter.load(Ordering::Relaxed));
                })
            },
        );

        // Benchmark without pipelining
        group.bench_with_input(
            BenchmarkId::new("no_pipeline", threads),
            &threads,
            |b, &num_threads| {
                b.iter(|| {
                    let dir = tempdir().unwrap();
                    let wal_path = dir.path().join("test.wal");
                    let wal = Arc::new(Mutex::new(
                        WAL::create(&wal_path, SyncPolicy::None).unwrap(),
                    ));

                    let config = PipelineConfig {
                        enable_pipelining: false,
                        ..Default::default()
                    };
                    let pwal = Arc::new(PipelinedWAL::with_config(wal, config));

                    let counter = Arc::new(AtomicU64::new(0));
                    let writes_per_thread = NUM_WRITES / num_threads;

                    let handles: Vec<_> = (0..num_threads)
                        .map(|t| {
                            let pwal = pwal.clone();
                            let value = value.clone();
                            let counter = counter.clone();
                            thread::spawn(move || {
                                for i in 0..writes_per_thread {
                                    let key = format!("key_{:06}_{:06}", t, i);
                                    let record = create_test_record(key.as_bytes(), &value);
                                    pwal.put(record, |records| {
                                        // Simulate memtable write
                                        counter.fetch_add(records.len() as u64, Ordering::Relaxed);
                                    })
                                    .unwrap();
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(counter.load(Ordering::Relaxed));
                })
            },
        );
    }

    group.finish();
}

fn bench_adaptive_delay(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_delay");
    group.throughput(Throughput::Elements(NUM_WRITES as u64));

    let value = vec![0u8; VALUE_SIZE];
    let threads = 4;

    // Fixed delay (old behavior)
    group.bench_function("fixed_delay", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let wal_path = dir.path().join("test.wal");
            let wal = Arc::new(Mutex::new(
                WAL::create(&wal_path, SyncPolicy::None).unwrap(),
            ));

            let config = PipelineConfig {
                min_delay: Duration::from_micros(100),
                max_delay: Duration::from_micros(100), // Fixed
                enable_pipelining: true,
                ..Default::default()
            };
            let pwal = Arc::new(PipelinedWAL::with_config(wal, config));

            let counter = Arc::new(AtomicU64::new(0));
            let writes_per_thread = NUM_WRITES / threads;

            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let pwal = pwal.clone();
                    let value = value.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for i in 0..writes_per_thread {
                            let key = format!("key_{:06}_{:06}", t, i);
                            let record = create_test_record(key.as_bytes(), &value);
                            pwal.put(record, |records| {
                                counter.fetch_add(records.len() as u64, Ordering::Relaxed);
                            })
                            .unwrap();
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            black_box(counter.load(Ordering::Relaxed));
        })
    });

    // Adaptive delay (new behavior)
    group.bench_function("adaptive_delay", |b| {
        b.iter(|| {
            let dir = tempdir().unwrap();
            let wal_path = dir.path().join("test.wal");
            let wal = Arc::new(Mutex::new(
                WAL::create(&wal_path, SyncPolicy::None).unwrap(),
            ));

            let config = PipelineConfig {
                min_delay: Duration::from_micros(50),
                max_delay: Duration::from_micros(500),
                adaptive_threshold: 16,
                enable_pipelining: true,
                ..Default::default()
            };
            let pwal = Arc::new(PipelinedWAL::with_config(wal, config));

            let counter = Arc::new(AtomicU64::new(0));
            let writes_per_thread = NUM_WRITES / threads;

            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let pwal = pwal.clone();
                    let value = value.clone();
                    let counter = counter.clone();
                    thread::spawn(move || {
                        for i in 0..writes_per_thread {
                            let key = format!("key_{:06}_{:06}", t, i);
                            let record = create_test_record(key.as_bytes(), &value);
                            pwal.put(record, |records| {
                                counter.fetch_add(records.len() as u64, Ordering::Relaxed);
                            })
                            .unwrap();
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            black_box(counter.load(Ordering::Relaxed));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pipelined_wal, bench_adaptive_delay);
criterion_main!(benches);
