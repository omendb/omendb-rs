// Multi-threaded benchmark to validate sharded buffer pool performance
// Compares lock contention reduction across 1, 2, 4, 8 threads

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::buffer::{BufferPool, BufferPoolError, BufferPoolOptions, PageId};
use std::thread;

fn bench_buffer_pool_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_sharding");
    group.sample_size(20);

    // Create buffer pool with 128MB capacity, 16KB frames
    // This gives us 8192 frames total, or 512 per shard (16 shards)
    let pool_opts = BufferPoolOptions {
        capacity_bytes: 128 * 1024 * 1024,
        frame_size: 16 * 1024,
        num_shards: 16,
    };
    let pool = BufferPool::new(pool_opts);

    // Workload: Random page accesses across many files
    // IMPORTANT: Keep working set much smaller than pool size to avoid deadlock
    // 8192 frames total / 16 shards = 512 frames per shard
    // Use only 800 unique pages total (~10% capacity, ~50 pages per shard)
    let num_files = 8;
    let pages_per_file = 100;

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_random_reads", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let pool = pool.clone();
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let pool = pool.clone();
                            thread::spawn(move || {
                                // Each thread does 100 random page loads (conservative to avoid shard exhaustion)
                                // Use thread_id as seed for deterministic randomness
                                let mut seed = 12345 + thread_id as u64;

                                for _ in 0..100 {
                                    // LCG for fast pseudo-random numbers
                                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                                    let file_id = (seed % num_files) as u64;

                                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                                    let offset = (seed % pages_per_file) as u64;

                                    let page_id = PageId { file_id, offset };

                                    // Simulate loading a page
                                    let frame_ref = pool
                                        .get_page(page_id, |buf| {
                                            // Simulate disk read by filling buffer with data
                                            buf.fill(0xAB);
                                            Ok::<(), BufferPoolError>(())
                                        })
                                        .unwrap();

                                    // CRITICAL: Drop frame immediately to release pin
                                    // Without this, all frames stay pinned and eviction deadlocks
                                    black_box(&frame_ref);
                                    drop(frame_ref);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_buffer_pool_contention_focused(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_hot_set");
    group.sample_size(20);

    // Scenario: Hot set contention
    // All threads compete for same small set of pages (worst case for sharding)
    let pool_opts = BufferPoolOptions {
        capacity_bytes: 128 * 1024 * 1024,
        frame_size: 16 * 1024,
        num_shards: 16,
    };
    let pool = BufferPool::new(pool_opts);

    // Small hot set: 10 files, 10 pages each = 100 total pages
    // This easily fits in buffer pool (8192 frames)
    let num_files = 10;
    let pages_per_file = 10;

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("hot_set_contention", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let pool = pool.clone();
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let pool = pool.clone();
                            thread::spawn(move || {
                                let mut seed = 12345 + thread_id as u64;

                                for _ in 0..100 {
                                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                                    let file_id = (seed % num_files) as u64;

                                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                                    let offset = (seed % pages_per_file) as u64;

                                    let page_id = PageId { file_id, offset };

                                    let frame_ref = pool
                                        .get_page(page_id, |buf| {
                                            buf.fill(0xAB);
                                            Ok::<(), BufferPoolError>(())
                                        })
                                        .unwrap();

                                    // CRITICAL: Drop frame immediately to release pin
                                    black_box(&frame_ref);
                                    drop(frame_ref);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_buffer_pool_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_scalability");
    group.sample_size(10);

    // Measure near-linear scalability on independent page sets
    // Each thread accesses different files (best case for sharding)
    let pool_opts = BufferPoolOptions {
        capacity_bytes: 128 * 1024 * 1024,
        frame_size: 16 * 1024,
        num_shards: 16,
    };
    let pool = BufferPool::new(pool_opts);

    let pages_per_thread = 100;

    for num_threads in [1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("independent_sets", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let pool = pool.clone();
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let pool = pool.clone();
                            thread::spawn(move || {
                                // Each thread accesses its own file range
                                let base_file_id = (thread_id * 100) as u64;

                                for i in 0..100 {
                                    let file_id = base_file_id + (i % pages_per_thread) as u64;
                                    let offset = (i / pages_per_thread) as u64;

                                    let page_id = PageId { file_id, offset };

                                    let frame_ref = pool
                                        .get_page(page_id, |buf| {
                                            buf.fill(0xAB);
                                            Ok::<(), BufferPoolError>(())
                                        })
                                        .unwrap();

                                    // CRITICAL: Drop frame immediately to release pin
                                    black_box(&frame_ref);
                                    drop(frame_ref);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_pool_contention,
    bench_buffer_pool_contention_focused,
    bench_buffer_pool_scalability
);
criterion_main!(benches);
