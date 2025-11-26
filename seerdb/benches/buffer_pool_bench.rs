use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use seerdb::buffer::{BufferPool, BufferPoolOptions};
use seerdb::{SSTable, SSTableBuilder};
use tempfile::tempdir;

// Build an SSTable with specified number of entries and value size
fn build_sstable(num_entries: usize, value_size: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.sst");

    let mut builder = SSTableBuilder::create(&path).unwrap();

    let value_bytes = vec![b'x'; value_size];
    let value = Bytes::from(value_bytes);

    for i in 0..num_entries {
        let key = format!("key_{:010}", i);
        builder.add(Bytes::from(key), value.clone()).unwrap();
    }

    builder.finish().unwrap();
    (dir, path)
}

fn bench_buffer_pool_vs_os_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_pool_vs_os_cache");
    group.sample_size(10); // Reduced sample size for heavy setup

    // Scenario: 50MB data (50k entries * 1KB)
    // Block cache is ~40MB (10k blocks * 4KB)
    // Buffer pool will be configured to 20MB to force eviction
    let num_entries = 50_000;
    let value_size = 1000;

    let (_dir, path) = build_sstable(num_entries, value_size);

    // 1. OS Cache (Standard Path)
    group.bench_function("os_cache", |b| {
        let mut sstable = SSTable::open(&path).unwrap();
        b.iter(|| {
            // Random reads
            // We use a simple LCG for pseudo-randomness to avoid rand dependency overhead in loop
            let mut seed = 12345;
            for _ in 0..100 {
                seed = (1103515245 * seed + 12345) % (1 << 31);
                let idx = seed as usize % num_entries;
                let key = format!("key_{:010}", idx);
                black_box(sstable.get(key.as_bytes()).unwrap());
            }
        });
    });

    // 2. Buffer Pool (LeanStore Path)
    group.bench_function("buffer_pool_20mb", |b| {
        let pool_options = BufferPoolOptions {
            capacity_bytes: 20 * 1024 * 1024, // 20MB
            frame_size: 4096,
            num_shards: 16,
        };
        let pool = BufferPool::new(pool_options);
        let mut sstable = SSTable::open_with_buffer_pool(&path, Some(pool)).unwrap();

        b.iter(|| {
            let mut seed = 12345;
            for _ in 0..100 {
                seed = (1103515245 * seed + 12345) % (1 << 31);
                let idx = seed as usize % num_entries;
                let key = format!("key_{:010}", idx);
                black_box(sstable.get(key.as_bytes()).unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_buffer_pool_vs_os_cache);
criterion_main!(benches);
