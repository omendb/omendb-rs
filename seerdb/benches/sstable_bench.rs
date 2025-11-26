// SSTable performance benchmark
// Measures binary search and bloom filter improvements

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::{SSTable, SSTableBuilder};
use tempfile::tempdir;

fn build_sstable(num_entries: usize) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.sst");

    let mut builder = SSTableBuilder::create(&path).unwrap();

    for i in 0..num_entries {
        let key = format!("key_{:010}", i);
        let value = format!("value_{:010}", i);
        let _ = builder.add(Bytes::from(key), Bytes::from(value));
    }

    builder.finish().unwrap();
    (dir, path)
}

fn bench_point_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_point_lookups");

    for size in [1000, 10000, 100000] {
        let (_dir, path) = build_sstable(size);

        group.bench_with_input(
            BenchmarkId::new("existing_keys", size),
            &size,
            |b, &size| {
                let mut sstable = SSTable::open(&path).unwrap();
                b.iter(|| {
                    // Lookup random existing keys
                    for i in (0..size).step_by(100) {
                        let key = format!("key_{:010}", i);
                        black_box(sstable.get(key.as_bytes()).unwrap());
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("missing_keys", size),
            &size,
            |b, &_size| {
                let mut sstable = SSTable::open(&path).unwrap();
                b.iter(|| {
                    // Lookup keys that don't exist (bloom filter should help)
                    for i in 0..100 {
                        let key = format!("missing_{:010}", i);
                        black_box(sstable.get(key.as_bytes()).unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_sequential_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("sstable_scan");

    for size in [1000, 10000] {
        let (_dir, path) = build_sstable(size);

        group.bench_with_input(BenchmarkId::new("full_scan", size), &size, |b, &_size| {
            b.iter(|| {
                let mut sstable = SSTable::open(&path).unwrap();
                let mut iter = sstable.iter().unwrap();
                let mut count = 0;
                while let Some(Ok(_entry)) = iter.next() {
                    count += 1;
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_point_lookups, bench_sequential_scan);
criterion_main!(benches);
