// Benchmark to measure impact of micro-optimizations on read performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::{DBOptions, DB};
use std::sync::Arc;
use tempfile::TempDir;

fn setup_db(num_keys: usize) -> (Arc<DB>, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 4 * 1024 * 1024,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Write test data
    for i in 0..num_keys {
        let key = format!("key_{:08}", i);
        let value = format!("value_{:08}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    db.flush().unwrap();
    (db, temp_dir)
}

fn bench_random_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_reads");

    let (db, _temp) = setup_db(10_000);

    group.bench_function("random_read_hot_cache", |b| {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        b.iter(|| {
            let key_id = rng.gen_range(0..10_000);
            let key = format!("key_{:08}", key_id);
            black_box(db.get(key.as_bytes()).unwrap())
        });
    });

    group.finish();
}

fn bench_sequential_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_reads");

    let (db, _temp) = setup_db(10_000);

    group.bench_function("sequential_read", |b| {
        let mut idx = 0;
        b.iter(|| {
            let key = format!("key_{:08}", idx % 10_000);
            idx += 1;
            black_box(db.get(key.as_bytes()).unwrap())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_random_reads, bench_sequential_reads);
criterion_main!(benches);
