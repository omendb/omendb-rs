use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use seerdb::{DBOptions, SyncPolicy, DB};
use tempfile::tempdir;

fn bench_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("recovery");

    // Define workload sizes (number of keys)
    // 10k keys * 100 bytes = 1MB WAL
    // 100k keys * 100 bytes = 10MB WAL
    // 1M keys * 100 bytes = 100MB WAL
    let sizes = [10_000, 100_000, 1_000_000];

    for &num_keys in &sizes {
        group.throughput(Throughput::Elements(num_keys as u64));

        // Prepare "golden" WAL
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("setup_db");
        let wal_path = db_path.join("wal.log");

        // 1. Create DB and fill WAL
        {
            let opts = DBOptions {
                data_dir: db_path.clone(),
                memtable_capacity: 1024 * 1024 * 1024, // 1GB (prevent flush)
                wal_sync_policy: SyncPolicy::None,     // Fast setup (no fsync)
                ..Default::default()
            };
            let db = DB::open(opts).unwrap();

            for i in 0..num_keys {
                let key = format!("key_{:09}", i);
                let value = format!("value_{:09}", i); // ~20 bytes total per record overhead
                db.put(key.as_bytes(), value.as_bytes()).unwrap();
            }
            // DB drop will NOT flush memtable (unless explicit flush called or overflow)
            // WAL remains populated.
        }

        // Verify WAL exists
        assert!(wal_path.exists());
        let wal_size = std::fs::metadata(&wal_path).unwrap().len();
        println!(
            "Prepared WAL size for {} keys: {} bytes",
            num_keys, wal_size
        );

        group.bench_with_input(
            criterion::BenchmarkId::new("wal_replay", num_keys),
            &num_keys,
            |b, &_| {
                b.iter_batched(
                    || {
                        // Setup: Create a fresh directory for this iteration
                        let run_dir = tempdir().unwrap();
                        let run_db_path = run_dir.path().join("run_db");
                        std::fs::create_dir_all(&run_db_path).unwrap();

                        // Copy the populated WAL to the run directory
                        std::fs::copy(&wal_path, run_db_path.join("wal.log")).unwrap();

                        (run_dir, run_db_path)
                    },
                    |(_run_dir, run_db_path)| {
                        // Measure: Open DB (triggers recovery)
                        let opts = DBOptions {
                            data_dir: run_db_path,
                            memtable_capacity: 1024 * 1024 * 1024,
                            ..Default::default()
                        };
                        black_box(DB::open(opts).unwrap());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_recovery);
criterion_main!(benches);
