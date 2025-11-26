// Baseline benchmark comparing RocksDB, sled, and fjall
// Tests: write throughput, read throughput, mixed workload, scan performance
//
// To run this benchmark, use:
// cargo run --example baseline_benchmark --features baseline-benchmarks

#[cfg(feature = "baseline-benchmarks")]
use std::time::Instant;

#[cfg(feature = "baseline-benchmarks")]
const NUM_OPERATIONS: usize = 100_000;
#[cfg(feature = "baseline-benchmarks")]
const VALUE_SIZE: usize = 1024; // 1KB values (typical for many workloads)

#[cfg(feature = "baseline-benchmarks")]
fn main() {
    println!("=== Storage Engine Baseline Benchmark ===\n");
    println!("Operations: {}", NUM_OPERATIONS);
    println!("Value size: {} bytes\n", VALUE_SIZE);

    // Clean up any existing test databases
    let _ = std::fs::remove_dir_all("/tmp/bench_rocksdb");
    let _ = std::fs::remove_dir_all("/tmp/bench_sled");
    let _ = std::fs::remove_dir_all("/tmp/bench_fjall");

    println!("{:=<70}", "=");
    println!("ROCKSDB");
    println!("{:=<70}\n", "=");
    benchmark_rocksdb();

    println!("\n{:=<70}", "=");
    println!("SLED");
    println!("{:=<70}\n", "=");
    benchmark_sled();

    println!("\n{:=<70}", "=");
    println!("FJALL");
    println!("{:=<70}\n", "=");
    benchmark_fjall();

    println!("\n{:=<70}", "=");
    println!("SEERDB (THIS PROJECT)");
    println!("{:=<70}\n", "=");
    benchmark_seerdb();

    println!("\n{:=<70}", "=");
    println!("Summary");
    println!("{:=<70}", "=");
    println!("All benchmarks complete. See results above.");
    println!("Key metrics: throughput (ops/sec), latency (us), write amplification");
}

fn benchmark_rocksdb() {
    use rocksdb::{Options, DB};

    let path = "/tmp/bench_rocksdb";
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024); // 64MB memtable

    let db = DB::open(&opts, path).expect("Failed to open RocksDB");

    // Workload 1: Sequential Writes
    println!("Workload 1: Sequential Writes ({} ops)", NUM_OPERATIONS);
    let value = vec![0u8; VALUE_SIZE];
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), &value).expect("Put failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 2: Random Reads
    println!("\nWorkload 2: Random Reads ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        let _ = db.get(key.as_bytes()).expect("Get failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 3: Mixed (50% read, 50% write)
    println!("\nWorkload 3: Mixed 50/50 ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        if i % 2 == 0 {
            // Write
            let key = format!("key_{:08}", i + NUM_OPERATIONS);
            db.put(key.as_bytes(), &value).expect("Put failed");
        } else {
            // Read
            let key = format!("key_{:08}", i);
            let _ = db.get(key.as_bytes()).expect("Get failed");
        }
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 4: Range Scans
    println!("\nWorkload 4: Range Scans (1000 scans, 100 keys each)");
    let start = Instant::now();
    for i in 0..1000 {
        let start_key = format!("key_{:08}", i * 100);
        let mut iter = db.raw_iterator();
        iter.seek(start_key.as_bytes());
        let mut _count = 0;
        while iter.valid() && _count < 100 {
            let _ = iter.key();
            let _ = iter.value();
            iter.next();
            _count += 1;
        }
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} scans/sec",
        1000.0 / elapsed.as_secs_f64()
    );
    println!(
        "  Latency: {:.2} ms/scan",
        elapsed.as_millis() as f64 / 1000.0
    );

    drop(db);
}

fn benchmark_sled() {
    let path = "/tmp/bench_sled";
    let db = sled::open(path).expect("Failed to open sled");

    // Workload 1: Sequential Writes
    println!("Workload 1: Sequential Writes ({} ops)", NUM_OPERATIONS);
    let value = vec![0u8; VALUE_SIZE];
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        db.insert(key.as_bytes(), value.as_slice())
            .expect("Insert failed");
    }
    db.flush().expect("Flush failed");
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 2: Random Reads
    println!("\nWorkload 2: Random Reads ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        let _ = db.get(key.as_bytes()).expect("Get failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 3: Mixed (50% read, 50% write)
    println!("\nWorkload 3: Mixed 50/50 ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        if i % 2 == 0 {
            // Write
            let key = format!("key_{:08}", i + NUM_OPERATIONS);
            db.insert(key.as_bytes(), value.as_slice())
                .expect("Insert failed");
        } else {
            // Read
            let key = format!("key_{:08}", i);
            let _ = db.get(key.as_bytes()).expect("Get failed");
        }
    }
    db.flush().expect("Flush failed");
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 4: Range Scans
    println!("\nWorkload 4: Range Scans (1000 scans, 100 keys each)");
    let start = Instant::now();
    for i in 0..1000 {
        let start_key = format!("key_{:08}", i * 100);
        let iter = db.range(start_key.as_bytes()..);
        for _ in iter.take(100) {
            // Iterate through items
        }
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} scans/sec",
        1000.0 / elapsed.as_secs_f64()
    );
    println!(
        "  Latency: {:.2} ms/scan",
        elapsed.as_millis() as f64 / 1000.0
    );

    drop(db);
}

fn benchmark_fjall() {
    let path = "/tmp/bench_fjall";
    let keyspace = fjall::Config::new(path)
        .open()
        .expect("Failed to open fjall");
    let partition = keyspace
        .open_partition("default", Default::default())
        .expect("Failed to open partition");

    // Workload 1: Sequential Writes (using Batch for performance)
    println!("Workload 1: Sequential Writes ({} ops)", NUM_OPERATIONS);
    let value = vec![0u8; VALUE_SIZE];
    let start = Instant::now();

    // Use batches of 10k items to avoid excessive memory usage
    let batch_size = 10_000;
    for batch_start in (0..NUM_OPERATIONS).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(NUM_OPERATIONS);
        let mut batch = keyspace.batch();

        for i in batch_start..batch_end {
            let key = format!("key_{:08}", i);
            batch.insert(&partition, key.as_bytes(), &value);
        }

        batch.commit().expect("Batch commit failed");
    }

    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 2: Random Reads
    println!("\nWorkload 2: Random Reads ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        let _ = partition.get(key.as_bytes()).expect("Get failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 3: Mixed (50% read, 50% write) - batch writes, individual reads
    println!("\nWorkload 3: Mixed 50/50 ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();

    // Collect write operations, execute reads immediately
    let mut batch = keyspace.batch();
    for i in 0..NUM_OPERATIONS {
        if i % 2 == 0 {
            // Write (batched)
            let key = format!("key_{:08}", i + NUM_OPERATIONS);
            batch.insert(&partition, key.as_bytes(), &value);
        } else {
            // Read (immediate)
            let key = format!("key_{:08}", i);
            let _ = partition.get(key.as_bytes()).expect("Get failed");
        }
    }
    batch.commit().expect("Batch commit failed");

    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 4: Range Scans
    println!("\nWorkload 4: Range Scans (1000 scans, 100 keys each)");
    let start = Instant::now();
    for i in 0..1000 {
        let start_key = format!("key_{:08}", i * 100);
        let iter = partition.range(start_key.as_bytes()..);
        for _ in iter.take(100) {
            // Iterate through items
        }
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} scans/sec",
        1000.0 / elapsed.as_secs_f64()
    );
    println!(
        "  Latency: {:.2} ms/scan",
        elapsed.as_millis() as f64 / 1000.0
    );

    drop(partition);
    drop(keyspace);
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_seerdb() {
    use seerdb::{DBOptions, DB};
    use std::path::PathBuf;

    let path = PathBuf::from("/tmp/bench_seerdb");
    let _ = std::fs::remove_dir_all(&path);

    let opts = DBOptions {
        data_dir: path.clone(),
        memtable_capacity: 64 * 1024 * 1024, // 64MB memtable (same as RocksDB)
        wal_sync_policy: seerdb::SyncPolicy::None, // Fast benchmark mode
        background_compaction: true,
        vlog_threshold: Some(4096), // Enable vLog for values >4KB
        ..Default::default()
    };

    let db = DB::open(opts).expect("Failed to open seerdb");

    // Workload 1: Sequential Writes
    println!("Workload 1: Sequential Writes ({} ops)", NUM_OPERATIONS);
    let value = vec![0u8; VALUE_SIZE];
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), &value).expect("Put failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 2: Random Reads
    println!("\nWorkload 2: Random Reads ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        let _ = db.get(key.as_bytes()).expect("Get failed");
    }
    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 3: Mixed (50% read, 50% write) - using batches for fair comparison with fjall
    println!("\nWorkload 3: Mixed 50/50 ({} ops)", NUM_OPERATIONS);
    let start = Instant::now();

    // Batch writes like fjall does (fair comparison)
    let mut batch = db.batch();
    for i in 0..NUM_OPERATIONS {
        if i % 2 == 0 {
            // Write (batched)
            let key = format!("key_{:08}", i + NUM_OPERATIONS);
            batch.put(key.as_bytes(), &value);
        } else {
            // Read (immediate)
            let key = format!("key_{:08}", i);
            let _ = db.get(key.as_bytes()).expect("Get failed");
        }
    }
    batch.commit().expect("Batch commit failed");

    let elapsed = start.elapsed();
    let throughput = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!(
        "  Latency: {:.2} us/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );

    // Workload 4: Range Scans (efficient iterator-based)
    println!("\nWorkload 4: Range Scans (1000 scans, 100 keys each)");
    let start = Instant::now();
    for i in 0..1000 {
        let start_key = format!("key_{:08}", i * 100);
        let end_key = format!("key_{:08}", i * 100 + 100);
        let mut count = 0;
        for result in db
            .range(start_key.as_bytes(), Some(end_key.as_bytes()))
            .unwrap()
        {
            let (_key, _value) = result.unwrap();
            count += 1;
            if count >= 100 {
                break;
            }
        }
    }
    let elapsed = start.elapsed();
    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.0} scans/sec",
        1000.0 / elapsed.as_secs_f64()
    );
    println!(
        "  Latency: {:.2} ms/scan",
        elapsed.as_millis() as f64 / 1000.0
    );

    drop(db);
}

#[cfg(not(feature = "baseline-benchmarks"))]
fn main() {
    eprintln!("Error: This benchmark requires the 'baseline-benchmarks' feature.");
    eprintln!("Run with: cargo run --example baseline_benchmark --features baseline-benchmarks");
    std::process::exit(1);
}
