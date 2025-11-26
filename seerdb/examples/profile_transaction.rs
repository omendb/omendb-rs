//! Transaction profiling example
//!
//! Measures where time is spent in transaction commit.
//!
//! Run with: cargo run --release --example profile_transaction

use seerdb::{DBOptions, SyncPolicy, DB};
use std::time::{Duration, Instant};
use tempfile::tempdir;

fn main() {
    println!("=== Transaction Commit Profiling ===\n");

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

    // Pre-populate some data
    for i in 0..1000 {
        let key = format!("key_{:06}", i);
        db.put(key.as_bytes(), b"initial_value").unwrap();
    }

    let iterations = 10_000;

    // Profile 1: Write-only transactions (no read-set)
    println!("Profile 1: Write-only transactions (no read-set validation)");
    let start = Instant::now();
    for i in 0..iterations {
        let mut txn = db.begin_transaction();
        let key = format!("new_key_{:06}", i);
        txn.put(key.as_bytes(), b"value").unwrap();
        txn.commit().unwrap();
    }
    let write_only_time = start.elapsed();
    println!(
        "  {} iterations in {:?} ({:.0} txn/sec)\n",
        iterations,
        write_only_time,
        iterations as f64 / write_only_time.as_secs_f64()
    );

    // Profile 2: Read-then-write transactions (1 read, 1 write)
    println!("Profile 2: Read-then-write transactions (1 read in read-set)");
    let start = Instant::now();
    for i in 0..iterations {
        let mut txn = db.begin_transaction();
        let read_key = format!("key_{:06}", i % 1000);
        let _ = txn.get(read_key.as_bytes());
        let write_key = format!("rw_key_{:06}", i);
        txn.put(write_key.as_bytes(), b"value").unwrap();
        txn.commit().unwrap();
    }
    let read_write_time = start.elapsed();
    println!(
        "  {} iterations in {:?} ({:.0} txn/sec)\n",
        iterations,
        read_write_time,
        iterations as f64 / read_write_time.as_secs_f64()
    );

    // Profile 3: Read-heavy transactions (10 reads, 1 write)
    println!("Profile 3: Read-heavy transactions (10 reads in read-set)");
    let start = Instant::now();
    for i in 0..iterations {
        let mut txn = db.begin_transaction();
        for j in 0..10 {
            let read_key = format!("key_{:06}", (i * 10 + j) % 1000);
            let _ = txn.get(read_key.as_bytes());
        }
        let write_key = format!("rh_key_{:06}", i);
        txn.put(write_key.as_bytes(), b"value").unwrap();
        txn.commit().unwrap();
    }
    let read_heavy_time = start.elapsed();
    println!(
        "  {} iterations in {:?} ({:.0} txn/sec)\n",
        iterations,
        read_heavy_time,
        iterations as f64 / read_heavy_time.as_secs_f64()
    );

    // Profile 4: Batch writes (10 writes per transaction)
    println!("Profile 4: Batch transactions (10 writes per txn)");
    let start = Instant::now();
    for i in 0..(iterations / 10) {
        let mut txn = db.begin_transaction();
        for j in 0..10 {
            let key = format!("batch_key_{:06}_{}", i, j);
            txn.put(key.as_bytes(), b"value").unwrap();
        }
        txn.commit().unwrap();
    }
    let batch_time = start.elapsed();
    println!(
        "  {} txns ({} writes) in {:?} ({:.0} writes/sec)\n",
        iterations / 10,
        iterations,
        batch_time,
        iterations as f64 / batch_time.as_secs_f64()
    );

    // Profile 5: Compare with raw puts
    println!("Profile 5: Raw puts (no transaction)");
    let start = Instant::now();
    for i in 0..iterations {
        let key = format!("raw_key_{:06}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }
    let raw_time = start.elapsed();
    println!(
        "  {} iterations in {:?} ({:.0} ops/sec)\n",
        iterations,
        raw_time,
        iterations as f64 / raw_time.as_secs_f64()
    );

    // Summary
    println!("=== Summary ===");
    println!(
        "Write-only txn:    {:.0} txn/sec",
        iterations as f64 / write_only_time.as_secs_f64()
    );
    println!(
        "Read-write txn:    {:.0} txn/sec ({:.1}x slower)",
        iterations as f64 / read_write_time.as_secs_f64(),
        read_write_time.as_secs_f64() / write_only_time.as_secs_f64()
    );
    println!(
        "Read-heavy txn:    {:.0} txn/sec ({:.1}x slower)",
        iterations as f64 / read_heavy_time.as_secs_f64(),
        read_heavy_time.as_secs_f64() / write_only_time.as_secs_f64()
    );
    println!(
        "Batch txn (10):    {:.0} writes/sec ({:.1}x faster)",
        iterations as f64 / batch_time.as_secs_f64(),
        write_only_time.as_secs_f64() / batch_time.as_secs_f64()
    );
    println!(
        "Raw put:           {:.0} ops/sec",
        iterations as f64 / raw_time.as_secs_f64()
    );
    println!(
        "\nTransaction overhead vs raw: {:.1}%",
        (write_only_time.as_secs_f64() / raw_time.as_secs_f64() - 1.0) * 100.0
    );
}
