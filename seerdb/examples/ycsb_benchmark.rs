// YCSB-style workload benchmarks for real-world performance validation
// Tests common database workload patterns: read-heavy, write-heavy, read-only, read-latest

use rand::Rng;
use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use std::time::Instant;

const NUM_RECORDS: usize = 100_000;
const NUM_OPERATIONS: usize = 100_000;
const VALUE_SIZE: usize = 1024; // 1KB values

fn main() {
    println!("=== YCSB Workload Benchmark ===");
    println!("Records: {}", NUM_RECORDS);
    println!("Operations: {}", NUM_OPERATIONS);
    println!("Value size: {} bytes\n", VALUE_SIZE);

    // Workload A: 50% read, 50% update (update heavy)
    println!("Workload A: Update Heavy (50% read, 50% update)");
    println!("-------------------------------------------------");
    run_workload_a();
    println!();

    // Workload B: 95% read, 5% update (read mostly)
    println!("Workload B: Read Mostly (95% read, 5% update)");
    println!("-------------------------------------------------");
    run_workload_b();
    println!();

    // Workload C: 100% read (read only)
    println!("Workload C: Read Only (100% read)");
    println!("-------------------------------------------------");
    run_workload_c();
    println!();

    // Workload D: 95% read latest, 5% insert (read recent data)
    println!("Workload D: Read Latest (95% read latest, 5% insert)");
    println!("-------------------------------------------------");
    run_workload_d();
    println!();

    println!("=== Benchmark Complete ===");
}

fn run_workload_a() {
    let (db, _path) = setup_db("ycsb_a");
    load_initial_data(&db);

    let start = Instant::now();
    let mut rng = rand::thread_rng();

    for _ in 0..NUM_OPERATIONS {
        let key_num = rng.gen_range(0..NUM_RECORDS);
        let key = format!("user{:08}", key_num);

        if rng.gen_bool(0.5) {
            // 50% read
            let _ = db.get(key.as_bytes());
        } else {
            // 50% update
            let value = generate_value(VALUE_SIZE);
            let _ = db.put(key.as_bytes(), value.as_bytes());
        }
    }

    let elapsed = start.elapsed();
    print_results(elapsed, &db);
    cleanup(db, _path);
}

fn run_workload_b() {
    let (db, _path) = setup_db("ycsb_b");
    load_initial_data(&db);

    let start = Instant::now();
    let mut rng = rand::thread_rng();

    for _ in 0..NUM_OPERATIONS {
        let key_num = rng.gen_range(0..NUM_RECORDS);
        let key = format!("user{:08}", key_num);

        if rng.gen_bool(0.95) {
            // 95% read
            let _ = db.get(key.as_bytes());
        } else {
            // 5% update
            let value = generate_value(VALUE_SIZE);
            let _ = db.put(key.as_bytes(), value.as_bytes());
        }
    }

    let elapsed = start.elapsed();
    print_results(elapsed, &db);
    cleanup(db, _path);
}

fn run_workload_c() {
    let (db, _path) = setup_db("ycsb_c");
    load_initial_data(&db);

    let start = Instant::now();
    let mut rng = rand::thread_rng();

    for _ in 0..NUM_OPERATIONS {
        let key_num = rng.gen_range(0..NUM_RECORDS);
        let key = format!("user{:08}", key_num);
        let _ = db.get(key.as_bytes()); // 100% read
    }

    let elapsed = start.elapsed();
    print_results(elapsed, &db);
    cleanup(db, _path);
}

fn run_workload_d() {
    let (db, _path) = setup_db("ycsb_d");
    load_initial_data(&db);

    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut next_key = NUM_RECORDS;

    for _ in 0..NUM_OPERATIONS {
        if rng.gen_bool(0.95) {
            // 95% read latest (recent keys have higher probability)
            // Zipfian distribution - but simplified to recent bias
            let key_num = rng.gen_range((next_key.saturating_sub(10000))..next_key);
            let key = format!("user{:08}", key_num);
            let _ = db.get(key.as_bytes());
        } else {
            // 5% insert new records
            let key = format!("user{:08}", next_key);
            let value = generate_value(VALUE_SIZE);
            let _ = db.put(key.as_bytes(), value.as_bytes());
            next_key += 1;
        }
    }

    let elapsed = start.elapsed();
    print_results(elapsed, &db);
    cleanup(db, _path);
}

fn setup_db(name: &str) -> (DB, PathBuf) {
    let path = PathBuf::from(format!("/tmp/seerdb_{}", name));
    let _ = std::fs::remove_dir_all(&path);

    let opts = DBOptions {
        data_dir: path.clone(),
        memtable_capacity: 64 * 1024 * 1024, // 64MB
        wal_sync_policy: seerdb::SyncPolicy::None,
        background_compaction: true,
        vlog_threshold: Some(4096), // Enable vLog for large values
        ..Default::default()
    };

    let db = DB::open(opts).expect("Failed to open database");
    (db, path)
}

fn load_initial_data(db: &DB) {
    let value = generate_value(VALUE_SIZE);
    for i in 0..NUM_RECORDS {
        let key = format!("user{:08}", i);
        db.put(key.as_bytes(), value.as_bytes())
            .expect("Failed to load data");
    }
    db.flush().expect("Failed to flush");
}

fn generate_value(size: usize) -> String {
    "x".repeat(size)
}

fn print_results(elapsed: std::time::Duration, db: &DB) {
    let ops_per_sec = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / NUM_OPERATIONS as f64;

    println!("  Time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
    println!("  Latency: {:.2} µs/op", latency_us);

    let stats = db.stats();
    println!("  Total SSTables: {}", stats.total_sstables);
    println!("  Write Amp: {:.2}x", stats.write_amplification);
}

fn cleanup(db: DB, path: PathBuf) {
    drop(db);
    let _ = std::fs::remove_dir_all(&path);
}
