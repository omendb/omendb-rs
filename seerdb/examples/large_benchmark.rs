// Large workload benchmark to test background flush effectiveness
use seerdb::{DBOptions, SyncPolicy, DB};
use std::time::Instant;
use tempfile::TempDir;

const NUM_OPERATIONS: usize = 1_000_000; // 1M ops = ~1GB
const VALUE_SIZE: usize = 1024;

fn bench_writes(name: &str, background_flush: bool) {
    println!("=== {} ===", name);

    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 128 * 1024 * 1024, // 128MB (triggers ~8 flushes for 1GB)
        background_flush,
        background_compaction: true,
        wal_sync_policy: SyncPolicy::None,
        vlog_threshold: Some(4096),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();
    let value = vec![0u8; VALUE_SIZE];

    println!("Writing {} operations...", NUM_OPERATIONS);
    let start = Instant::now();

    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), &value).unwrap();

        if i > 0 && i % 100_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            println!("  {:7} ops: {:.0} ops/sec", i, rate);
        }
    }

    // Final flush to ensure all data is written
    db.flush().unwrap();

    let elapsed = start.elapsed();
    let ops_per_sec = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
    println!(
        "  Latency: {:.2} µs/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );
    println!();
}

fn bench_mixed(name: &str, background_flush: bool) {
    println!("=== {} (Mixed 50/50) ===", name);

    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        memtable_capacity: 128 * 1024 * 1024,
        background_flush,
        background_compaction: true,
        wal_sync_policy: SyncPolicy::None,
        vlog_threshold: Some(4096),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();
    let value = vec![0u8; VALUE_SIZE];

    // Pre-populate
    println!("Pre-populating with {} entries...", NUM_OPERATIONS);
    for i in 0..NUM_OPERATIONS {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), &value).unwrap();
    }

    // Mixed workload
    println!("Running mixed workload ({} ops)...", NUM_OPERATIONS);
    let start = Instant::now();

    for i in 0..NUM_OPERATIONS {
        if i % 2 == 0 {
            // Read
            let key = format!("key_{:08}", i);
            let _ = db.get(key.as_bytes()).unwrap();
        } else {
            // Write
            let key = format!("key_{:08}", i);
            db.put(key.as_bytes(), &value).unwrap();
        }

        if i > 0 && i % 100_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            println!("  {:7} ops: {:.0} ops/sec", i, rate);
        }
    }

    let elapsed = start.elapsed();
    let ops_per_sec = NUM_OPERATIONS as f64 / elapsed.as_secs_f64();

    println!("\nResults:");
    println!("  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
    println!(
        "  Latency: {:.2} µs/op",
        elapsed.as_micros() as f64 / NUM_OPERATIONS as f64
    );
    println!();
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Large Workload Benchmark (1M ops = 1GB)                  ║");
    println!("║  Testing background flush effectiveness                    ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Test 1: Writes without background flush
    bench_writes("Write Workload: Baseline (no background flush)", false);

    // Test 2: Writes with background flush
    bench_writes("Write Workload: With background flush", true);

    // Test 3: Mixed without background flush
    bench_mixed("Mixed Workload: Baseline (no background flush)", false);

    // Test 4: Mixed with background flush
    bench_mixed("Mixed Workload: With background flush", true);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Analysis                                                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("If background flush is working correctly:");
    println!("  - Writes with background flush should be 10-20%% faster");
    println!("  - Mixed with background flush should be 40-60%% faster");
    println!("\nReason: Large dataset (1GB) triggers ~8 flushes, eliminating");
    println!("flush blocking should significantly improve throughput.");
}
