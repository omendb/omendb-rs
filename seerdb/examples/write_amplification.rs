// Measures write amplification with and without vLog (WiscKey value separation)
// Write amplification = (Bytes written to disk) / (Logical bytes written by user)

use seerdb::{DBOptions, DB};
use std::path::PathBuf;

fn main() {
    println!("=== Write Amplification Benchmark ===\n");

    // Configuration
    const NUM_OPERATIONS: usize = 100_000;
    const VALUE_SIZE: usize = 8192; // 8KB values (larger than vLog threshold)
    const VLOG_THRESHOLD: usize = 4096; // 4KB threshold for vLog

    // Benchmark 1: WITHOUT vLog (traditional LSM)
    println!("Benchmark 1: Traditional LSM (no value separation)");
    println!("-------------------------------------------------");
    let (write_amp_no_vlog, _) = run_benchmark(NUM_OPERATIONS, VALUE_SIZE, None);
    println!("Write Amplification: {:.2}x\n", write_amp_no_vlog);

    // Benchmark 2: WITH vLog (WiscKey value separation)
    println!(
        "Benchmark 2: WiscKey vLog (value separation, threshold={}KB)",
        VLOG_THRESHOLD / 1024
    );
    println!("-------------------------------------------------");
    let (write_amp_with_vlog, _) = run_benchmark(NUM_OPERATIONS, VALUE_SIZE, Some(VLOG_THRESHOLD));
    println!("Write Amplification: {:.2}x\n", write_amp_with_vlog);

    // Comparison
    println!("=== Results Summary ===");
    println!("Traditional LSM:       {:.2}x", write_amp_no_vlog);
    println!("WiscKey vLog:          {:.2}x", write_amp_with_vlog);
    if write_amp_no_vlog > write_amp_with_vlog {
        let improvement = write_amp_no_vlog / write_amp_with_vlog;
        println!(
            "\nImprovement:           {:.2}x better with vLog",
            improvement
        );
        println!("Target:                10x better (from paper)");
        if improvement >= 5.0 {
            println!("Status:                ✅ Significant improvement achieved");
        } else if improvement >= 2.0 {
            println!("Status:                ⚠️  Moderate improvement (less than claimed)");
        } else {
            println!("Status:                ❌ Minimal improvement");
        }
    }
}

fn run_benchmark(num_ops: usize, value_size: usize, vlog_threshold: Option<usize>) -> (f64, u64) {
    // Create temporary database
    let temp_dir = format!(
        "/tmp/seerdb_writeamp_{}",
        vlog_threshold
            .map(|t| format!("vlog{}", t))
            .unwrap_or_else(|| "novlog".to_string())
    );
    let path = PathBuf::from(&temp_dir);
    let _ = std::fs::remove_dir_all(&path);

    let opts = DBOptions {
        data_dir: path.clone(),
        memtable_capacity: 4 * 1024 * 1024, // 4MB memtable (smaller to trigger more flushes/compactions)
        wal_sync_policy: seerdb::SyncPolicy::None, // Fast benchmark mode
        background_compaction: false,       // Synchronous compaction for deterministic results
        vlog_threshold,
        ..Default::default()
    };

    let db = DB::open(opts).expect("Failed to open database");

    // Write sequential keys with fixed-size values
    let value = "x".repeat(value_size);
    for i in 0..num_ops {
        let key = format!("key_{:08}", i);
        db.put(key.as_bytes(), value.as_bytes())
            .expect("Put failed");
    }

    // Force final flush to ensure all data is written
    db.flush().expect("Flush failed");

    // Get stats
    let stats = db.stats();

    // Calculate write amplification
    let logical_bytes = stats.logical_bytes_written;
    let physical_bytes = stats.physical_bytes_written;
    let write_amp = if logical_bytes > 0 {
        physical_bytes as f64 / logical_bytes as f64
    } else {
        0.0
    };

    // Print details
    println!("  Operations:          {}", num_ops);
    println!("  Value size:          {} bytes", value_size);
    println!(
        "  Logical bytes:       {} MB ({} bytes)",
        logical_bytes / 1_000_000,
        logical_bytes
    );
    println!(
        "  Physical bytes:      {} MB ({} bytes)",
        physical_bytes / 1_000_000,
        physical_bytes
    );
    println!("  Total SSTables:      {}", stats.total_sstables);
    println!("  Total flushes:       {}", stats.total_flushes);
    println!("  Total compactions:   {}", stats.total_compactions);
    println!(
        "  Disk usage:          {} MB",
        stats.total_disk_bytes / 1_000_000
    );

    // Cleanup
    drop(db);
    let _ = std::fs::remove_dir_all(&path);

    (write_amp, physical_bytes)
}
