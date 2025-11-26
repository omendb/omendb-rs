// Demonstration of seerdb observability features
// Shows metrics collection, structured logging, and health monitoring

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use tempfile::tempdir;

fn main() {
    // Enable structured logging (JSON format for production)
    // Set RUST_LOG=info for normal operation, RUST_LOG=debug for verbose
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    println!("=== seerdb Observability Demo ===\n");

    let temp_dir = tempdir().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    println!("1. Opening database with observability enabled...");
    let opts = DBOptions {
        data_dir: data_dir.clone(),
        memtable_capacity: 1024 * 1024, // 1MB memtable (triggers frequent flushes)
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();
    println!("   ✅ Database opened\n");

    // === METRICS DEMO ===
    println!("2. Performing operations to generate metrics...");

    // Write operations
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'v'; 100]; // 100 bytes
        db.put(key.as_bytes(), &value).unwrap();
    }
    println!("   ✅ 1000 PUT operations completed");

    // Read operations
    for i in 0..500 {
        let key = format!("key_{:04}", i);
        let _ = db.get(key.as_bytes()).unwrap();
    }
    println!("   ✅ 500 GET operations completed");

    // Delete operations
    for i in 0..100 {
        let key = format!("key_{:04}", i);
        db.delete(key.as_bytes()).unwrap();
    }
    println!("   ✅ 100 DELETE operations completed");

    // Flush to trigger SSTable creation
    db.flush().unwrap();
    println!("   ✅ FLUSH completed\n");

    // === METRICS COLLECTION ===
    println!("3. Collecting metrics...");
    let stats = db.stats();

    println!("\n   📊 Operation Counts:");
    println!("      Total PUTs:        {}", stats.total_puts);
    println!("      Total GETs:        {}", stats.total_gets);
    println!("      Total DELETEs:     {}", stats.total_deletes);
    println!("      Total FLUSHes:     {}", stats.total_flushes);
    println!("      Total COMPACTIONs: {}", stats.total_compactions);

    println!("\n   ⚡ Throughput:");
    println!("      Writes/sec:  {:.2}", stats.writes_per_sec);
    println!("      Reads/sec:   {:.2}", stats.reads_per_sec);
    println!("      Deletes/sec: {:.2}", stats.deletes_per_sec);

    println!("\n   ⏱️  Latency (PUT):");
    println!("      p50:  {} µs", stats.put_latency_p50_us);
    println!("      p95:  {} µs", stats.put_latency_p95_us);
    println!("      p99:  {} µs", stats.put_latency_p99_us);
    println!("      p999: {} µs", stats.put_latency_p999_us);

    println!("\n   ⏱️  Latency (GET):");
    println!("      p50:  {} µs", stats.get_latency_p50_us);
    println!("      p95:  {} µs", stats.get_latency_p95_us);
    println!("      p99:  {} µs", stats.get_latency_p99_us);
    println!("      p999: {} µs", stats.get_latency_p999_us);

    println!("\n   💾 Storage:");
    println!(
        "      Memtable size:    {} KB ({:.1}% of capacity)",
        stats.memtable_size_bytes / 1024,
        stats.memtable_utilization_pct
    );
    println!("      WAL size:         {} KB", stats.wal_size_bytes / 1024);
    println!(
        "      Total disk usage: {} KB",
        stats.total_disk_bytes / 1024
    );
    println!("      Total SSTables:   {}", stats.total_sstables);

    println!("\n   📈 LSM Tree Structure:");
    for (level, count) in stats.sstables_per_level.iter().enumerate() {
        if *count > 0 {
            println!(
                "      Level {}: {} SSTables ({} KB)",
                level,
                count,
                stats.level_sizes_bytes.get(level).unwrap_or(&0) / 1024
            );
        }
    }

    println!("\n   🕐 Uptime: {} seconds", stats.uptime_seconds);

    // === HEALTH CHECKS ===
    println!("\n4. Running health checks...");
    let health = db.health();

    println!("\n   {}", health);

    if health.healthy {
        println!("   ✅ Overall Status: HEALTHY");
    } else if health.has_unhealthy() {
        println!("   ❌ Overall Status: UNHEALTHY");
    } else if health.has_degraded() {
        println!("   ⚠️  Overall Status: DEGRADED");
    }

    // === PRODUCTION MONITORING TIPS ===
    println!("\n5. Production Monitoring Tips:");
    println!("\n   📊 Metrics to Monitor:");
    println!("      • Throughput (writes_per_sec, reads_per_sec)");
    println!("      • Latency percentiles (p99, p999)");
    println!("      • Memtable utilization (should be < 90%)");
    println!("      • Disk usage growth rate");
    println!("      • Compaction frequency");

    println!("\n   🚨 Alert Thresholds:");
    println!("      • p99 latency > 100ms → investigate");
    println!("      • Memtable utilization > 90% → increase capacity");
    println!("      • Disk usage > 80% → add storage");
    println!("      • Health status = Unhealthy → immediate action");

    println!("\n   📝 Logging:");
    println!("      • Set RUST_LOG=info for production");
    println!("      • Use RUST_LOG=debug for troubleshooting");
    println!("      • Export to JSON for log aggregation");
    println!("      • Example: RUST_LOG=seerdb=info cargo run --example observability_demo");

    println!("\n   💡 Integration:");
    println!("      • Export stats to Prometheus");
    println!("      • Send logs to Elasticsearch/Loki");
    println!("      • Alert on health status changes");
    println!("      • Dashboard: Grafana/Datadog");

    println!("\n6. Simulating degraded state...");
    // Write more data to increase memtable utilization
    for i in 1000..5000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'v'; 100];
        db.put(key.as_bytes(), &value).unwrap();
    }

    let stats2 = db.stats();
    println!(
        "   Memtable utilization increased to {:.1}%",
        stats2.memtable_utilization_pct
    );

    let health2 = db.health();
    println!("\n   Updated health status:");
    for check in &health2.checks {
        let icon = match check.status {
            seerdb::CheckStatus::Healthy => "✅",
            seerdb::CheckStatus::Degraded => "⚠️",
            seerdb::CheckStatus::Unhealthy => "❌",
        };
        print!("   {} {}", icon, check.name);
        if let Some(msg) = &check.message {
            print!(": {}", msg);
        }
        println!();
    }

    println!("\n7. Cleanup...");
    drop(db);
    println!("   ✅ Database closed gracefully\n");

    println!("=== Observability Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Use db.stats() for comprehensive metrics");
    println!("  • Use db.health() for operational health checks");
    println!("  • Enable tracing for structured logging");
    println!("  • Monitor latency percentiles (p99/p999) in production");
    println!("  • Set up alerts on health status changes");
    println!("  • Export metrics to your monitoring system");
}
