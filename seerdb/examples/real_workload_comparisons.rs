// Real Workload Comparisons - Phase 4 Profiling
//
// Compares seerdb vs RocksDB vs fjall on three realistic workloads:
// 1. Graph pattern (HNSW graph: prefix scans)
// 2. Time series (sequential timestamps: range queries)
// 3. Random KV (random access: point lookups)
//
// Run with: cargo run --release --features baseline-benchmarks --example real_workload_comparisons

use std::time::{Duration, Instant};
use tempfile::tempdir;

#[cfg(feature = "baseline-benchmarks")]
use fjall::Config as FjallConfig;
#[cfg(feature = "baseline-benchmarks")]
use rocksdb::{Options as RocksDBOptions, DB as RocksDBInstance};

use seerdb::{DBOptions, DB};

const WARMUP_OPS: usize = 1_000;

// === Workload 1: Graph Pattern (HNSW graph edges) ===

fn benchmark_graph_seerdb() -> (Duration, Duration, f64) {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 64 * 1024 * 1024, // 64MB
        block_cache_capacity: 16_384,
        ..Default::default()
    };
    let db = DB::open(options).unwrap();

    // Setup: 10K nodes × 32 edges/node = 320K entries
    let nodes = 10_000;
    let edges_per_node = 32;
    let value = vec![0u8; 128]; // 128 bytes (edge metadata)

    // Write phase
    let write_start = Instant::now();
    for node_id in 0..nodes {
        for edge_id in 0..edges_per_node {
            let key = format!("node:{}:edge:{:04}", node_id, edge_id);
            db.put(key.as_bytes(), &value).unwrap();
        }
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase: Prefix scans (typical HNSW query)
    let prefixes: Vec<String> = (0..100)
        .map(|i| format!("node:{}:", i * (nodes / 100)))
        .collect();

    // Warmup
    for prefix in prefixes.iter().take(10) {
        let iter = db.prefix(prefix.as_bytes()).unwrap();
        let _: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
    }

    let read_start = Instant::now();
    let mut total_entries = 0;
    for prefix in prefixes.iter() {
        let iter = db.prefix(prefix.as_bytes()).unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        total_entries += entries.len();
    }
    let read_duration = read_start.elapsed();

    let stats = db.stats();
    (write_duration, read_duration, stats.cache_hit_rate)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_graph_rocksdb() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let mut opts = RocksDBOptions::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    let db = RocksDBInstance::open(&opts, dir.path()).unwrap();

    let nodes = 10_000;
    let edges_per_node = 32;
    let value = vec![0u8; 128];

    // Write phase
    let write_start = Instant::now();
    for node_id in 0..nodes {
        for edge_id in 0..edges_per_node {
            let key = format!("node:{}:edge:{:04}", node_id, edge_id);
            db.put(key.as_bytes(), &value).unwrap();
        }
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase: Prefix scans
    let prefixes: Vec<String> = (0..100)
        .map(|i| format!("node:{}:", i * (nodes / 100)))
        .collect();

    // Warmup
    for prefix in prefixes.iter().take(10) {
        let mut iter = db.prefix_iterator(prefix.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }

    let read_start = Instant::now();
    for prefix in prefixes.iter() {
        let mut iter = db.prefix_iterator(prefix.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_graph_fjall() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let config = FjallConfig::new(dir.path());
    let keyspace = config.open().unwrap();
    let partition = keyspace
        .open_partition("default", Default::default())
        .unwrap();

    let nodes = 10_000;
    let edges_per_node = 32;
    let value = vec![0u8; 128];

    // Write phase
    let write_start = Instant::now();
    for node_id in 0..nodes {
        for edge_id in 0..edges_per_node {
            let key = format!("node:{}:edge:{:04}", node_id, edge_id);
            partition.insert(key.as_bytes(), &value).unwrap();
        }
    }
    keyspace.persist(fjall::PersistMode::SyncAll).unwrap();
    let write_duration = write_start.elapsed();

    // Read phase: Prefix scans
    let prefixes: Vec<String> = (0..100)
        .map(|i| format!("node:{}:", i * (nodes / 100)))
        .collect();

    // Warmup
    for prefix in prefixes.iter().take(10) {
        let mut iter = partition.prefix(prefix.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }

    let read_start = Instant::now();
    for prefix in prefixes.iter() {
        let mut iter = partition.prefix(prefix.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

// === Workload 2: Time Series (sequential timestamps) ===

fn benchmark_timeseries_seerdb() -> (Duration, Duration, f64) {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 64 * 1024 * 1024,
        block_cache_capacity: 16_384,
        ..Default::default()
    };
    let db = DB::open(options).unwrap();

    // Setup: 1M time series entries (timestamp → sensor data)
    let entries = 1_000_000;
    let value = vec![0u8; 64]; // Sensor reading (64 bytes)

    // Write phase (sequential timestamps)
    let write_start = Instant::now();
    for ts in 0..entries {
        let key = format!("ts:{:016}", ts);
        db.put(key.as_bytes(), &value).unwrap();
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase: Range queries (typical time series query)
    let ranges: Vec<(u64, u64)> = (0..100)
        .map(|i| {
            let start = i * (entries / 100);
            let end = start + 10_000; // 10K entry window
            (start, end)
        })
        .collect();

    // Warmup
    for (start, end) in ranges.iter().take(10) {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let iter = db
            .range(start_key.as_bytes(), Some(end_key.as_bytes()))
            .unwrap();
        let _: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
    }

    let read_start = Instant::now();
    let mut total_entries = 0;
    for (start, end) in ranges.iter() {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let iter = db
            .range(start_key.as_bytes(), Some(end_key.as_bytes()))
            .unwrap();
        let entries: Vec<_> = iter.collect::<Result<Vec<_>, _>>().unwrap();
        total_entries += entries.len();
    }
    let read_duration = read_start.elapsed();

    let stats = db.stats();
    (write_duration, read_duration, stats.cache_hit_rate)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_timeseries_rocksdb() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let mut opts = RocksDBOptions::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    let db = RocksDBInstance::open(&opts, dir.path()).unwrap();

    let entries = 1_000_000;
    let value = vec![0u8; 64];

    // Write phase
    let write_start = Instant::now();
    for ts in 0..entries {
        let key = format!("ts:{:016}", ts);
        db.put(key.as_bytes(), &value).unwrap();
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase
    let ranges: Vec<(u64, u64)> = (0..100)
        .map(|i| {
            let start = i * (entries / 100);
            let end = start + 10_000;
            (start, end)
        })
        .collect();

    // Warmup
    for (start, end) in ranges.iter().take(10) {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let mut iter = db.iterator(rocksdb::IteratorMode::From(
            start_key.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        while let Some(Ok((key, _))) = iter.next() {
            if key.as_ref() >= end_key.as_bytes() {
                break;
            }
        }
    }

    let read_start = Instant::now();
    for (start, end) in ranges.iter() {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let mut iter = db.iterator(rocksdb::IteratorMode::From(
            start_key.as_bytes(),
            rocksdb::Direction::Forward,
        ));
        while let Some(Ok((key, _))) = iter.next() {
            if key.as_ref() >= end_key.as_bytes() {
                break;
            }
        }
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_timeseries_fjall() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let config = FjallConfig::new(dir.path());
    let keyspace = config.open().unwrap();
    let partition = keyspace
        .open_partition("default", Default::default())
        .unwrap();

    let entries = 1_000_000;
    let value = vec![0u8; 64];

    // Write phase
    let write_start = Instant::now();
    for ts in 0..entries {
        let key = format!("ts:{:016}", ts);
        partition.insert(key.as_bytes(), &value).unwrap();
    }
    keyspace.persist(fjall::PersistMode::SyncAll).unwrap();
    let write_duration = write_start.elapsed();

    // Read phase
    let ranges: Vec<(u64, u64)> = (0..100)
        .map(|i| {
            let start = i * (entries / 100);
            let end = start + 10_000;
            (start, end)
        })
        .collect();

    // Warmup
    for (start, end) in ranges.iter().take(10) {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let mut iter = partition.range(start_key.as_bytes()..end_key.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }

    let read_start = Instant::now();
    for (start, end) in ranges.iter() {
        let start_key = format!("ts:{:016}", start);
        let end_key = format!("ts:{:016}", end);
        let mut iter = partition.range(start_key.as_bytes()..end_key.as_bytes());
        while let Some(Ok(_)) = iter.next() {}
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

// === Workload 3: Random KV (point lookups) ===

fn benchmark_random_seerdb() -> (Duration, Duration, f64) {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 64 * 1024 * 1024,
        block_cache_capacity: 16_384,
        ..Default::default()
    };
    let db = DB::open(options).unwrap();

    // Setup: 500K random key-value pairs
    let entries = 500_000;
    let value = vec![0u8; 1024]; // 1KB values

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let keys: Vec<String> = (0..entries)
        .map(|_| format!("key:{:016x}", rng.gen::<u64>()))
        .collect();

    // Write phase (random keys)
    let write_start = Instant::now();
    for key in keys.iter() {
        db.put(key.as_bytes(), &value).unwrap();
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase: Point lookups (random order)
    let mut read_keys = keys.clone();
    use rand::seq::SliceRandom;
    read_keys.shuffle(&mut rng);

    // Warmup
    for key in read_keys.iter().take(WARMUP_OPS) {
        let _ = db.get(key.as_bytes()).unwrap();
    }

    let read_start = Instant::now();
    for key in read_keys.iter().take(100_000) {
        let _ = db.get(key.as_bytes()).unwrap();
    }
    let read_duration = read_start.elapsed();

    let stats = db.stats();
    (write_duration, read_duration, stats.cache_hit_rate)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_random_rocksdb() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let mut opts = RocksDBOptions::default();
    opts.create_if_missing(true);
    opts.set_write_buffer_size(64 * 1024 * 1024);
    let db = RocksDBInstance::open(&opts, dir.path()).unwrap();

    let entries = 500_000;
    let value = vec![0u8; 1024];

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let keys: Vec<String> = (0..entries)
        .map(|_| format!("key:{:016x}", rng.gen::<u64>()))
        .collect();

    // Write phase
    let write_start = Instant::now();
    for key in keys.iter() {
        db.put(key.as_bytes(), &value).unwrap();
    }
    db.flush().unwrap();
    let write_duration = write_start.elapsed();

    // Read phase
    let mut read_keys = keys.clone();
    use rand::seq::SliceRandom;
    read_keys.shuffle(&mut rng);

    // Warmup
    for key in read_keys.iter().take(WARMUP_OPS) {
        let _ = db.get(key.as_bytes()).unwrap();
    }

    let read_start = Instant::now();
    for key in read_keys.iter().take(100_000) {
        let _ = db.get(key.as_bytes()).unwrap();
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

#[cfg(feature = "baseline-benchmarks")]
fn benchmark_random_fjall() -> (Duration, Duration) {
    let dir = tempdir().unwrap();
    let config = FjallConfig::new(dir.path());
    let keyspace = config.open().unwrap();
    let partition = keyspace
        .open_partition("default", Default::default())
        .unwrap();

    let entries = 500_000;
    let value = vec![0u8; 1024];

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let keys: Vec<String> = (0..entries)
        .map(|_| format!("key:{:016x}", rng.gen::<u64>()))
        .collect();

    // Write phase
    let write_start = Instant::now();
    for key in keys.iter() {
        partition.insert(key.as_bytes(), &value).unwrap();
    }
    keyspace.persist(fjall::PersistMode::SyncAll).unwrap();
    let write_duration = write_start.elapsed();

    // Read phase
    let mut read_keys = keys.clone();
    use rand::seq::SliceRandom;
    read_keys.shuffle(&mut rng);

    // Warmup
    for key in read_keys.iter().take(WARMUP_OPS) {
        let _ = partition.get(key.as_bytes()).unwrap();
    }

    let read_start = Instant::now();
    for key in read_keys.iter().take(100_000) {
        let _ = partition.get(key.as_bytes()).unwrap();
    }
    let read_duration = read_start.elapsed();

    (write_duration, read_duration)
}

fn main() {
    println!("=== Real Workload Comparisons - Phase 4 Profiling ===\n");
    println!("Workloads:");
    println!("1. Graph (HNSW): 10K nodes × 32 edges = 320K entries, 100 prefix scans");
    println!("2. Time series: 1M entries, 100 range queries (10K entries each)");
    println!("3. Random KV: 500K entries, 100K point lookups\n");

    // Workload 1: Graph pattern
    println!("=== Workload 1: Graph Pattern (HNSW Graph) ===\n");

    print!("seerdb... ");
    let (seer_write, seer_read, seer_cache) = benchmark_graph_seerdb();
    println!(
        "Write: {:.2}s, Read: {:.2}s, Cache: {:.2}%",
        seer_write.as_secs_f64(),
        seer_read.as_secs_f64(),
        seer_cache * 100.0
    );

    #[cfg(feature = "baseline-benchmarks")]
    {
        print!("RocksDB... ");
        let (rocks_write, rocks_read) = benchmark_graph_rocksdb();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            rocks_write.as_secs_f64(),
            rocks_read.as_secs_f64()
        );

        print!("fjall... ");
        let (fjall_write, fjall_read) = benchmark_random_fjall();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            fjall_write.as_secs_f64(),
            fjall_read.as_secs_f64()
        );

        println!(
            "\nSpeedup vs RocksDB: Write {:.2}x, Read {:.2}x",
            rocks_write.as_secs_f64() / seer_write.as_secs_f64(),
            rocks_read.as_secs_f64() / seer_read.as_secs_f64()
        );
        println!(
            "Speedup vs fjall: Write {:.2}x, Read {:.2}x\n",
            fjall_write.as_secs_f64() / seer_write.as_secs_f64(),
            fjall_read.as_secs_f64() / seer_read.as_secs_f64()
        );
    }

    // Workload 2: Time series
    println!("=== Workload 2: Time Series (Sequential Timestamps) ===\n");

    print!("seerdb... ");
    let (seer_write, seer_read, seer_cache) = benchmark_timeseries_seerdb();
    println!(
        "Write: {:.2}s, Read: {:.2}s, Cache: {:.2}%",
        seer_write.as_secs_f64(),
        seer_read.as_secs_f64(),
        seer_cache * 100.0
    );

    #[cfg(feature = "baseline-benchmarks")]
    {
        print!("RocksDB... ");
        let (rocks_write, rocks_read) = benchmark_timeseries_rocksdb();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            rocks_write.as_secs_f64(),
            rocks_read.as_secs_f64()
        );

        print!("fjall... ");
        let (fjall_write, fjall_read) = benchmark_timeseries_fjall();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            fjall_write.as_secs_f64(),
            fjall_read.as_secs_f64()
        );

        println!(
            "\nSpeedup vs RocksDB: Write {:.2}x, Read {:.2}x",
            rocks_write.as_secs_f64() / seer_write.as_secs_f64(),
            rocks_read.as_secs_f64() / seer_read.as_secs_f64()
        );
        println!(
            "Speedup vs fjall: Write {:.2}x, Read {:.2}x\n",
            fjall_write.as_secs_f64() / seer_write.as_secs_f64(),
            fjall_read.as_secs_f64() / seer_read.as_secs_f64()
        );
    }

    // Workload 3: Random KV
    println!("=== Workload 3: Random Key-Value (Point Lookups) ===\n");

    print!("seerdb... ");
    let (seer_write, seer_read, seer_cache) = benchmark_random_seerdb();
    println!(
        "Write: {:.2}s, Read: {:.2}s, Cache: {:.2}%",
        seer_write.as_secs_f64(),
        seer_read.as_secs_f64(),
        seer_cache * 100.0
    );

    #[cfg(feature = "baseline-benchmarks")]
    {
        print!("RocksDB... ");
        let (rocks_write, rocks_read) = benchmark_random_rocksdb();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            rocks_write.as_secs_f64(),
            rocks_read.as_secs_f64()
        );

        print!("fjall... ");
        let (fjall_write, fjall_read) = benchmark_random_fjall();
        println!(
            "Write: {:.2}s, Read: {:.2}s",
            fjall_write.as_secs_f64(),
            fjall_read.as_secs_f64()
        );

        println!(
            "\nSpeedup vs RocksDB: Write {:.2}x, Read {:.2}x",
            rocks_write.as_secs_f64() / seer_write.as_secs_f64(),
            rocks_read.as_secs_f64() / seer_read.as_secs_f64()
        );
        println!(
            "Speedup vs fjall: Write {:.2}x, Read {:.2}x",
            fjall_write.as_secs_f64() / seer_write.as_secs_f64(),
            fjall_read.as_secs_f64() / seer_read.as_secs_f64()
        );
    }

    println!("\n=== Phase 4 Profiling Complete ===");
    println!("\nRun with --features baseline-benchmarks to compare against RocksDB and fjall");
}
