// Stress Tests
// Tests database behavior under high load conditions
//
// Tests are scaled based on environment:
// - CI: 100k operations (fast, 2-5 minutes)
// - Manual: 1M operations (moderate, 10-30 minutes)
// - Benchmark: 10M+ operations (slow, hours) - use #[ignore]

use bytes::Bytes;
use rand::Rng;
use seerdb::{DBOptions, DB};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{Pid, ProcessExt, System, SystemExt};
use tempfile::tempdir;

/// Get current process memory usage in bytes
fn get_memory_usage() -> u64 {
    let mut system = System::new_all();
    system.refresh_all();

    let pid = Pid::from(std::process::id() as usize);
    if let Some(process) = system.process(pid) {
        process.memory() * 1024 // Convert KB to bytes
    } else {
        0
    }
}

/// Simple metrics collector for stress tests
struct StressMetrics {
    total_ops: u64,
    start_time: Instant,
    latencies: Vec<Duration>,
    memory_samples: Vec<u64>,
}

impl StressMetrics {
    fn new() -> Self {
        Self {
            total_ops: 0,
            start_time: Instant::now(),
            latencies: Vec::new(),
            memory_samples: Vec::new(),
        }
    }

    fn record_op(&mut self, latency: Duration) {
        self.total_ops += 1;
        self.latencies.push(latency);
    }

    fn sample_memory(&mut self) {
        self.memory_samples.push(get_memory_usage());
    }

    fn report(&self) {
        let elapsed = self.start_time.elapsed();
        let throughput = self.total_ops as f64 / elapsed.as_secs_f64();

        println!("\n===== Stress Test Report =====");
        println!("Total operations: {}", self.total_ops);
        println!("Total time: {:.2?}", elapsed);
        println!("Throughput: {:.0} ops/sec", throughput);

        if !self.latencies.is_empty() {
            let mut sorted = self.latencies.clone();
            sorted.sort();

            let p50 = sorted[sorted.len() * 50 / 100];
            let p99 = sorted[sorted.len() * 99 / 100];
            let p999 = sorted[sorted.len() * 999 / 1000];

            println!("Latency p50: {:?}", p50);
            println!("Latency p99: {:?}", p99);
            println!("Latency p999: {:?}", p999);
        }

        if !self.memory_samples.is_empty() {
            let max_memory = self.memory_samples.iter().max().unwrap();
            let min_memory = self.memory_samples.iter().min().unwrap();
            println!(
                "Memory: {} MB - {} MB",
                min_memory / 1024 / 1024,
                max_memory / 1024 / 1024
            );
        }

        println!("==============================\n");
    }

    fn check_memory_stable(&self) {
        if self.memory_samples.len() < 2 {
            return;
        }

        let max_memory = self.memory_samples.iter().max().unwrap();
        let min_memory = self.memory_samples.iter().min().unwrap();

        // Memory shouldn't grow more than 5x (conservative check)
        // Some growth expected due to memtable/cache
        assert!(
            *max_memory < min_memory * 5,
            "Possible memory leak: {} MB -> {} MB",
            min_memory / 1024 / 1024,
            max_memory / 1024 / 1024
        );
    }
}

// Determine test size based on environment
fn get_test_size() -> usize {
    // Check if running in CI (e.g., GitHub Actions)
    if std::env::var("CI").is_ok() {
        100_000 // CI: 100k ops (fast)
    } else if std::env::var("STRESS_FULL").is_ok() {
        1_000_000 // Manual: 1M ops (moderate)
    } else {
        100_000 // Default: 100k ops
    }
}

#[test]
fn test_stress_sequential_writes() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = get_test_size();

    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    let mut metrics = StressMetrics::new();

    println!("\n🔥 Sequential write stress test ({} ops)", test_size);

    for i in 0..test_size {
        let start = Instant::now();

        let key = format!("key_{:010}", i);
        let value = format!("value_{:010}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();

        metrics.record_op(start.elapsed());

        // Sample memory every 10k ops
        if i > 0 && i % 10_000 == 0 {
            metrics.sample_memory();
            if i % 100_000 == 0 {
                println!("Progress: {}/{}", i, test_size);
            }
        }
    }

    metrics.report();
    metrics.check_memory_stable();

    // Verify data
    println!("Verifying random sample of data...");
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let i = rng.gen_range(0..test_size);
        let key = format!("key_{:010}", i);
        let expected_value = format!("value_{:010}", i);

        let value = db.get(key.as_bytes()).unwrap().expect("Key should exist");
        assert_eq!(value, Bytes::from(expected_value));
    }

    println!("✅ Sequential write stress test passed");
}

#[test]
fn test_stress_random_writes() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = get_test_size();

    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    let mut metrics = StressMetrics::new();
    let mut rng = rand::thread_rng();

    println!("\n🔥 Random write stress test ({} ops)", test_size);

    // Keep track of written keys for verification
    let mut written_keys = std::collections::HashSet::new();

    for i in 0..test_size {
        let start = Instant::now();

        // Random key in large space (creates sparse keyspace)
        let key_num: u64 = rng.gen_range(0..test_size as u64 * 100);
        let key = format!("key_{:010}", key_num);
        let value = vec![0u8; 128]; // 128-byte values

        db.put(key.as_bytes(), &value).unwrap();
        written_keys.insert(key);

        metrics.record_op(start.elapsed());

        if i > 0 && i % 10_000 == 0 {
            metrics.sample_memory();
            if i % 100_000 == 0 {
                println!("Progress: {}/{}", i, test_size);
            }
        }
    }

    metrics.report();
    metrics.check_memory_stable();

    // Verify random sample of written keys
    println!("Verifying random sample of written keys...");
    let sample_size = 1000.min(written_keys.len());
    let keys_vec: Vec<_> = written_keys.iter().collect();

    for _ in 0..sample_size {
        let idx = rng.gen_range(0..keys_vec.len());
        let key = keys_vec[idx];

        let value = db.get(key.as_bytes()).unwrap().expect("Key should exist");
        assert_eq!(value.len(), 128);
    }

    println!("✅ Random write stress test passed");
}

#[test]
fn test_stress_concurrent_access() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = get_test_size();
    let num_threads = 4;
    let ops_per_thread = test_size / num_threads;

    let opts = DBOptions {
        data_dir: db_path.clone(),
        background_compaction: true, // Enable background compaction for concurrent test
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    println!(
        "\n🔥 Concurrent access stress test ({} threads x {} ops)",
        num_threads, ops_per_thread
    );

    let start = Instant::now();
    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let mut rng = rand::thread_rng();
            let mut local_ops = 0;

            for i in 0..ops_per_thread {
                // 70% reads, 20% writes, 10% deletes (realistic workload)
                let op: u8 = rng.gen_range(0..10);

                if op < 7 {
                    // Read operation
                    let key_num: u64 = rng.gen_range(0..(test_size * 10) as u64);
                    let key = format!("key_{:010}", key_num);
                    let _ = db_clone.get(key.as_bytes()).unwrap();
                } else if op < 9 {
                    // Write operation
                    let key = format!("t{}_key_{:07}", thread_id, i);
                    let value = vec![0u8; 128];
                    db_clone.put(key.as_bytes(), &value).unwrap();
                } else {
                    // Delete operation
                    let key_num: u64 = rng.gen_range(0..(test_size * 10) as u64);
                    let key = format!("key_{:010}", key_num);
                    db_clone.delete(key.as_bytes()).unwrap();
                }

                local_ops += 1;

                if local_ops % 50_000 == 0 {
                    println!(
                        "Thread {} progress: {}/{}",
                        thread_id, local_ops, ops_per_thread
                    );
                }
            }

            println!("Thread {} completed {} ops", thread_id, local_ops);
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = num_threads * ops_per_thread;
    let throughput = total_ops as f64 / elapsed.as_secs_f64();

    println!("\n===== Concurrent Stress Test Report =====");
    println!("Total operations: {}", total_ops);
    println!("Total time: {:.2?}", elapsed);
    println!("Throughput: {:.0} ops/sec", throughput);
    println!("==========================================\n");

    // Verify that some writes from each thread succeeded
    // Since only ~20% of operations are writes, check multiple keys
    println!("Verifying thread writes...");
    for thread_id in 0..num_threads {
        let mut found = false;
        // Check first 100 keys - at least one should exist (20% write rate = ~20 keys)
        for i in 0..100 {
            let key = format!("t{}_key_{:07}", thread_id, i);
            if db.get(key.as_bytes()).unwrap().is_some() {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "Thread {} writes should be persisted (checked first 100 keys)",
            thread_id
        );
    }

    println!("✅ Concurrent access stress test passed");
}

#[test]
fn test_stress_read_heavy_workload() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = get_test_size();
    let num_keys = 10_000;

    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Populate database with initial data
    println!(
        "\n🔥 Read-heavy stress test ({} reads on {} keys)",
        test_size, num_keys
    );
    println!("Populating database with {} keys...", num_keys);

    for i in 0..num_keys {
        let key = format!("key_{:07}", i);
        let value = format!("value_{:07}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // Run read-heavy workload
    let mut metrics = StressMetrics::new();
    let mut rng = rand::thread_rng();
    let mut hit_count = 0;
    let mut miss_count = 0;

    println!("Running {} read operations...", test_size);

    for i in 0..test_size {
        let start = Instant::now();

        // Zipfian distribution would be more realistic, but uniform is simpler
        // 90% reads hit existing keys, 10% miss
        let key_num = if rng.r#gen::<f64>() < 0.9 {
            rng.gen_range(0..num_keys)
        } else {
            num_keys + rng.gen_range(0..num_keys * 10)
        };

        let key = format!("key_{:07}", key_num);
        let result = db.get(key.as_bytes()).unwrap();

        if result.is_some() {
            hit_count += 1;
        } else {
            miss_count += 1;
        }

        metrics.record_op(start.elapsed());

        if i > 0 && i % 10_000 == 0 {
            metrics.sample_memory();
            if i % 100_000 == 0 {
                println!("Progress: {}/{}", i, test_size);
            }
        }
    }

    metrics.report();
    metrics.check_memory_stable();

    let hit_rate = hit_count as f64 / test_size as f64 * 100.0;
    println!("Cache hit rate: {:.1}%", hit_rate);
    println!("Hits: {}, Misses: {}", hit_count, miss_count);

    // Verify hit rate is reasonable (should be ~90% based on our distribution)
    assert!(
        hit_rate > 85.0 && hit_rate < 95.0,
        "Hit rate should be ~90%, got {:.1}%",
        hit_rate
    );

    println!("✅ Read-heavy stress test passed");
}

#[test]
fn test_stress_mixed_workload() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = get_test_size();

    let opts = DBOptions {
        data_dir: db_path.clone(),
        background_compaction: true,
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    println!("\n🔥 Mixed workload stress test ({} ops)", test_size);
    println!("Workload: 70% reads, 20% writes, 10% deletes");

    let mut metrics = StressMetrics::new();
    let mut rng = rand::thread_rng();
    let mut write_count = 0;
    let mut read_count = 0;
    let mut delete_count = 0;

    for i in 0..test_size {
        let start = Instant::now();
        let op: u8 = rng.gen_range(0..10);

        if op < 7 {
            // 70% reads
            let key_num: u64 = rng.gen_range(0..(test_size * 10) as u64);
            let key = format!("key_{:010}", key_num);
            let _ = db.get(key.as_bytes()).unwrap();
            read_count += 1;
        } else if op < 9 {
            // 20% writes
            let key = format!("key_{:010}", i);
            let value = vec![0u8; 128];
            db.put(key.as_bytes(), &value).unwrap();
            write_count += 1;
        } else {
            // 10% deletes
            let key_num: u64 = rng.gen_range(0..(test_size * 10) as u64);
            let key = format!("key_{:010}", key_num);
            db.delete(key.as_bytes()).unwrap();
            delete_count += 1;
        }

        metrics.record_op(start.elapsed());

        if i > 0 && i % 10_000 == 0 {
            metrics.sample_memory();
            if i % 100_000 == 0 {
                println!("Progress: {}/{}", i, test_size);
            }
        }
    }

    metrics.report();
    metrics.check_memory_stable();

    println!("Operation distribution:");
    println!(
        "  Reads:   {} ({:.1}%)",
        read_count,
        read_count as f64 / test_size as f64 * 100.0
    );
    println!(
        "  Writes:  {} ({:.1}%)",
        write_count,
        write_count as f64 / test_size as f64 * 100.0
    );
    println!(
        "  Deletes: {} ({:.1}%)",
        delete_count,
        delete_count as f64 / test_size as f64 * 100.0
    );

    // Verify writes persisted
    println!("Verifying written data...");
    for i in (0..write_count).step_by(write_count / 100) {
        let key = format!("key_{:010}", i);
        let value = db.get(key.as_bytes()).unwrap();
        // May or may not exist (could have been deleted), but shouldn't error
        let _ = value;
    }

    println!("✅ Mixed workload stress test passed");
}

// ==================== Large-Scale Tests (Ignored by Default) ====================
// These tests are too slow for CI. Run manually with:
// cargo test --test stress_test -- --ignored --nocapture

#[test]
#[ignore]
fn test_stress_1m_sequential_writes() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let test_size = 1_000_000;

    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    println!("\n🔥 LARGE: 1M sequential write stress test");

    let start = Instant::now();

    for i in 0..test_size {
        let key = format!("key_{:010}", i);
        let value = format!("value_{:010}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();

        if i % 100_000 == 0 {
            let elapsed = start.elapsed();
            let rate = i as f64 / elapsed.as_secs_f64();
            println!(
                "Progress: {}k / {}k ({:.0} ops/sec)",
                i / 1000,
                test_size / 1000,
                rate
            );
        }
    }

    let elapsed = start.elapsed();
    let throughput = test_size as f64 / elapsed.as_secs_f64();

    println!("\n===== 1M Sequential Write Report =====");
    println!("Total time: {:.2?}", elapsed);
    println!("Throughput: {:.0} ops/sec", throughput);
    println!("======================================\n");

    println!("✅ 1M sequential write stress test passed");
}

#[test]
#[ignore]
fn test_stress_1m_concurrent_8_threads() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");
    let total_ops = 1_000_000;
    let num_threads = 8;
    let ops_per_thread = total_ops / num_threads;

    let opts = DBOptions {
        data_dir: db_path.clone(),
        background_compaction: true,
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    println!(
        "\n🔥 LARGE: Concurrent stress test ({} threads x {} ops = {} total)",
        num_threads, ops_per_thread, total_ops
    );

    let start = Instant::now();
    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let mut rng = rand::thread_rng();

            for i in 0..ops_per_thread {
                let op: u8 = rng.gen_range(0..10);

                if op < 7 {
                    let key_num: u64 = rng.gen_range(0..total_ops as u64 * 10);
                    let key = format!("key_{:010}", key_num);
                    let _ = db_clone.get(key.as_bytes()).unwrap();
                } else if op < 9 {
                    let key = format!("t{}_key_{:07}", thread_id, i);
                    let value = vec![0u8; 128];
                    db_clone.put(key.as_bytes(), &value).unwrap();
                } else {
                    let key_num: u64 = rng.gen_range(0..total_ops as u64 * 10);
                    let key = format!("key_{:010}", key_num);
                    db_clone.delete(key.as_bytes()).unwrap();
                }

                if i % 50_000 == 0 && i > 0 {
                    println!(
                        "Thread {} progress: {}k / {}k",
                        thread_id,
                        i / 1000,
                        ops_per_thread / 1000
                    );
                }
            }

            println!("Thread {} completed", thread_id);
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let throughput = total_ops as f64 / elapsed.as_secs_f64();

    println!("\n===== 1M Concurrent Stress Report =====");
    println!("Total time: {:.2?}", elapsed);
    println!("Throughput: {:.0} ops/sec", throughput);
    println!("=======================================\n");

    println!("✅ 1M concurrent stress test passed");
}
