// Billion-Key Scale Test
// Validates seerdb at massive scale (100M, 500M, 1B keys)
// Run on Fedora (i9-13900KF, 32GB) for SOTA performance validation

use seerdb::{DBOptions, DB};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const KB: usize = 1024;
const MB: usize = 1024 * KB;
const GB: usize = 1024 * MB;

struct ScaleConfig {
    name: &'static str,
    num_keys: u64,
    key_size: usize,
    value_size: usize,
    num_threads: usize,
    batch_size: usize,
    sample_rate: u64,
}

impl ScaleConfig {
    fn estimated_data_size(&self) -> usize {
        self.num_keys as usize * (self.key_size + self.value_size)
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_ops(ops: f64) -> String {
    if ops >= 1_000_000.0 {
        format!("{:.2}M ops/sec", ops / 1_000_000.0)
    } else if ops >= 1_000.0 {
        format!("{:.1}K ops/sec", ops / 1_000.0)
    } else {
        format!("{:.0} ops/sec", ops)
    }
}

fn make_key(i: u64, key_size: usize) -> String {
    format!("{:0width$}", i, width = key_size)
}

fn make_value(key: &str, value_size: usize) -> Vec<u8> {
    let key_bytes = key.as_bytes();
    (0..value_size)
        .map(|j| key_bytes[j % key_bytes.len()])
        .collect()
}

fn run_scale_test(
    config: &ScaleConfig,
    data_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    println!("Scale Test: {}", config.name);
    println!(
        "  Keys: {} ({} estimated)",
        config.num_keys,
        format_size(config.estimated_data_size())
    );
    println!(
        "  Key size: {} bytes, Value size: {} bytes",
        config.key_size, config.value_size
    );
    println!(
        "  Threads: {}, Batch size: {}",
        config.num_threads, config.batch_size
    );
    println!("{}", "=".repeat(60));

    // Use default options for reliability
    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts)?);

    let written = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let keys_per_thread = config.num_keys / config.num_threads as u64;

    // Progress reporter thread
    let written_clone = Arc::clone(&written);
    let total_keys = config.num_keys;
    let progress_handle = thread::spawn(move || {
        let mut last_count = 0u64;
        let mut last_time = Instant::now();
        loop {
            thread::sleep(Duration::from_secs(5));
            let current = written_clone.load(Ordering::Relaxed);
            let elapsed = last_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                (current - last_count) as f64 / elapsed
            } else {
                0.0
            };
            let pct = current as f64 / total_keys as f64 * 100.0;

            print!(
                "\r  Progress: {:.1}% ({}/{}) - {} instantaneous    ",
                pct,
                current,
                total_keys,
                format_ops(rate)
            );
            io::stdout().flush().ok();

            if current >= total_keys {
                break;
            }
            last_count = current;
            last_time = Instant::now();
        }
    });

    // Writer threads using batch API for proper LSM-tree behavior
    let mut handles = vec![];
    for thread_id in 0..config.num_threads {
        let db = Arc::clone(&db);
        let written = Arc::clone(&written);
        let batch_size = config.batch_size;
        let key_size = config.key_size;
        let value_size = config.value_size;
        let start_key = thread_id as u64 * keys_per_thread;
        let end_key = start_key + keys_per_thread;

        handles.push(thread::spawn(move || {
            let mut batch = db.batch();
            let mut batch_count = 0;

            for i in start_key..end_key {
                let key = make_key(i, key_size);
                let value = make_value(&key, value_size);

                batch.put(key.as_bytes(), &value);
                batch_count += 1;

                if batch_count >= batch_size {
                    batch.commit().expect("batch commit failed");
                    written.fetch_add(batch_count as u64, Ordering::Relaxed);
                    batch = db.batch();
                    batch_count = 0;
                }
            }

            // Commit remaining
            if batch_count > 0 {
                batch.commit().expect("batch commit failed");
                written.fetch_add(batch_count as u64, Ordering::Relaxed);
            }
        }));
    }

    // Wait for writers
    for h in handles {
        h.join().expect("thread panicked");
    }

    let write_elapsed = start.elapsed();
    progress_handle.join().ok();
    println!();

    // Force flush
    println!("  Flushing...");
    let flush_start = Instant::now();
    db.flush()?;
    let flush_elapsed = flush_start.elapsed();

    let write_ops = config.num_keys as f64 / write_elapsed.as_secs_f64();
    println!(
        "  Write complete: {} in {:.1}s ({})",
        config.num_keys,
        write_elapsed.as_secs_f64(),
        format_ops(write_ops)
    );
    println!("  Flush time: {:.1}s", flush_elapsed.as_secs_f64());

    // Sample verification
    println!(
        "  Verifying (sampling every {} keys)...",
        config.sample_rate
    );
    let verify_start = Instant::now();
    let mut verified = 0u64;
    let mut errors = 0u64;

    for i in (0..config.num_keys).step_by(config.sample_rate as usize) {
        let key = make_key(i, config.key_size);
        match db.get(key.as_bytes())? {
            Some(value) => {
                let expected = make_value(&key, config.value_size);
                if value.as_ref() != expected.as_slice() {
                    errors += 1;
                    if errors <= 5 {
                        eprintln!("    Value mismatch at key {}", i);
                    }
                }
                verified += 1;
            }
            None => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("    Missing key: {} (formatted: '{}')", i, key);
                }
            }
        }
    }

    let verify_elapsed = verify_start.elapsed();
    let verify_ops = verified as f64 / verify_elapsed.as_secs_f64();
    println!(
        "  Verified {} samples in {:.1}s ({}) - {} errors",
        verified,
        verify_elapsed.as_secs_f64(),
        format_ops(verify_ops),
        errors
    );

    // Read benchmark (random point reads)
    println!("  Random read benchmark...");
    let read_start = Instant::now();
    let read_count = 100_000u64;
    let mut found = 0u64;

    for i in 0..read_count {
        let key_num = (i * 7919) % config.num_keys;
        let key = make_key(key_num, config.key_size);
        if db.get(key.as_bytes())?.is_some() {
            found += 1;
        }
    }

    let read_elapsed = read_start.elapsed();
    let read_ops = read_count as f64 / read_elapsed.as_secs_f64();
    println!(
        "  Random reads: {}/{} found in {:.3}s ({})",
        found,
        read_count,
        read_elapsed.as_secs_f64(),
        format_ops(read_ops)
    );

    // Range scan benchmark
    println!("  Range scan benchmark...");
    let scan_start = Instant::now();
    let scan_count = db.iter()?.take(1_000_000).count();
    let scan_elapsed = scan_start.elapsed();
    let scan_ops = scan_count as f64 / scan_elapsed.as_secs_f64();
    println!(
        "  Scanned {} entries in {:.3}s ({})",
        scan_count,
        scan_elapsed.as_secs_f64(),
        format_ops(scan_ops)
    );

    if errors > 0 {
        return Err(format!("{} verification errors", errors).into());
    }

    println!("\n  [ok] {} complete - all checks passed", config.name);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Billion-Key Scale Test for seerdb");
    println!("==================================");
    println!();

    let args: Vec<String> = std::env::args().collect();
    let scale = args.get(1).map(|s| s.as_str()).unwrap_or("100m");
    let data_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "/tmp/seerdb_scale_test".to_string());

    let configs = match scale.to_lowercase().as_str() {
        "1m" => vec![ScaleConfig {
            name: "1M Keys (Quick Test)",
            num_keys: 1_000_000,
            key_size: 16,
            value_size: 100,
            num_threads: 4,
            batch_size: 1000,
            sample_rate: 1_000,
        }],
        "10m" => vec![ScaleConfig {
            name: "10M Keys",
            num_keys: 10_000_000,
            key_size: 16,
            value_size: 100,
            num_threads: 8,
            batch_size: 1000,
            sample_rate: 10_000,
        }],
        "100m" => vec![ScaleConfig {
            name: "100M Keys",
            num_keys: 100_000_000,
            key_size: 16,
            value_size: 100,
            num_threads: 8,
            batch_size: 1000,
            sample_rate: 100_000,
        }],
        "500m" => vec![ScaleConfig {
            name: "500M Keys",
            num_keys: 500_000_000,
            key_size: 16,
            value_size: 100,
            num_threads: 8,
            batch_size: 1000,
            sample_rate: 500_000,
        }],
        "1b" | "billion" => vec![ScaleConfig {
            name: "1B Keys",
            num_keys: 1_000_000_000,
            key_size: 16,
            value_size: 100,
            num_threads: 8,
            batch_size: 1000,
            sample_rate: 1_000_000,
        }],
        "all" => vec![
            ScaleConfig {
                name: "100M Keys",
                num_keys: 100_000_000,
                key_size: 16,
                value_size: 100,
                num_threads: 8,
                batch_size: 1000,
                sample_rate: 100_000,
            },
            ScaleConfig {
                name: "500M Keys",
                num_keys: 500_000_000,
                key_size: 16,
                value_size: 100,
                num_threads: 8,
                batch_size: 1000,
                sample_rate: 500_000,
            },
            ScaleConfig {
                name: "1B Keys",
                num_keys: 1_000_000_000,
                key_size: 16,
                value_size: 100,
                num_threads: 8,
                batch_size: 1000,
                sample_rate: 1_000_000,
            },
        ],
        _ => {
            eprintln!("Usage: {} [1m|10m|100m|500m|1b|all] [data_dir]", args[0]);
            std::process::exit(1);
        }
    };

    let data_dir = PathBuf::from(&data_dir);
    println!("Data directory: {}", data_dir.display());
    println!("Scale: {}", scale);

    // Clean up any existing data
    if data_dir.exists() {
        println!("Cleaning existing data directory...");
        std::fs::remove_dir_all(&data_dir)?;
    }
    std::fs::create_dir_all(&data_dir)?;

    for config in &configs {
        // Clean between tests
        if data_dir.exists() {
            std::fs::remove_dir_all(&data_dir)?;
        }
        std::fs::create_dir_all(&data_dir)?;

        run_scale_test(config, &data_dir)?;
    }

    println!("\n\n[ok] All scale tests completed successfully!");
    Ok(())
}
