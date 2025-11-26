// Stress tests for cloud storage robustness
//
// Tests:
// - High concurrency (100+ parallel operations)
// - Retry configuration
// - Error propagation
// - Performance under load

#![cfg(feature = "object-store")]

use object_store::memory::InMemory;
use seerdb::{ObjectStoreBackend, RetryConfig, Storage};
use std::path::Path;
use std::sync::Arc;

/// Test high concurrency - 100 parallel write operations
#[test]
fn test_high_concurrency_writes() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = Arc::new(rt.block_on(async { ObjectStoreBackend::new(store, String::new()) }));

    let _guard = rt.enter();

    // Spawn 100 parallel write operations
    let mut handles = vec![];
    for i in 0..100 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let path_str = format!("concurrent_{}.sst", i);
            let path = Path::new(&path_str);
            let data = format!("data_{}", i);
            backend.write_sstable(path, data.as_bytes())
        });
        handles.push(handle);
    }

    // All writes should succeed
    let mut successes = 0;
    for handle in handles {
        if handle.join().unwrap().is_ok() {
            successes += 1;
        }
    }

    println!("High concurrency writes: {}/100 succeeded", successes);
    assert_eq!(successes, 100, "all concurrent writes should succeed");

    // Verify all files are readable
    let mut read_successes = 0;
    for i in 0..100 {
        let path_str = format!("concurrent_{}.sst", i);
        let path = Path::new(&path_str);
        if backend.exists(path).unwrap_or(false) {
            read_successes += 1;
        }
    }

    assert_eq!(read_successes, 100, "all written files should exist");
}

/// Test high concurrency reads - 200 parallel read operations on shared data
#[test]
fn test_high_concurrency_reads() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = Arc::new(rt.block_on(async { ObjectStoreBackend::new(store, String::new()) }));

    let _guard = rt.enter();

    // Write 10 files
    for i in 0..10 {
        let path_str = format!("shared_{}.sst", i);
        let path = Path::new(&path_str);
        let data = format!("shared_data_{}", i);
        backend.write_sstable(path, data.as_bytes()).unwrap();
    }

    // Spawn 200 parallel read operations (20 reads per file)
    let mut handles = vec![];
    for i in 0..200 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let file_idx = i % 10;
            let path_str = format!("shared_{}.sst", file_idx);
            let path = Path::new(&path_str);
            let expected = format!("shared_data_{}", file_idx);

            match backend.read_block(path, 0, expected.len() as u32) {
                Ok(data) => data == expected.as_bytes(),
                Err(_) => false,
            }
        });
        handles.push(handle);
    }

    // All reads should succeed
    let mut successes = 0;
    for handle in handles {
        if handle.join().unwrap() {
            successes += 1;
        }
    }

    println!("High concurrency reads: {}/200 succeeded", successes);
    assert_eq!(successes, 200, "all concurrent reads should succeed");
}

/// Test mixed read/write workload
#[test]
fn test_mixed_workload() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = Arc::new(rt.block_on(async { ObjectStoreBackend::new(store, String::new()) }));

    let _guard = rt.enter();

    // Pre-populate some files
    for i in 0..20 {
        let path_str = format!("mixed_{}.sst", i);
        let path = Path::new(&path_str);
        let data = format!("initial_data_{}", i);
        backend.write_sstable(path, data.as_bytes()).unwrap();
    }

    // Spawn mixed operations: 50 reads, 30 writes, 20 deletes
    let mut handles = vec![];

    // 50 reads
    for i in 0..50 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let file_idx = i % 20;
            let path_str = format!("mixed_{}.sst", file_idx);
            let path = Path::new(&path_str);
            backend.exists(path).is_ok()
        });
        handles.push(handle);
    }

    // 30 writes
    for i in 100..130 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let path_str = format!("mixed_{}.sst", i);
            let path = Path::new(&path_str);
            let data = format!("new_data_{}", i);
            backend.write_sstable(path, data.as_bytes()).is_ok()
        });
        handles.push(handle);
    }

    // 20 deletes
    for i in 0..20 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let path_str = format!("mixed_{}.sst", i);
            let path = Path::new(&path_str);
            backend.delete_sstable(path).is_ok()
        });
        handles.push(handle);
    }

    // Count successes
    let mut successes = 0;
    for handle in handles {
        if handle.join().unwrap() {
            successes += 1;
        }
    }

    println!("Mixed workload: {}/100 operations succeeded", successes);
    assert!(
        successes >= 90,
        "at least 90% of mixed operations should succeed"
    );
}

/// Test RetryConfig presets
#[test]
fn test_retry_config_presets() {
    // Test default config
    let default = RetryConfig::default();
    assert_eq!(default.max_attempts, 3);
    assert_eq!(default.base_delay_ms, 100);
    assert_eq!(default.max_delay_ms, 5000);
    assert!(default.jitter);

    // Test no-retry config
    let none = RetryConfig::none();
    assert_eq!(none.max_attempts, 0);

    // Test aggressive config
    let aggressive = RetryConfig::aggressive();
    assert_eq!(aggressive.max_attempts, 5);
    assert_eq!(aggressive.base_delay_ms, 50);
    assert_eq!(aggressive.max_delay_ms, 10000);
}

/// Test custom retry configuration
#[test]
fn test_custom_retry_config() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let custom_config = RetryConfig {
        max_attempts: 10,
        base_delay_ms: 50,
        max_delay_ms: 2000,
        jitter: false,
    };

    let backend = rt.block_on(async {
        ObjectStoreBackend::with_retry_config(store, String::new(), custom_config)
    });

    let _guard = rt.enter();

    // Basic operations should work with custom config
    let path = Path::new("custom_config.sst");
    let data = b"test data";

    backend.write_sstable(path, data).unwrap();
    assert!(backend.exists(path).unwrap());

    let read_data = backend.read_block(path, 0, data.len() as u32).unwrap();
    assert_eq!(&read_data, data);
}

/// Test list operations under load
#[test]
fn test_list_operations_stress() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = Arc::new(rt.block_on(async { ObjectStoreBackend::new(store, String::new()) }));

    let _guard = rt.enter();

    // Write 50 files
    for i in 0..50 {
        let path_str = format!("list_test_{}.sst", i);
        let path = Path::new(&path_str);
        let data = format!("data_{}", i);
        backend.write_sstable(path, data.as_bytes()).unwrap();
    }

    // Perform 20 parallel list operations
    let mut handles = vec![];
    for _ in 0..20 {
        let backend = backend.clone();
        let handle = std::thread::spawn(move || {
            let list_result = backend.list_sstables(Path::new(""));
            match list_result {
                Ok(files) => files.len() >= 50, // >= because other tests may have written files
                Err(_) => false,
            }
        });
        handles.push(handle);
    }

    // All list operations should succeed
    let mut successes = 0;
    for handle in handles {
        if handle.join().unwrap() {
            successes += 1;
        }
    }

    println!("List stress test: {}/20 succeeded", successes);
    assert_eq!(successes, 20, "all list operations should succeed");
}

/// Test operations on non-existent files
#[test]
fn test_nonexistent_file_operations() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = rt.block_on(async { ObjectStoreBackend::new(store, String::new()) });

    let _guard = rt.enter();

    let path = Path::new("nonexistent.sst");

    // exists() should return false, not error
    assert_eq!(backend.exists(path).unwrap(), false);

    // read_block() should error
    assert!(backend.read_block(path, 0, 100).is_err());

    // delete() on non-existent file may succeed (idempotent) or error depending on backend
    // We don't assert on this as behavior varies
}

/// Benchmark: Measure operation throughput
#[test]
#[ignore] // Run with --ignored flag for benchmarks
fn bench_write_throughput() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let store = Arc::new(InMemory::new());

    let backend = rt.block_on(async { ObjectStoreBackend::new(store, String::new()) });

    let _guard = rt.enter();

    let start = std::time::Instant::now();
    let count = 1000;

    for i in 0..count {
        let path_str = format!("bench_{}.sst", i);
        let path = Path::new(&path_str);
        let data = vec![0u8; 4096]; // 4KB per file
        backend.write_sstable(path, &data).unwrap();
    }

    let elapsed = start.elapsed();
    let ops_per_sec = count as f64 / elapsed.as_secs_f64();

    println!(
        "Write throughput: {:.0} ops/sec ({} files in {:?})",
        ops_per_sec, count, elapsed
    );
}
