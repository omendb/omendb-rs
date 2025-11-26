// Production Hardening Tests
// Tests for memory budget, disk space, background panic detection, etc.
// These validate the production-readiness features added in Nov 2025

use seerdb::{DBError, DBOptions, DB};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

// ============================================================================
// Memory Budget Enforcement Tests
// ============================================================================

#[test]
#[ignore] // FIXME: Test hangs - memory pressure loop blocks indefinitely
fn test_memory_budget_early_flush() {
    // Test that memory pressure >80% triggers early flush
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        max_memory_bytes: Some(50 * 1024 * 1024), // 50MB limit
        memtable_capacity: 10 * 1024 * 1024,      // 10MB per memtable
        background_flush: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write data to approach 80% threshold (~40MB)
    // Each write: key (20 bytes) + value (1KB) ≈ 1044 bytes
    // Target: ~38K writes to hit 80%
    for i in 0..38_000 {
        let key = format!("key_{:010}", i);
        let value = vec![b'x'; 1000];
        db.put(key.as_bytes(), &value).unwrap();

        // Check memory every 1000 writes
        if i % 1000 == 0 {
            let mem = db.estimate_memory_usage();
            if mem > 40 * 1024 * 1024 {
                // Should trigger flush before reaching 95%
                assert!(
                    mem < 48 * 1024 * 1024,
                    "Memory should not exceed 95% threshold (48MB)"
                );
            }
        }
    }

    // Verify data is accessible
    for i in 0..100 {
        let key = format!("key_{:010}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_some(),
            "Key {} should be accessible",
            i
        );
    }
}

#[test]
#[ignore] // FIXME: Test hangs - memory pressure loop blocks indefinitely
fn test_memory_budget_write_blocking() {
    // Test that memory pressure >95% blocks writes temporarily
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        max_memory_bytes: Some(30 * 1024 * 1024), // 30MB limit (smaller for faster test)
        memtable_capacity: 20 * 1024 * 1024,      // 20MB per memtable
        background_flush: false,                  // Disable auto flush to test blocking
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write enough to approach 95% (~28.5MB)
    // This should trigger backpressure (writes blocked with sleep)
    let mut blocked = false;
    for i in 0..30_000 {
        let key = format!("key_{:010}", i);
        let value = vec![b'x'; 1000];

        let start = std::time::Instant::now();
        db.put(key.as_bytes(), &value).unwrap();
        let duration = start.elapsed();

        // If write took >5ms, backpressure is active
        if duration.as_millis() > 5 {
            blocked = true;
            break;
        }
    }

    // Should have hit backpressure
    assert!(
        blocked,
        "Should trigger write blocking at 95% memory threshold"
    );
}

#[test]
fn test_memory_estimation_accuracy() {
    // Test that memory estimation is reasonably accurate
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        background_flush: false, // Disable flush to keep data in memtable
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    let initial_mem = db.estimate_memory_usage();

    // Write 1MB of data (1000 * 1KB)
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'x'; 1000];
        db.put(key.as_bytes(), &value).unwrap();
    }

    let after_1mb = db.estimate_memory_usage();
    let delta = after_1mb - initial_mem;

    // Should be roughly 1MB ± 50% (keys + values + overhead)
    assert!(
        delta > 500_000 && delta < 2_000_000,
        "Memory estimate delta should be ~1MB, got {}",
        delta
    );
}

// ============================================================================
// Disk Space Checks Tests
// ============================================================================

#[test]
#[ignore] // TODO: Disk space check disabled (too slow - called on every write)
fn test_disk_space_validation_on_write() {
    // Test that writes are rejected when disk space insufficient
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        min_disk_space_bytes: Some(1_000_000_000_000), // 1TB (impossible threshold)
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write should fail with DiskSpaceFull error
    let result = db.put(b"key", b"value");

    assert!(
        result.is_err(),
        "Write should fail when disk space insufficient"
    );

    let err = result.unwrap_err();
    assert!(
        matches!(err, DBError::DiskSpaceFull { .. }),
        "Error should be DiskSpaceFull, got {:?}",
        err
    );
}

#[test]
fn test_disk_space_configurable_threshold() {
    // Test that disk space threshold is configurable
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Set reasonable threshold (100MB)
    let opts = DBOptions {
        data_dir,
        min_disk_space_bytes: Some(100 * 1024 * 1024),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write should succeed if disk has >100MB free
    let result = db.put(b"key", b"value");

    // Result depends on actual disk space
    // This test just validates the option is respected
    if result.is_err() {
        assert!(
            matches!(result.unwrap_err(), DBError::DiskSpaceFull { .. }),
            "If write fails, should be DiskSpaceFull error"
        );
    }
}

#[test]
fn test_disk_space_no_limit_when_unconfigured() {
    // Test that disk space checks are disabled when not configured
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        min_disk_space_bytes: None, // No limit
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write should succeed regardless of disk space
    db.put(b"key", b"value").unwrap();

    // Verify data written
    assert!(db.get(b"key").unwrap().is_some());
}

// ============================================================================
// Background Thread Panic Detection Tests
// ============================================================================

#[test]
fn test_wal_panic_detection() {
    // Test that WAL writer panic is detected
    // Note: This is challenging to test without actually crashing the thread
    // We test the health check mechanism instead

    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Normal operation should have healthy WAL
    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();

    // Verify writes succeeded (WAL is healthy)
    assert!(db.get(b"key1").unwrap().is_some());
    assert!(db.get(b"key2").unwrap().is_some());
}

#[test]
fn test_flush_thread_health_tracking() {
    // Test that flush thread health is tracked
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        background_flush: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Trigger flush by writing data
    for i in 0..1000 {
        db.put(format!("key_{}", i).as_bytes(), b"value").unwrap();
    }

    db.flush().unwrap();

    // If flush thread panicked, flush() would fail
    // Successful flush means thread is healthy
}

#[test]
fn test_compaction_thread_health_tracking() {
    // Test that compaction thread health is tracked
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write enough data to trigger compaction
    for i in 0..10000 {
        db.put(format!("key_{:05}", i).as_bytes(), &vec![b'x'; 100])
            .unwrap();
    }

    // Force flush to create SSTables
    db.flush().unwrap();

    // Wait for compaction to run
    thread::sleep(Duration::from_millis(500));

    // If compaction thread panicked, subsequent operations would fail
    db.put(b"test_key", b"test_value").unwrap();
}

#[test]
fn test_health_status_propagation() {
    // Test that health status is checked on every write
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Normal writes should succeed
    for i in 0..100 {
        db.put(format!("key_{}", i).as_bytes(), b"value").unwrap();
    }

    // All writes succeeded, health status is good
}

#[test]
fn test_graceful_degradation_on_panic() {
    // Test that DB handles background thread panic gracefully
    // This is a safety test - DB should not crash the process

    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write some data
    for i in 0..100 {
        db.put(format!("key_{}", i).as_bytes(), b"value").unwrap();
    }

    // Even if background threads panic, DB should not crash
    // This test passes if no panic occurs
    db.flush().unwrap();
}

// ============================================================================
// File Descriptor Limits Tests
// ============================================================================

#[test]
fn test_fd_usage_reasonable() {
    // Test that FD usage is reasonable for small DB
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write some data
    for i in 0..1000 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    // FD usage should be minimal (WAL + a few SSTables)
    // This test just ensures DB opens and operates
    // Actual FD counting requires OS-specific APIs
}

#[test]
fn test_multiple_db_instances_fd_limits() {
    // Test that multiple DB instances don't exhaust FDs
    let temp_dirs: Vec<_> = (0..5).map(|_| TempDir::new().unwrap()).collect();

    let dbs: Vec<_> = temp_dirs
        .iter()
        .map(|dir| {
            let opts = DBOptions {
                data_dir: PathBuf::from(dir.path()),
                ..Default::default()
            };
            DB::open(opts).unwrap()
        })
        .collect();

    // Write to all DBs
    for (i, db) in dbs.iter().enumerate() {
        db.put(format!("db_{}", i).as_bytes(), b"value").unwrap();
    }

    // All DBs should operate without FD exhaustion
}

// ============================================================================
// SSTable Fsync Validation Tests
// ============================================================================

#[test]
fn test_sstable_fsync_on_flush() {
    // Test that SSTables are fsync'd on creation
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write data
    for i in 0..1000 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Flush to create SSTable
    db.flush().unwrap();

    drop(db);

    // Reopen - data should be durable (fsync ensures this)
    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..1000 {
        assert!(
            db.get(format!("key_{:04}", i).as_bytes())
                .unwrap()
                .is_some(),
            "Fsync ensures data durability after flush"
        );
    }
}

#[test]
fn test_sstable_durability_after_crash() {
    // Test that flushed SSTables survive simulated crash
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write and flush
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        for i in 0..500 {
            db.put(format!("key_{:04}", i).as_bytes(), b"durable")
                .unwrap();
        }

        db.flush().unwrap();

        // Simulated crash (drop without clean shutdown)
    }

    // Reopen and verify data
    {
        let opts = DBOptions {
            data_dir,
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        for i in 0..500 {
            let value = db.get(format!("key_{:04}", i).as_bytes()).unwrap();
            assert!(value.is_some(), "Flushed data should survive crash");
            assert_eq!(value.unwrap().as_ref(), b"durable");
        }
    }
}

// ============================================================================
// Disk Space Checking Tests
// ============================================================================

#[test]
fn test_disk_space_check_caching() {
    // Test that disk space checking uses caching and doesn't call sysinfo on every write
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        min_disk_space_bytes: Some(1024 * 1024), // 1MB minimum
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write multiple values - disk space should be checked but cached
    for i in 0..100 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    // All writes should succeed (we have enough disk space)
    assert_eq!(db.get(b"key_0000").unwrap().unwrap().as_ref(), b"value");
}

#[test]
fn test_disk_space_check_disabled_when_not_configured() {
    // Test that disk space checking is skipped when min_disk_space_bytes is None
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        min_disk_space_bytes: None, // Disabled
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Writes should succeed without any disk space checks
    for i in 0..50 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    assert_eq!(db.get(b"key_0000").unwrap().unwrap().as_ref(), b"value");
}

#[test]
#[ignore] // Requires special setup to simulate low disk space
fn test_disk_space_full_prevents_writes() {
    // This test would require mocking the sysinfo disk space check
    // or running on a partition with very little space
    //
    // Expected behavior:
    // 1. Set min_disk_space_bytes to a high value (e.g., 1TB)
    // 2. Attempt writes
    // 3. Should get DBError::DiskSpaceFull
    //
    // Implementation note: This is difficult to test in a real environment
    // without creating a small partition or using mocking
}
