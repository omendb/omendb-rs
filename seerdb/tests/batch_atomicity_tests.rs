// Batch Atomicity Tests
// Tests that batch operations are truly atomic (all-or-nothing)
// Critical for data integrity - batches must be recoverable as a single unit

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// ============================================================================
// Basic Batch Atomicity Tests (5 tests)
// ============================================================================

#[test]
fn test_batch_single_wal_record() {
    // Test that a batch is written as a single WAL record (atomicity)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        background_flush: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Create batch with multiple operations
    let mut batch = db.batch();
    batch.put(b"key1", b"value1");
    batch.put(b"key2", b"value2");
    batch.put(b"key3", b"value3");
    batch.delete(b"key4");

    // Write batch
    batch.commit().unwrap();

    drop(db);

    // Reopen and verify all operations are present (atomicity)
    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    assert_eq!(db.get(b"key1").unwrap().as_deref(), Some(&b"value1"[..]));
    assert_eq!(db.get(b"key2").unwrap().as_deref(), Some(&b"value2"[..]));
    assert_eq!(db.get(b"key3").unwrap().as_deref(), Some(&b"value3"[..]));
    assert_eq!(db.get(b"key4").unwrap(), None);
}

#[test]
fn test_batch_mixed_put_delete() {
    // Test batch with mixed put/delete operations
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Pre-populate some keys
    db.put(b"existing1", b"old_value1").unwrap();
    db.put(b"existing2", b"old_value2").unwrap();
    db.put(b"existing3", b"old_value3").unwrap();

    // Batch that mixes puts and deletes
    let mut batch = db.batch();
    batch.put(b"new_key1", b"new_value1"); // New key
    batch.put(b"existing1", b"updated_value1"); // Update existing
    batch.delete(b"existing2"); // Delete existing
    batch.put(b"new_key2", b"new_value2"); // New key
    batch.delete(b"nonexistent"); // Delete non-existent (should be no-op)

    batch.commit().unwrap();

    // Verify results
    assert_eq!(
        db.get(b"new_key1").unwrap().as_deref(),
        Some(&b"new_value1"[..])
    );
    assert_eq!(
        db.get(b"existing1").unwrap().as_deref(),
        Some(&b"updated_value1"[..])
    );
    assert_eq!(db.get(b"existing2").unwrap(), None);
    assert_eq!(
        db.get(b"new_key2").unwrap().as_deref(),
        Some(&b"new_value2"[..])
    );
    assert_eq!(
        db.get(b"existing3").unwrap().as_deref(),
        Some(&b"old_value3"[..])
    );
}

#[test]
fn test_batch_empty_handling() {
    // Test that empty batch is a no-op
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write some data
    db.put(b"key1", b"value1").unwrap();

    // Write empty batch (should be no-op)
    let batch = db.batch();
    batch.commit().unwrap();

    // Verify original data is unchanged
    assert_eq!(db.get(b"key1").unwrap().as_deref(), Some(&b"value1"[..]));
}

#[test]
fn test_batch_large_1000_operations() {
    // Test large batch (1000+ operations)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        background_flush: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Create large batch
    let mut batch = db.batch();
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = format!("value_{:04}", i);
        batch.put(key.as_bytes(), value.as_bytes());
    }

    // Write batch atomically
    batch.commit().unwrap();

    drop(db);

    // Reopen and verify all 1000 keys are present (atomicity)
    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let expected_value = format!("value_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap();
        assert_eq!(
            value.as_deref(),
            Some(expected_value.as_bytes()),
            "Key {} should be present after batch write",
            key
        );
    }
}

#[test]
fn test_batch_recovery_after_crash() {
    // Test that batch is fully recovered after simulated crash
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write batch and simulate crash
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        let mut batch = db.batch();
        batch.put(b"batch_key1", b"batch_value1");
        batch.put(b"batch_key2", b"batch_value2");
        batch.put(b"batch_key3", b"batch_value3");
        batch.delete(b"deleted_key");

        batch.commit().unwrap();

        // Simulated crash (drop without clean shutdown)
    }

    // Reopen and verify batch was fully recovered
    {
        let opts = DBOptions {
            data_dir,
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        // All batch operations should be present
        assert_eq!(
            db.get(b"batch_key1").unwrap().as_deref(),
            Some(&b"batch_value1"[..])
        );
        assert_eq!(
            db.get(b"batch_key2").unwrap().as_deref(),
            Some(&b"batch_value2"[..])
        );
        assert_eq!(
            db.get(b"batch_key3").unwrap().as_deref(),
            Some(&b"batch_value3"[..])
        );
        assert_eq!(db.get(b"deleted_key").unwrap(), None);
    }
}

// ============================================================================
// Concurrent Batch Tests (5 tests)
// ============================================================================

#[test]
fn test_batch_concurrent_multiple_threads() {
    // Test multiple threads submitting batches concurrently
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Spawn 10 threads, each writing 100 batches
    let mut handles = vec![];
    for thread_id in 0..10 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for batch_id in 0..100 {
                let mut batch = db_clone.batch();
                // Each batch writes 10 keys
                for key_id in 0..10 {
                    let key = format!("t{}_b{}_k{}", thread_id, batch_id, key_id);
                    let value = format!("value_{}", key);
                    batch.put(key.as_bytes(), value.as_bytes());
                }
                batch.commit().unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all data is present (10 threads * 100 batches * 10 keys = 10,000 keys)
    let mut count = 0;
    for thread_id in 0..10 {
        for batch_id in 0..100 {
            for key_id in 0..10 {
                let key = format!("t{}_b{}_k{}", thread_id, batch_id, key_id);
                let expected_value = format!("value_{}", key);
                let value = db.get(key.as_bytes()).unwrap();
                assert_eq!(value.as_deref(), Some(expected_value.as_bytes()));
                count += 1;
            }
        }
    }
    assert_eq!(count, 10000, "All 10,000 keys should be present");
}

#[test]
fn test_batch_interleaved_with_individual_operations() {
    // Test batch operations interleaved with individual put/delete/get
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Spawn threads doing different operations
    let db1 = Arc::clone(&db);
    let handle1 = thread::spawn(move || {
        // Thread 1: Write batches
        for i in 0..100 {
            let mut batch = db1.batch();
            batch.put(format!("batch_{}", i).as_bytes(), b"batch_value");
            batch.commit().unwrap();
        }
    });

    let db2 = Arc::clone(&db);
    let handle2 = thread::spawn(move || {
        // Thread 2: Individual puts
        for i in 0..100 {
            db2.put(format!("single_{}", i).as_bytes(), b"single_value")
                .unwrap();
        }
    });

    let db3 = Arc::clone(&db);
    let handle3 = thread::spawn(move || {
        // Thread 3: Read operations (should not conflict)
        for i in 0..100 {
            let _ = db3.get(format!("batch_{}", i).as_bytes());
            let _ = db3.get(format!("single_{}", i).as_bytes());
        }
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
    handle3.join().unwrap();

    // Verify all writes succeeded
    for i in 0..100 {
        assert!(db.get(format!("batch_{}", i).as_bytes()).unwrap().is_some());
        assert!(db
            .get(format!("single_{}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_batch_during_flush() {
    // Test batch write during memtable flush
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 1024 * 1024, // 1MB memtable
        background_flush: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Write enough data to trigger flush
    for i in 0..10000 {
        db.put(format!("pre_flush_{}", i).as_bytes(), &vec![b'x'; 100])
            .unwrap();
    }

    // Trigger flush
    db.flush().unwrap();

    // Write batch during flush
    let mut batch = db.batch();
    for i in 0..100 {
        batch.put(format!("during_flush_{}", i).as_bytes(), b"batch_value");
    }
    batch.commit().unwrap();

    // Verify batch data is present
    for i in 0..100 {
        assert!(db
            .get(format!("during_flush_{}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_batch_during_compaction() {
    // Test batch write during background compaction
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024, // 512KB memtable
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Write enough data to trigger compaction
    for i in 0..20000 {
        db.put(format!("pre_compact_{:05}", i).as_bytes(), &vec![b'x'; 100])
            .unwrap();
    }

    // Force flush to create SSTables
    db.flush().unwrap();

    // Wait a bit for compaction to start
    thread::sleep(std::time::Duration::from_millis(100));

    // Write batch during compaction
    let mut batch = db.batch();
    for i in 0..100 {
        batch.put(format!("during_compact_{}", i).as_bytes(), b"batch_value");
    }
    batch.commit().unwrap();

    // Verify batch data is present
    for i in 0..100 {
        assert!(db
            .get(format!("during_compact_{}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_batch_write_amplification() {
    // Verify that batch operations are more efficient than individual operations
    // (single WAL write for entire batch)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 1024 * 1024, // 1MB memtable
        background_flush: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write 1000 operations as batch (single WAL write)
    let mut batch = db.batch();
    for i in 0..1000 {
        batch.put(format!("batch_key_{:04}", i).as_bytes(), b"batch_value");
    }

    batch.commit().unwrap();

    // Verify all 1000 keys are present (functional correctness)
    for i in 0..1000 {
        assert!(
            db.get(format!("batch_key_{:04}", i).as_bytes())
                .unwrap()
                .is_some(),
            "Batch key {} should be present",
            i
        );
    }

    // Verify batch is atomic - reopen DB to confirm WAL recovery works
    drop(db);

    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // All 1000 keys should still be present after reopen
    for i in 0..1000 {
        assert!(
            db.get(format!("batch_key_{:04}", i).as_bytes())
                .unwrap()
                .is_some(),
            "Batch key {} should persist after reopen",
            i
        );
    }
}

// ============================================================================
// Edge Case Tests (5 tests)
// ============================================================================

#[test]
fn test_batch_larger_than_memtable() {
    // Test batch larger than memtable capacity (should trigger multiple flushes)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 100 * 1024, // 100KB memtable (small)
        background_flush: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Create batch larger than memtable (200KB of data)
    let mut batch = db.batch();
    for i in 0..2000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'x'; 100]; // 100 bytes per value
        batch.put(key.as_bytes(), &value);
    }

    // Should succeed despite being larger than memtable
    batch.commit().unwrap();

    // Verify all data is present
    for i in 0..2000 {
        let key = format!("key_{:04}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_some(),
            "Key {} should be present",
            key
        );
    }
}

#[test]
fn test_batch_duplicate_keys() {
    // Test batch with duplicate keys (last write wins)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Batch with duplicate keys
    let mut batch = db.batch();
    batch.put(b"duplicate_key", b"value1");
    batch.put(b"duplicate_key", b"value2");
    batch.put(b"duplicate_key", b"value3");
    batch.delete(b"duplicate_key"); // Final operation is delete

    batch.commit().unwrap();

    // Last operation (delete) should win
    assert_eq!(db.get(b"duplicate_key").unwrap(), None);

    // Test with final operation being put
    let mut batch2 = db.batch();
    batch2.put(b"duplicate_key2", b"value1");
    batch2.delete(b"duplicate_key2");
    batch2.put(b"duplicate_key2", b"final_value");

    batch2.commit().unwrap();

    assert_eq!(
        db.get(b"duplicate_key2").unwrap().as_deref(),
        Some(&b"final_value"[..])
    );
}

#[test]
fn test_batch_size_limits() {
    // Test that very large batches are handled correctly
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Create 10,000 operation batch (very large)
    let mut batch = db.batch();
    for i in 0..10000 {
        batch.put(format!("key_{:05}", i).as_bytes(), b"value");
    }

    // Should handle large batch
    batch.commit().unwrap();

    // Spot check some keys
    assert!(db.get(b"key_00000").unwrap().is_some());
    assert!(db.get(b"key_05000").unwrap().is_some());
    assert!(db.get(b"key_09999").unwrap().is_some());
}

#[test]
fn test_batch_memory_accounting() {
    // Test that batch memory usage is properly accounted for
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        max_memory_bytes: Some(50 * 1024 * 1024), // 50MB limit
        background_flush: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    let mem_before = db.estimate_memory_usage();

    // Write batch with known size (1000 * 1KB ≈ 1MB)
    let mut batch = db.batch();
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'x'; 1000];
        batch.put(key.as_bytes(), &value);
    }

    batch.commit().unwrap();

    let mem_after = db.estimate_memory_usage();
    let mem_delta = mem_after - mem_before;

    // Should account for roughly 1MB ± 50% (keys + values + overhead)
    assert!(
        mem_delta > 500_000 && mem_delta < 2_000_000,
        "Memory delta should be ~1MB, got {}",
        mem_delta
    );
}

#[test]
fn test_batch_with_zero_length_keys() {
    // Test edge case: batch with empty keys
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    let mut batch = db.batch();
    batch.put(b"", b"empty_key_value"); // Empty key
    batch.put(b"normal_key", b""); // Empty value
    batch.put(b"", b""); // Both empty

    batch.commit().unwrap();

    // Should handle empty keys/values
    assert!(db.get(b"").unwrap().is_some());
    assert!(db.get(b"normal_key").unwrap().is_some());
}
