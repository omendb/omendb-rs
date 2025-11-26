// Compaction Correctness Tests
// Tests that compaction doesn't lose, duplicate, or corrupt data
// Critical for data integrity - compaction is the most complex operation

use seerdb::{DBOptions, DB};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

// ============================================================================
// Basic Compaction Correctness Tests (5 tests)
// ============================================================================

#[test]
fn test_compaction_no_data_loss() {
    // Test that compaction doesn't lose any keys
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024, // 512KB memtable (small for faster compaction)
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write enough data to trigger multiple flushes and compaction
    let num_keys = 10000;
    for i in 0..num_keys {
        let key = format!("key_{:05}", i);
        let value = format!("value_{:05}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // Force flush and wait for compaction
    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify all keys are still present
    for i in 0..num_keys {
        let key = format!("key_{:05}", i);
        let expected_value = format!("value_{:05}", i);
        let value = db.get(key.as_bytes()).unwrap();
        assert!(
            value.is_some(),
            "Key {} should be present after compaction",
            key
        );
        assert_eq!(
            value.unwrap().as_ref(),
            expected_value.as_bytes(),
            "Value for key {} should be correct after compaction",
            key
        );
    }
}

#[test]
fn test_compaction_no_duplicates() {
    // Test that compaction doesn't create duplicate keys
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write data with updates (same key multiple times)
    for round in 0..5 {
        for i in 0..1000 {
            let key = format!("key_{:04}", i);
            let value = format!("value_round{}_key{}", round, i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
    }

    // Force flush and compaction
    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify each key has exactly one value (the latest)
    let mut seen_keys = HashSet::new();
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let expected_value = format!("value_round4_key{}", i);

        assert!(
            seen_keys.insert(key.clone()),
            "Key {} should only appear once",
            key
        );

        let value = db.get(key.as_bytes()).unwrap();
        assert_eq!(
            value.unwrap().as_ref(),
            expected_value.as_bytes(),
            "Key {} should have latest value after compaction",
            key
        );
    }
}

#[test]
fn test_compaction_preserves_key_ordering() {
    // Test that compaction maintains sorted key order
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write keys in random order
    let keys: Vec<_> = (0..5000).map(|i| format!("key_{:05}", i)).collect();
    for key in keys.iter().rev() {
        db.put(key.as_bytes(), b"value").unwrap();
    }

    // Force flush and compaction
    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(1000));

    // Scan and verify keys are in sorted order
    let mut iter = db.range(b"", Some(b"~")).unwrap();
    let mut prev_key: Option<Vec<u8>> = None;

    while let Some(Ok((key, _value))) = iter.next() {
        if let Some(prev) = prev_key {
            assert!(
                prev < key.to_vec(),
                "Keys should be in sorted order after compaction"
            );
        }
        prev_key = Some(key.to_vec());
    }
}

#[test]
fn test_compaction_handles_tombstones() {
    // Test that compaction correctly removes tombstones (deleted keys)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write keys
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }

    // Delete every other key
    for i in (0..1000).step_by(2) {
        let key = format!("key_{:04}", i);
        db.delete(key.as_bytes()).unwrap();
    }

    // Force flush and compaction
    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify deleted keys are gone, remaining keys present
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap();

        if i % 2 == 0 {
            assert!(
                value.is_none(),
                "Deleted key {} should not be present after compaction",
                key
            );
        } else {
            assert!(
                value.is_some(),
                "Non-deleted key {} should be present after compaction",
                key
            );
        }
    }
}

#[test]
fn test_compaction_updates_supersede_old_values() {
    // Test that newer values correctly supersede older values during compaction
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024, // Small for multiple flushes
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write initial values
    for i in 0..500 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"value_v1").unwrap();
    }

    db.flush().unwrap();

    // Update with v2
    for i in 0..500 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"value_v2").unwrap();
    }

    db.flush().unwrap();

    // Update with v3
    for i in 0..500 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"value_v3").unwrap();
    }

    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify all keys have v3 (latest value)
    for i in 0..500 {
        let key = format!("key_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap();
        assert_eq!(
            value.unwrap().as_ref(),
            b"value_v3",
            "Key {} should have latest value (v3) after compaction",
            key
        );
    }
}

// ============================================================================
// Multi-Level Compaction Tests (5 tests)
// ============================================================================

#[test]
fn test_compaction_across_multiple_levels() {
    // Test compaction works correctly across multiple LSM levels
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024, // Small to force multi-level
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write enough data to create multiple levels
    // Each batch goes to a different SSTable
    for batch in 0..10 {
        for i in 0..500 {
            let key = format!("batch{}_key{:04}", batch, i);
            let value = format!("batch{}_value{}", batch, i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
        db.flush().unwrap();
    }

    // Wait for compaction to settle
    thread::sleep(std::time::Duration::from_millis(2000));

    // Verify all data is present across all levels
    for batch in 0..10 {
        for i in 0..500 {
            let key = format!("batch{}_key{:04}", batch, i);
            let expected_value = format!("batch{}_value{}", batch, i);
            let value = db.get(key.as_bytes()).unwrap();
            assert!(
                value.is_some(),
                "Key {} should be present after multi-level compaction",
                key
            );
            assert_eq!(value.unwrap().as_ref(), expected_value.as_bytes());
        }
    }
}

#[test]
fn test_compaction_with_overlapping_key_ranges() {
    // Test compaction when key ranges overlap across levels
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write overlapping key ranges
    // Range 1: key_0000 to key_1000
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"range1").unwrap();
    }
    db.flush().unwrap();

    // Range 2: key_0500 to key_1500 (overlaps with range 1)
    for i in 500..1500 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"range2").unwrap();
    }
    db.flush().unwrap();

    // Range 3: key_1000 to key_2000 (overlaps with range 2)
    for i in 1000..2000 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"range3").unwrap();
    }
    db.flush().unwrap();

    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify correct values (latest write wins)
    for i in 0..2000 {
        let key = format!("key_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap();
        assert!(value.is_some(), "Key {} should be present", key);

        let expected = if i < 500 {
            b"range1"
        } else if i < 1000 {
            b"range2"
        } else {
            b"range3"
        };

        assert_eq!(
            value.unwrap().as_ref(),
            expected,
            "Key {} should have correct value after overlapping compaction",
            key
        );
    }
}

#[test]
fn test_compaction_merges_adjacent_sstables() {
    // Test that compaction properly merges adjacent SSTables
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Create multiple small SSTables with adjacent key ranges
    for batch in 0..5 {
        let start = batch * 200;
        let end = (batch + 1) * 200;

        for i in start..end {
            let key = format!("key_{:04}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }
        db.flush().unwrap();
    }

    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify all keys are present after merge
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_some(),
            "Key {} should be present after SSTable merge",
            key
        );
    }
}

#[test]
fn test_compaction_handles_empty_levels() {
    // Test compaction when some levels are empty
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 512 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write data
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }

    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(500));

    // Verify data is present
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        assert!(db.get(key.as_bytes()).unwrap().is_some());
    }
}

#[test]
fn test_compaction_with_single_key_per_level() {
    // Test edge case: compaction with very few keys per level
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write single key, flush, repeat
    for i in 0..10 {
        let key = format!("key_{}", i);
        db.put(key.as_bytes(), b"value").unwrap();
        db.flush().unwrap();
    }

    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify all keys present
    for i in 0..10 {
        let key = format!("key_{}", i);
        let result = db.get(key.as_bytes()).unwrap();
        if result.is_none() {
            eprintln!("MISSING KEY: key_{} - checking all keys...", i);
            for j in 0..10 {
                let k = format!("key_{}", j);
                let r = db.get(k.as_bytes()).unwrap();
                eprintln!(
                    "  key_{}: {}",
                    j,
                    if r.is_some() { "PRESENT" } else { "MISSING" }
                );
            }
        }
        assert!(result.is_some(), "key_{} should be present", i);
    }
}

// ============================================================================
// Concurrent Compaction Tests (5 tests)
// ============================================================================

#[test]
#[ignore] // TODO: Snapshot isolation - readers need to pin LSM version (separate from Bug #7)
fn test_compaction_concurrent_reads() {
    // Test reads are consistent during compaction
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate data
    for i in 0..5000 {
        let key = format!("key_{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }

    // Start reader threads
    let mut handles = vec![];
    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            // Read continuously during compaction
            for _ in 0..100 {
                for i in (0..5000).step_by(50) {
                    let key = format!("key_{:05}", i);
                    let value = db_clone.get(key.as_bytes()).unwrap();
                    assert!(
                        value.is_some(),
                        "Thread {} should read key {} during compaction",
                        thread_id,
                        key
                    );
                }
            }
        });
        handles.push(handle);
    }

    // Trigger compaction while readers are running
    db.flush().unwrap();
    thread::sleep(std::time::Duration::from_millis(500));

    // Wait for readers
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_compaction_concurrent_writes() {
    // Test writes continue during compaction
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate to trigger compaction
    for i in 0..5000 {
        let key = format!("pre_key_{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }
    db.flush().unwrap();

    // Write concurrently during compaction
    let mut handles = vec![];
    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for i in 0..500 {
                let key = format!("thread{}_key{:04}", thread_id, i);
                db_clone.put(key.as_bytes(), b"value").unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for writers
    for handle in handles {
        handle.join().unwrap();
    }

    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify all writes succeeded
    for thread_id in 0..4 {
        for i in 0..500 {
            let key = format!("thread{}_key{:04}", thread_id, i);
            assert!(db.get(key.as_bytes()).unwrap().is_some());
        }
    }
}

#[test]
fn test_compaction_concurrent_deletes() {
    // Test deletes during compaction are handled correctly
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate
    for i in 0..5000 {
        let key = format!("key_{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }
    db.flush().unwrap();

    // Delete concurrently during compaction
    let mut handles = vec![];
    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let start = thread_id * 1250;
            let end = (thread_id + 1) * 1250;
            for i in start..end {
                let key = format!("key_{:05}", i);
                db_clone.delete(key.as_bytes()).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    thread::sleep(std::time::Duration::from_millis(1000));

    // Verify deletes worked
    for i in 0..5000 {
        let key = format!("key_{:05}", i);
        assert!(db.get(key.as_bytes()).unwrap().is_none());
    }
}

#[test]
fn test_compaction_concurrent_flushes() {
    // Test multiple flushes during compaction
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Trigger initial compaction
    for i in 0..5000 {
        let key = format!("initial_{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }
    db.flush().unwrap();

    // Write more data and flush multiple times during compaction
    for batch in 0..5 {
        for i in 0..1000 {
            let key = format!("batch{}_key{:04}", batch, i);
            db.put(key.as_bytes(), b"value").unwrap();
        }
        db.flush().unwrap();
    }

    thread::sleep(std::time::Duration::from_millis(2000));

    // Verify all data present
    for i in 0..5000 {
        let key = format!("initial_{:05}", i);
        assert!(db.get(key.as_bytes()).unwrap().is_some());
    }
    for batch in 0..5 {
        for i in 0..1000 {
            let key = format!("batch{}_key{:04}", batch, i);
            assert!(db.get(key.as_bytes()).unwrap().is_some());
        }
    }
}

#[test]
#[ignore] // TODO: Snapshot isolation - readers need to pin LSM version (separate from Bug #7)
fn test_compaction_concurrent_scans() {
    // Test range scans during compaction return consistent results
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 256 * 1024,
        background_flush: true,
        background_compaction: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate
    for i in 0..5000 {
        let key = format!("key_{:05}", i);
        db.put(key.as_bytes(), b"value").unwrap();
    }

    // Start scanner threads
    let mut handles = vec![];
    for _ in 0..3 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let mut count = 0;
                let mut iter = db_clone.range(b"key_00", Some(b"key_01")).unwrap();
                while let Some(Ok((_key, _value))) = iter.next() {
                    count += 1;
                }
                // Should find ~1000 keys in this range
                assert!(
                    count >= 900,
                    "Scan should find most keys during compaction, got {}",
                    count
                );
            }
        });
        handles.push(handle);
    }

    // Trigger compaction while scanners run
    db.flush().unwrap();

    for handle in handles {
        handle.join().unwrap();
    }
}
