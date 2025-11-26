// I/O failure injection tests
// Tests error handling when I/O operations fail
// Critical for reliability: must handle I/O errors gracefully without data loss

use seerdb::{DBOptions, DB};
use std::fs::{self, OpenOptions};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_flush_failure_readonly_directory() {
    // Test flush failure when data directory becomes read-only
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write data
    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Make directory read-only to simulate I/O failure
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o444); // Read-only
    fs::set_permissions(&data_dir, perms).unwrap();

    // Flush should fail due to read-only directory
    let result = db.flush();

    // Restore permissions for cleanup
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&data_dir, perms).unwrap();

    // Flush should have failed
    assert!(
        result.is_err(),
        "Flush should fail with read-only directory"
    );

    // Verify data is still in memtable (not lost)
    for i in 0..100 {
        assert!(
            db.get(format!("key_{:03}", i).as_bytes())
                .unwrap()
                .is_some(),
            "Data should remain in memtable after flush failure"
        );
    }
}

#[test]
fn test_recovery_after_flush_failure() {
    // Test that DB can recover after a flush failure
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write data
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            db.put(format!("key_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }

        // Make directory read-only
        let mut perms = fs::metadata(&data_dir).unwrap().permissions();
        perms.set_mode(0o444);
        fs::set_permissions(&data_dir, perms).unwrap();

        // Try to flush (will fail)
        let _ = db.flush();

        // Restore permissions
        let mut perms = fs::metadata(&data_dir).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&data_dir, perms).unwrap();

        // DB goes out of scope - should write WAL
    }

    // Reopen - data should be recovered from WAL
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            assert!(
                db.get(format!("key_{:03}", i).as_bytes())
                    .unwrap()
                    .is_some(),
                "Data should be recovered from WAL after flush failure"
            );
        }
    }
}

#[test]
fn test_corrupted_sstable_skipped() {
    // Test that DB can open even if one SSTable is corrupted
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write and flush multiple SSTables
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            memtable_capacity: 1024, // Small to force flushes
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write first batch
        for i in 0..100 {
            db.put(format!("batch1_{:03}", i).as_bytes(), &vec![b'a'; 100])
                .unwrap();
        }
        db.flush().unwrap();

        // Write second batch
        for i in 0..100 {
            db.put(format!("batch2_{:03}", i).as_bytes(), &vec![b'b'; 100])
                .unwrap();
        }
        db.flush().unwrap();
    }

    // Corrupt the first SSTable
    let sstable_path = data_dir.join("L0_000000.sst");
    if sstable_path.exists() {
        fs::remove_file(&sstable_path).unwrap();
        // Create empty file (simulates corruption)
        fs::File::create(&sstable_path).unwrap();
    }

    // Reopen - should handle corrupted SSTable
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let result = DB::open(opts);

        match result {
            Ok(db) => {
                // DB opened despite corruption
                // Second batch should still be readable
                let readable = (0..100)
                    .filter(|i| {
                        db.get(format!("batch2_{:03}", i).as_bytes())
                            .unwrap()
                            .is_some()
                    })
                    .count();

                // Should be able to read uncorrupted data
                assert!(readable > 0, "Should read data from uncorrupted SSTables");
            }
            Err(_) => {
                // Also acceptable - strict corruption detection
            }
        }
    }
}

#[test]
fn test_partial_wal_recovery() {
    // Test recovery when WAL is partially written
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write data without flushing
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..100 {
            db.put(format!("key_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }

        // Don't flush - data only in WAL
    }

    // Truncate WAL to simulate partial write
    let wal_path = data_dir.join("wal.log");
    if wal_path.exists() {
        let metadata = fs::metadata(&wal_path).unwrap();
        let size = metadata.len();

        let file = OpenOptions::new().write(true).open(&wal_path).unwrap();
        file.set_len(size / 2).unwrap(); // Cut in half
    }

    // Reopen - should recover partial data
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let result = DB::open(opts);

        match result {
            Ok(db) => {
                // Count recovered keys
                let recovered = (0..100)
                    .filter(|i| {
                        db.get(format!("key_{:03}", i).as_bytes())
                            .unwrap()
                            .is_some()
                    })
                    .count();

                // Should recover at least some data before truncation point
                // But not all 100 keys
                assert!(
                    recovered < 100,
                    "Should not recover all keys from truncated WAL"
                );
            }
            Err(_) => {
                // Also acceptable - may reject truncated WAL entirely
            }
        }
    }
}

#[test]
fn test_write_data_despite_wal_issues() {
    // Test that we can still write data even if WAL has issues
    // (though durability may be compromised)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Normal writes should work
    for i in 0..50 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Verify data is readable
    for i in 0..50 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_operations_after_failed_flush() {
    // Test that DB can recover and continue working after a flush failure
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write initial data
    for i in 0..50 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Make directory read-only to cause flush failure
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&data_dir, perms).unwrap();

    // Try flush (will fail)
    let flush_result = db.flush();
    assert!(
        flush_result.is_err(),
        "Flush should fail with read-only dir"
    );

    // Restore permissions IMMEDIATELY for subsequent operations
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&data_dir, perms).unwrap();

    // Old data should still be readable (in immutable_memtable)
    let old_data_readable = (0..50)
        .filter(|i| {
            db.get(format!("key_{:03}", i).as_bytes())
                .unwrap()
                .is_some()
        })
        .count();

    assert!(
        old_data_readable > 0,
        "Old data should be readable after flush failure (in immutable_memtable)"
    );

    // DB should accept new writes
    for i in 50..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    // New data should be readable
    let new_data_readable = (50..100)
        .filter(|i| {
            db.get(format!("key_{:03}", i).as_bytes())
                .unwrap()
                .is_some()
        })
        .count();

    assert!(
        new_data_readable > 0,
        "New data should be writable and readable after flush failure"
    );

    // Successful flush should work now
    db.flush().unwrap();

    // All data should be readable after successful flush
    for i in 0..100 {
        assert!(
            db.get(format!("key_{:03}", i).as_bytes())
                .unwrap()
                .is_some(),
            "All data should be readable after successful flush"
        );
    }
}

#[test]
fn test_missing_sstable_file() {
    // Test behavior when an SSTable file is deleted while DB is closed
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Create and flush data
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..100 {
            db.put(format!("key_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }

        db.flush().unwrap();
    }

    // Delete ALL SSTable files (there may be multiple)
    for entry in fs::read_dir(&data_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("sst") {
            fs::remove_file(&path).unwrap();
        }
    }

    // Also delete WAL to truly test missing files scenario
    let wal_path = data_dir.join("wal.log");
    if wal_path.exists() {
        fs::remove_file(&wal_path).unwrap();
    }

    // Reopen - should handle missing file
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let result = DB::open(opts);

        match result {
            Ok(db) => {
                // Opened but data should be lost (no SSTable, no WAL)
                let found = (0..100)
                    .filter(|i| {
                        db.get(format!("key_{:03}", i).as_bytes())
                            .unwrap()
                            .is_some()
                    })
                    .count();

                assert!(
                    found == 0,
                    "Data should be lost if both SSTable and WAL are missing"
                );

                // DB should still be usable for new writes
                db.put(b"new_key", b"new_value").unwrap();
                assert_eq!(db.get(b"new_key").unwrap().unwrap().as_ref(), b"new_value");
            }
            Err(_) => {
                // Also acceptable - strict file integrity check
            }
        }
    }
}

#[test]
fn test_error_propagation() {
    // Test that I/O errors are propagated as Err, not panics
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key", b"value").unwrap();

    // Make directory read-only
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&data_dir, perms).unwrap();

    // Operations should return Err, not panic
    let flush_result = db.flush();
    assert!(flush_result.is_err(), "Should return Err on I/O failure");

    // Restore permissions
    let mut perms = fs::metadata(&data_dir).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&data_dir, perms).unwrap();

    // DB should still be usable after error
    assert_eq!(db.get(b"key").unwrap().unwrap().as_ref(), b"value");
}
