// Crash recovery tests
// Tests WAL replay after simulated crashes
// Critical for durability: data must survive crashes

use seerdb::{DBOptions, SyncPolicy, DB};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_recovery_after_clean_shutdown() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write data
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

        // Clean shutdown (drop DB)
    }

    // Reopen and verify data
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..100 {
            assert!(
                db.get(format!("key_{:03}", i).as_bytes())
                    .unwrap()
                    .is_some(),
                "Key {} should exist after recovery",
                i
            );
        }
    }
}

#[test]
fn test_wal_replay_unflushed_writes() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write data WITHOUT flushing
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            db.put(
                format!("key_{:03}", i).as_bytes(),
                format!("value_{:03}", i).as_bytes(),
            )
            .unwrap();
        }

        // Simulate crash (drop without flush)
    }

    // Reopen and verify WAL replay recovered data
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            let value = db
                .get(format!("key_{:03}", i).as_bytes())
                .unwrap()
                .expect(&format!("Key {} should exist after WAL replay", i));
            assert_eq!(
                value.as_ref(),
                format!("value_{:03}", i).as_bytes(),
                "Value for key {} incorrect after recovery",
                i
            );
        }
    }
}

#[test]
fn test_wal_replay_with_deletes() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write then delete some keys
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write 100 keys
        for i in 0..100 {
            db.put(format!("key_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }

        // Delete first 50
        for i in 0..50 {
            db.delete(format!("key_{:03}", i).as_bytes()).unwrap();
        }

        // Crash without flush
    }

    // Reopen and verify deletes were replayed
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // First 50 should be deleted
        for i in 0..50 {
            assert!(
                db.get(format!("key_{:03}", i).as_bytes())
                    .unwrap()
                    .is_none(),
                "Key {} should be deleted after recovery",
                i
            );
        }

        // Last 50 should exist
        for i in 50..100 {
            assert!(
                db.get(format!("key_{:03}", i).as_bytes())
                    .unwrap()
                    .is_some(),
                "Key {} should exist after recovery",
                i
            );
        }
    }
}

#[test]
fn test_wal_replay_with_overwrites() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write, overwrite, then crash
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Initial write
        for i in 0..50 {
            db.put(format!("key_{:03}", i).as_bytes(), b"v1").unwrap();
        }

        // Overwrite with new value
        for i in 0..50 {
            db.put(format!("key_{:03}", i).as_bytes(), b"v2").unwrap();
        }

        // Crash without flush
    }

    // Reopen and verify latest value is preserved
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            let value = db.get(format!("key_{:03}", i).as_bytes()).unwrap().unwrap();
            assert_eq!(
                value.as_ref(),
                b"v2",
                "Should see latest value after recovery"
            );
        }
    }
}

#[test]
fn test_recovery_after_flush() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write, flush, write more, crash
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write and flush
        for i in 0..50 {
            db.put(format!("flushed_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }
        db.flush().unwrap();

        // Write more without flushing
        for i in 0..50 {
            db.put(format!("unflushed_{:03}", i).as_bytes(), b"value")
                .unwrap();
        }

        // Crash
    }

    // Reopen and verify both sets
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Flushed data should be in SSTables
        for i in 0..50 {
            assert!(db
                .get(format!("flushed_{:03}", i).as_bytes())
                .unwrap()
                .is_some());
        }

        // Unflushed data should be recovered from WAL
        for i in 0..50 {
            assert!(db
                .get(format!("unflushed_{:03}", i).as_bytes())
                .unwrap()
                .is_some());
        }
    }
}

#[test]
fn test_sync_policy_none_may_lose_data() {
    // NOTE: This test documents expected behavior with SyncPolicy::None
    // Data may be lost on crash if not fsync'd

    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write with no fsync
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::None,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key", b"value").unwrap();

        // Simulated crash - data may not be on disk yet
    }

    // Reopen - data may or may not be present
    // This is acceptable behavior for SyncPolicy::None (performance over durability)
    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let _db = DB::open(opts).unwrap();

    // We don't assert data exists because SyncPolicy::None makes no guarantees
    // This test just documents the behavior
}

#[test]
fn test_sync_policy_sync_data_guarantees() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write with SyncData (fsync)
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key", b"value").unwrap();

        // Simulated crash - data IS on disk (fsync guarantees)
    }

    // Reopen - data MUST be present
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        assert!(
            db.get(b"key").unwrap().is_some(),
            "SyncData must guarantee durability"
        );
    }
}

#[test]
fn test_multiple_open_close_cycles() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Multiple open/close cycles accumulate data
    for cycle in 0..10 {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write data specific to this cycle
        for i in 0..10 {
            db.put(format!("cycle_{}_{:02}", cycle, i).as_bytes(), b"value")
                .unwrap();
        }

        // Sometimes flush, sometimes don't
        if cycle % 2 == 0 {
            db.flush().unwrap();
        }

        // Close DB
    }

    // Final reopen - all data should be present
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for cycle in 0..10 {
            for i in 0..10 {
                assert!(
                    db.get(format!("cycle_{}_{:02}", cycle, i).as_bytes())
                        .unwrap()
                        .is_some(),
                    "Data from cycle {} should persist",
                    cycle
                );
            }
        }
    }
}

#[test]
fn test_recovery_preserves_ordering() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write operations in specific order
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key", b"v1").unwrap();
        db.put(b"key", b"v2").unwrap();
        db.delete(b"key").unwrap();
        db.put(b"key", b"v3").unwrap();

        // Crash
    }

    // Reopen and verify final state
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let value = db.get(b"key").unwrap().unwrap();
        assert_eq!(
            value.as_ref(),
            b"v3",
            "Should see final value after WAL replay"
        );
    }
}

#[test]
fn test_large_wal_replay() {
    // Test that large WAL files are replayed correctly
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write many operations without flushing
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            memtable_capacity: 10 * 1024 * 1024, // Large to prevent auto-flush
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..10000 {
            db.put(format!("key_{:05}", i).as_bytes(), &vec![b'v'; 100])
                .unwrap();
        }

        // Crash without flush (large WAL)
    }

    // Reopen and verify all data recovered
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..10000 {
            assert!(
                db.get(format!("key_{:05}", i).as_bytes())
                    .unwrap()
                    .is_some(),
                "Key {} should be recovered from large WAL",
                i
            );
        }
    }
}
