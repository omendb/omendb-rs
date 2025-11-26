// End-to-end integration tests for the complete DB interface
// Tests full lifecycle: open, write, read, flush, compact, close, recover

use bytes::Bytes;
use seerdb::wal::SyncPolicy;
use seerdb::{DBOptions, DB};
use tempfile::tempdir;

#[test]
fn test_db_full_lifecycle() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    // Phase 1: Create DB and write data
    {
        let db = DB::open(options.clone()).unwrap();

        // Write 1000 entries
        for i in 0..1000 {
            let key = format!("key_{:04}", i);
            let value = format!("value_{:04}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Verify all data is readable
        for i in 0..1000 {
            let key = format!("key_{:04}", i);
            let value = format!("value_{:04}", i);
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }

    // Phase 2: Reopen and verify data persisted
    {
        let db = DB::open(options.clone()).unwrap();

        // All data should be recovered
        for i in 0..1000 {
            let key = format!("key_{:04}", i);
            let value = format!("value_{:04}", i);
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }
}

#[test]
fn test_db_with_deletes() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    {
        let db = DB::open(options.clone()).unwrap();

        // Write 100 entries
        for i in 0..100 {
            let key = format!("key_{}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        // Delete every other key
        for i in (0..100).step_by(2) {
            let key = format!("key_{}", i);
            db.delete(key.as_bytes()).unwrap();
        }

        // Verify deletions
        for i in 0..100 {
            let key = format!("key_{}", i);
            if i % 2 == 0 {
                assert_eq!(db.get(key.as_bytes()).unwrap(), None);
            } else {
                assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value")));
            }
        }
    }

    // Reopen and verify deletions persisted
    {
        let db = DB::open(options).unwrap();

        for i in 0..100 {
            let key = format!("key_{}", i);
            if i % 2 == 0 {
                assert_eq!(db.get(key.as_bytes()).unwrap(), None);
            } else {
                assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value")));
            }
        }
    }
}

#[test]
fn test_db_overwrites() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    {
        let db = DB::open(options.clone()).unwrap();

        // Write initial values
        for i in 0..50 {
            let key = format!("key_{}", i);
            db.put(key.as_bytes(), b"old_value").unwrap();
        }

        // Overwrite with new values
        for i in 0..50 {
            let key = format!("key_{}", i);
            db.put(key.as_bytes(), b"new_value").unwrap();
        }

        // Verify new values
        for i in 0..50 {
            let key = format!("key_{}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap(),
                Some(Bytes::from("new_value"))
            );
        }
    }

    // Reopen and verify overwrites persisted
    {
        let db = DB::open(options).unwrap();

        for i in 0..50 {
            let key = format!("key_{}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap(),
                Some(Bytes::from("new_value"))
            );
        }
    }
}

#[test]
fn test_db_multiple_flushes() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        memtable_capacity: 10 * 1024, // 10KB - small enough to trigger flushes
        ..Default::default()
    };

    {
        let db = DB::open(options.clone()).unwrap();

        // Write enough data to trigger multiple flushes
        for i in 0..200 {
            let key = format!("key_{:04}", i);
            let value = vec![b'x'; 100]; // 100 bytes
            db.put(key.as_bytes(), &value).unwrap();
        }

        // Verify all data is accessible
        for i in 0..200 {
            let key = format!("key_{:04}", i);
            let value = vec![b'x'; 100];
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }

    // Reopen and verify data
    {
        let db = DB::open(options).unwrap();

        for i in 0..200 {
            let key = format!("key_{:04}", i);
            let value = vec![b'x'; 100];
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }
}

#[test]
fn test_db_large_values() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    {
        let db = DB::open(options.clone()).unwrap();

        // Write large values (simulating vector embeddings)
        for i in 0..100 {
            let key = format!("doc_{}", i);
            let value = vec![i as u8; 4096]; // 4KB embedding
            db.put(key.as_bytes(), &value).unwrap();
        }

        // Verify large values
        for i in 0..100 {
            let key = format!("doc_{}", i);
            let value = vec![i as u8; 4096];
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }
}

#[test]
fn test_db_crash_recovery_with_uncommitted_data() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        wal_sync_policy: SyncPolicy::SyncData,
        ..Default::default()
    };

    // Simulate crash: write data but don't close cleanly
    {
        let db = DB::open(options.clone()).unwrap();

        for i in 0..200 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Don't call any close/flush - simulate crash
        std::mem::drop(db);
    }

    // Recover from crash
    {
        let db = DB::open(options).unwrap();

        // All data should be recovered from WAL
        for i in 0..200 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            assert_eq!(
                db.get(key.as_bytes()).unwrap(),
                Some(Bytes::from(value)),
                "Failed to recover key_{} after crash",
                i
            );
        }
    }
}

#[test]
fn test_db_mixed_operations() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    {
        let db = DB::open(options.clone()).unwrap();

        // Mixed workload: puts, gets, deletes, overwrites
        for i in 0..100 {
            let key = format!("key_{}", i);

            // Write
            db.put(key.as_bytes(), b"value1").unwrap();

            // Read
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value1")));

            // Overwrite
            db.put(key.as_bytes(), b"value2").unwrap();
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value2")));

            // Delete odd keys
            if i % 2 == 1 {
                db.delete(key.as_bytes()).unwrap();
                assert_eq!(db.get(key.as_bytes()).unwrap(), None);
            }
        }

        // Verify final state
        for i in 0..100 {
            let key = format!("key_{}", i);
            if i % 2 == 1 {
                assert_eq!(db.get(key.as_bytes()).unwrap(), None);
            } else {
                assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value2")));
            }
        }
    }
}

#[test]
fn test_db_empty_database() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    let db = DB::open(options).unwrap();

    // Empty database
    assert_eq!(db.get(b"nonexistent").unwrap(), None);
    assert_eq!(db.memtable_len(), 0);
    assert_eq!(db.memtable_size(), 0);
}

#[test]
#[ignore] // TODO: Background compaction + reopen - investigate snapshot isolation
fn test_db_reopen_multiple_times() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    // Open and close 5 times, accumulating data
    for round in 0..5 {
        let db = DB::open(options.clone()).unwrap();

        // Write 20 keys per round
        for i in 0..20 {
            let key = format!("round_{}_{}", round, i);
            let value = format!("value_{}_{}", round, i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Verify all previous rounds' data still exists
        for prev_round in 0..=round {
            for i in 0..20 {
                let key = format!("round_{}_{}", prev_round, i);
                let value = format!("value_{}_{}", prev_round, i);
                assert_eq!(
                    db.get(key.as_bytes()).unwrap(),
                    Some(Bytes::from(value)),
                    "Failed at round {} checking data from round {}",
                    round,
                    prev_round
                );
            }
        }
    }

    // Final verification: all 100 keys should exist
    let db = DB::open(options).unwrap();
    for round in 0..5 {
        for i in 0..20 {
            let key = format!("round_{}_{}", round, i);
            let value = format!("value_{}_{}", round, i);
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(value)));
        }
    }
}

#[test]
fn test_db_sequential_vs_random_keys() {
    let dir = tempdir().unwrap();
    let options = DBOptions {
        data_dir: dir.path().to_path_buf(),
        ..Default::default()
    };

    let db = DB::open(options).unwrap();

    // Sequential keys
    for i in 0..100 {
        let key = format!("seq_{:04}", i);
        db.put(key.as_bytes(), b"sequential").unwrap();
    }

    // Random-ish keys
    for i in 0..100 {
        let key = format!("rand_{:04}", (i * 97) % 1000); // Pseudo-random
        db.put(key.as_bytes(), b"random").unwrap();
    }

    // Verify sequential
    for i in 0..100 {
        let key = format!("seq_{:04}", i);
        assert_eq!(
            db.get(key.as_bytes()).unwrap(),
            Some(Bytes::from("sequential"))
        );
    }

    // Verify random
    for i in 0..100 {
        let key = format!("rand_{:04}", (i * 97) % 1000);
        assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("random")));
    }
}
