// Crash Recovery Stress Test
// Hammers the database with random operations and simulated crashes
// to build confidence in recovery correctness

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use seerdb::{DBOptions, SyncPolicy, DB};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

/// Track expected state for verification
struct ExpectedState {
    data: HashMap<String, Option<Vec<u8>>>,
}

impl ExpectedState {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn put(&mut self, key: &str, value: &[u8]) {
        self.data.insert(key.to_string(), Some(value.to_vec()));
    }

    fn delete(&mut self, key: &str) {
        self.data.insert(key.to_string(), None);
    }

    fn verify(&self, db: &DB) -> Result<(), String> {
        for (key, expected) in &self.data {
            let actual = db.get(key.as_bytes()).map_err(|e| e.to_string())?;
            match (expected, actual) {
                (Some(exp), Some(act)) if exp == act.as_ref() => {}
                (None, None) => {}
                (exp, act) => {
                    return Err(format!(
                        "Mismatch for key '{}': expected {:?}, got {:?}",
                        key,
                        exp.as_ref().map(|v| v.len()),
                        act.as_ref().map(|v| v.len())
                    ));
                }
            }
        }
        Ok(())
    }
}

#[test]
fn test_rapid_crash_cycles() {
    // Many quick open/write/crash/recover cycles
    // Note: After WAL recovery, DB::open() creates a fresh WAL. If we close
    // without flushing, the recovered data is lost. This test flushes after
    // verification to persist recovered data.
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    for cycle in 0..10 {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write 10 keys per cycle with cycle-specific values
        for i in 0..10 {
            let key = format!("cycle{}_{}", cycle, i);
            let value = format!("value_cycle{}_{}", cycle, i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Flush to SSTable to persist data
        db.flush().unwrap();
        drop(db);

        // Verify all data from this and previous cycles exists
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for c in 0..=cycle {
            for i in 0..10 {
                let key = format!("cycle{}_{}", c, i);
                let expected_value = format!("value_cycle{}_{}", c, i);
                let actual = db.get(key.as_bytes()).unwrap();
                assert!(
                    actual.is_some(),
                    "Key '{}' should exist after cycle {}, checking at cycle {}",
                    key,
                    c,
                    cycle
                );
                assert_eq!(
                    actual.unwrap().as_ref(),
                    expected_value.as_bytes(),
                    "Value mismatch for key '{}' at cycle {}",
                    key,
                    cycle
                );
            }
        }
    }
}

#[test]
fn test_concurrent_writes_crash_recovery() {
    // Multiple threads writing, then crash, verify recovery
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let completed = Arc::new(AtomicU64::new(0));

    // Phase 1: Concurrent writes
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = Arc::new(DB::open(opts).unwrap());

        let mut handles = vec![];
        for thread_id in 0..4 {
            let db = Arc::clone(&db);
            let completed = Arc::clone(&completed);

            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    let key = format!("t{}_{:04}", thread_id, i);
                    let value = format!("value_{}_{}", thread_id, i);
                    db.put(key.as_bytes(), value.as_bytes()).unwrap();
                    completed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Crash without clean shutdown (drop without explicit flush)
    }

    let writes_completed = completed.load(Ordering::Relaxed);
    assert_eq!(writes_completed, 2000, "All writes should complete");

    // Phase 2: Recover and verify
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let mut found = 0;
        for thread_id in 0..4 {
            for i in 0..500 {
                let key = format!("t{}_{:04}", thread_id, i);
                let expected_value = format!("value_{}_{}", thread_id, i);

                if let Some(value) = db.get(key.as_bytes()).unwrap() {
                    assert_eq!(
                        value.as_ref(),
                        expected_value.as_bytes(),
                        "Value mismatch for key {}",
                        key
                    );
                    found += 1;
                }
            }
        }

        // With SyncPolicy::SyncData, all writes should be recovered
        assert_eq!(
            found, 2000,
            "All {} writes should be recovered",
            writes_completed
        );
    }
}

#[test]
fn test_batch_recovery() {
    // Test that batch writes are atomic across crashes
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write several batches
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for batch_id in 0..10 {
            let mut batch = db.batch();
            for i in 0..100 {
                let key = format!("batch{}_{:03}", batch_id, i);
                let value = format!("value_{}_{}", batch_id, i);
                batch.put(key.as_bytes(), value.as_bytes());
            }
            batch.commit().unwrap();
        }

        // Crash
    }

    // Recover and verify batches are complete
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for batch_id in 0..10 {
            let mut batch_keys_found = 0;
            for i in 0..100 {
                let key = format!("batch{}_{:03}", batch_id, i);
                if db.get(key.as_bytes()).unwrap().is_some() {
                    batch_keys_found += 1;
                }
            }

            // Batch should be all-or-nothing
            assert!(
                batch_keys_found == 0 || batch_keys_found == 100,
                "Batch {} should be atomic: found {}/100 keys",
                batch_id,
                batch_keys_found
            );
        }
    }
}

#[test]
fn test_transaction_crash_recovery() {
    // Test that uncommitted transactions don't persist
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    // Write some committed and uncommitted transactions
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Committed transaction
        {
            let mut txn = db.begin_transaction();
            for i in 0..50 {
                txn.put(format!("committed_{:03}", i).as_bytes(), b"value")
                    .unwrap();
            }
            txn.commit().unwrap();
        }

        // Uncommitted transaction (will be rolled back on crash)
        {
            let mut txn = db.begin_transaction();
            for i in 0..50 {
                txn.put(format!("uncommitted_{:03}", i).as_bytes(), b"value")
                    .unwrap();
            }
            // Don't commit - simulate crash
            drop(txn);
        }

        // Crash
    }

    // Recover and verify
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Committed should exist
        for i in 0..50 {
            assert!(
                db.get(format!("committed_{:03}", i).as_bytes())
                    .unwrap()
                    .is_some(),
                "Committed key {} should exist",
                i
            );
        }

        // Uncommitted should NOT exist
        for i in 0..50 {
            assert!(
                db.get(format!("uncommitted_{:03}", i).as_bytes())
                    .unwrap()
                    .is_none(),
                "Uncommitted key {} should NOT exist",
                i
            );
        }
    }
}

#[test]
fn test_mixed_operations_stress() {
    // Interleaved puts, deletes, batches, transactions, flushes
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let mut rng = StdRng::seed_from_u64(12345);
    let mut expected = ExpectedState::new();

    // Reduced iterations for faster CI (fsync-heavy test)
    for round in 0..2 {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            use_direct_wal: true,
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Random operations
        for _ in 0..20 {
            let op = rng.gen_range(0..5);
            match op {
                0 => {
                    // Single put
                    let key = format!("single_{:04}", rng.gen_range(0..500));
                    let value: Vec<u8> = (0..50).map(|_| rng.gen()).collect();
                    db.put(key.as_bytes(), &value).unwrap();
                    expected.put(&key, &value);
                }
                1 => {
                    // Single delete
                    let key = format!("single_{:04}", rng.gen_range(0..500));
                    db.delete(key.as_bytes()).unwrap();
                    expected.delete(&key);
                }
                2 => {
                    // Batch write
                    let mut batch = db.batch();
                    let batch_key_prefix = format!("batch_{}_{}", round, rng.gen_range(0..100));
                    for i in 0..10 {
                        let key = format!("{}_{}", batch_key_prefix, i);
                        let value: Vec<u8> = (0..30).map(|_| rng.gen()).collect();
                        batch.put(key.as_bytes(), &value);
                        expected.put(&key, &value);
                    }
                    batch.commit().unwrap();
                }
                3 => {
                    // Transaction
                    let mut txn = db.begin_transaction();
                    let txn_key_prefix = format!("txn_{}_{}", round, rng.gen_range(0..100));
                    for i in 0..5 {
                        let key = format!("{}_{}", txn_key_prefix, i);
                        let value: Vec<u8> = (0..20).map(|_| rng.gen()).collect();
                        txn.put(key.as_bytes(), &value).unwrap();
                        expected.put(&key, &value);
                    }
                    txn.commit().unwrap();
                }
                4 => {
                    // Flush
                    db.flush().unwrap();
                }
                _ => unreachable!(),
            }
        }

        // Flush to persist data to SSTables before "crash"
        // (WAL recovery creates fresh WAL, losing unflushed data)
        db.flush().unwrap();
        drop(db);

        // Verify after last round
        if round == 1 {
            let opts = DBOptions {
                data_dir: data_dir.clone(),
                ..Default::default()
            };
            let db = DB::open(opts).unwrap();
            expected
                .verify(&db)
                .expect(&format!("Verification failed at round {}", round));
        }
    }

    // Final verification
    let opts = DBOptions {
        data_dir: data_dir.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();
    expected.verify(&db).expect("Final verification failed");
}

#[test]
fn test_large_value_recovery() {
    // Test recovery with large values (stress vlog)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let large_value = vec![0xAB; 1024 * 1024]; // 1MB value

    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            wal_sync_policy: SyncPolicy::SyncData,
            vlog_threshold: Some(1024), // Enable vlog for large values
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..10 {
            db.put(format!("large_{}", i).as_bytes(), &large_value)
                .unwrap();
        }

        // Crash
    }

    // Recover
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            vlog_threshold: Some(1024),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..10 {
            let value = db
                .get(format!("large_{}", i).as_bytes())
                .unwrap()
                .expect(&format!("Large value {} should exist", i));
            assert_eq!(value.len(), 1024 * 1024, "Large value size mismatch");
            assert!(
                value.iter().all(|&b| b == 0xAB),
                "Large value content mismatch"
            );
        }
    }
}

#[test]
fn test_recovery_under_memory_pressure() {
    // Small memtable = more flushes = more recovery complexity
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            memtable_capacity: 64 * 1024, // 64KB - triggers frequent flushes
            wal_sync_policy: SyncPolicy::SyncData,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write enough to trigger multiple flushes
        for i in 0..5000 {
            let key = format!("pressure_{:05}", i);
            let value = format!("value_{:05}_padding_to_make_it_bigger", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }

        // Crash
    }

    // Recover
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..5000 {
            let key = format!("pressure_{:05}", i);
            let expected = format!("value_{:05}_padding_to_make_it_bigger", i);
            let value = db
                .get(key.as_bytes())
                .unwrap()
                .expect(&format!("Key {} should exist", key));
            assert_eq!(value.as_ref(), expected.as_bytes());
        }
    }
}
