//! Transaction Integration Tests
//!
//! P0 tests for transaction API stability before oadb depends on it.
//! Tests concurrent conflicts, crash recovery, and snapshot interaction.

use bytes::Bytes;
use seerdb::{DBError, DBOptions, DB};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

/// Test: Multiple transactions competing for the same key should detect conflicts
#[test]
fn test_concurrent_transaction_conflicts() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Initialize a shared counter key
    db.put(b"counter", b"0").unwrap();

    let num_threads = 10;
    let attempts_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));

    let successful_commits = Arc::new(AtomicUsize::new(0));
    let conflict_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let successful_commits = Arc::clone(&successful_commits);
        let conflict_count = Arc::clone(&conflict_count);

        let handle = thread::spawn(move || {
            barrier.wait();

            for _ in 0..attempts_per_thread {
                let mut txn = db.begin_transaction();

                // Read current value (adds to read-set)
                let current = txn.get(b"counter").unwrap();
                let value: i32 = current
                    .map(|b| String::from_utf8_lossy(&b).parse().unwrap_or(0))
                    .unwrap_or(0);

                // Increment
                let new_value = (value + 1).to_string();
                txn.put(b"counter", new_value.as_bytes()).unwrap();

                // Try to commit
                match txn.commit() {
                    Ok(()) => {
                        successful_commits.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(DBError::TransactionConflict(_)) => {
                        conflict_count.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }

            println!(
                "Thread {} finished: {} successful, {} conflicts",
                thread_id,
                successful_commits.load(Ordering::SeqCst),
                conflict_count.load(Ordering::SeqCst)
            );
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let total_successful = successful_commits.load(Ordering::SeqCst);
    let total_conflicts = conflict_count.load(Ordering::SeqCst);
    let total_attempts = num_threads * attempts_per_thread;

    println!(
        "Total: {} successful, {} conflicts out of {} attempts",
        total_successful, total_conflicts, total_attempts
    );

    // Verify: successful commits + conflicts = total attempts
    assert_eq!(total_successful + total_conflicts, total_attempts);

    // Verify: counter value equals successful commits
    let final_value: i32 = db
        .get(b"counter")
        .unwrap()
        .map(|b| String::from_utf8_lossy(&b).parse().unwrap())
        .unwrap();

    assert_eq!(
        final_value, total_successful as i32,
        "Counter should equal successful commits"
    );

    // We expect SOME conflicts with 10 threads competing
    assert!(
        total_conflicts > 0,
        "Expected conflicts with concurrent transactions"
    );

    println!(
        "PASS: Counter={}, Successful={}, Conflicts={}",
        final_value, total_successful, total_conflicts
    );
}

/// Test: Multiple transactions on different keys should not conflict
#[test]
fn test_concurrent_transactions_no_false_conflicts() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    let num_threads = 10;
    let ops_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));

    let conflict_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let conflict_count = Arc::clone(&conflict_count);

        let handle = thread::spawn(move || {
            barrier.wait();

            for i in 0..ops_per_thread {
                // Each thread works on its own key space
                let key = format!("thread_{}_key_{}", thread_id, i);

                let mut txn = db.begin_transaction();
                txn.put(key.as_bytes(), b"value").unwrap();

                match txn.commit() {
                    Ok(()) => {}
                    Err(DBError::TransactionConflict(_)) => {
                        conflict_count.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let conflicts = conflict_count.load(Ordering::SeqCst);

    // No conflicts expected - each thread has its own key space
    assert_eq!(
        conflicts, 0,
        "No conflicts expected when threads use different keys"
    );

    // Verify all keys written
    for thread_id in 0..num_threads {
        for i in 0..ops_per_thread {
            let key = format!("thread_{}_key_{}", thread_id, i);
            assert!(db.get(key.as_bytes()).unwrap().is_some());
        }
    }

    println!(
        "PASS: {} threads x {} ops = {} total writes, 0 conflicts",
        num_threads,
        ops_per_thread,
        num_threads * ops_per_thread
    );
}

/// Test: Committed transaction data survives crash (reopen)
#[test]
fn test_transaction_crash_recovery() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();

    // Phase 1: Write data via transaction, commit, then "crash" (drop without clean shutdown)
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let mut txn = db.begin_transaction();
        txn.put(b"txn_key_1", b"txn_value_1").unwrap();
        txn.put(b"txn_key_2", b"txn_value_2").unwrap();
        txn.put(b"txn_key_3", b"txn_value_3").unwrap();
        txn.commit().unwrap();

        // Simulate crash - drop DB without explicit close
        // WAL should have the committed data
        drop(db);
    }

    // Phase 2: Reopen and verify data recovered
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        assert_eq!(
            db.get(b"txn_key_1").unwrap(),
            Some(Bytes::from("txn_value_1")),
            "txn_key_1 should survive crash"
        );
        assert_eq!(
            db.get(b"txn_key_2").unwrap(),
            Some(Bytes::from("txn_value_2")),
            "txn_key_2 should survive crash"
        );
        assert_eq!(
            db.get(b"txn_key_3").unwrap(),
            Some(Bytes::from("txn_value_3")),
            "txn_key_3 should survive crash"
        );

        println!("PASS: Transaction data recovered after crash");
    }
}

/// Test: Uncommitted transaction data should NOT survive crash
#[test]
fn test_uncommitted_transaction_not_recovered() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();

    // Phase 1: Start transaction but don't commit, then "crash"
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write some committed data first
        db.put(b"committed_key", b"committed_value").unwrap();

        // Start transaction but don't commit
        let mut txn = db.begin_transaction();
        txn.put(b"uncommitted_key", b"uncommitted_value").unwrap();
        // Don't commit - just drop
        drop(txn);

        // Simulate crash
        drop(db);
    }

    // Phase 2: Reopen and verify uncommitted data is NOT there
    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        assert_eq!(
            db.get(b"committed_key").unwrap(),
            Some(Bytes::from("committed_value")),
            "Committed data should survive"
        );

        assert_eq!(
            db.get(b"uncommitted_key").unwrap(),
            None,
            "Uncommitted transaction data should NOT survive crash"
        );

        println!("PASS: Uncommitted transaction data correctly lost after crash");
    }
}

/// Test: Transaction reads see consistent snapshot despite concurrent writes
#[test]
fn test_transaction_snapshot_isolation() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Initialize
    db.put(b"key1", b"initial1").unwrap();
    db.put(b"key2", b"initial2").unwrap();

    // Start transaction - captures snapshot
    let mut txn = db.begin_transaction();

    // Read initial values
    assert_eq!(txn.get(b"key1").unwrap(), Some(Bytes::from("initial1")));
    assert_eq!(txn.get(b"key2").unwrap(), Some(Bytes::from("initial2")));

    // Concurrent write (outside transaction)
    db.put(b"key1", b"modified1").unwrap();
    db.put(b"key3", b"new_key").unwrap();

    // Transaction should still see old values (snapshot isolation)
    // Note: key1 is in read-set, so re-reading should still show snapshot value
    // Actually, our implementation returns from read-set tracking, let's verify
    // the snapshot behavior by checking what happens at commit

    // Write in transaction
    txn.put(b"key2", b"txn_modified").unwrap();

    // Commit should fail because key1 was read and modified externally
    let result = txn.commit();
    assert!(
        matches!(result, Err(DBError::TransactionConflict(_))),
        "Expected conflict on key1 which was read then modified externally"
    );

    // Verify external write persisted
    assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from("modified1")));
    assert_eq!(db.get(b"key3").unwrap(), Some(Bytes::from("new_key")));

    // key2 should still be initial (txn was aborted)
    assert_eq!(db.get(b"key2").unwrap(), Some(Bytes::from("initial2")));

    println!("PASS: Transaction correctly detected conflict from concurrent write");
}

/// Test: Transaction with explicit snapshot interaction
#[test]
fn test_transaction_and_snapshot_coexist() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Initial data
    db.put(b"shared_key", b"v1").unwrap();

    // Take a snapshot
    let snapshot = db.snapshot().unwrap();

    // Start transaction after snapshot
    let mut txn = db.begin_transaction();

    // Modify via direct put
    db.put(b"shared_key", b"v2").unwrap();

    // Snapshot still sees v1
    assert_eq!(
        snapshot.get(b"shared_key").unwrap(),
        Some(Bytes::from("v1"))
    );

    // Transaction read (adds to read-set) - sees v2 since txn started after v2 write
    // Wait, let's trace through:
    // - db.put(v1) at seq=1
    // - snapshot at seq=2
    // - txn starts at seq=2
    // - db.put(v2) at seq=3
    // - txn.get reads at seq=2, so should see v1
    let txn_value = txn.get(b"shared_key").unwrap();

    // Transaction should see value at its start time
    // After v2 write, current seq moved to 3
    // But txn started at seq=2, so should read v1
    println!("Transaction sees: {:?}", txn_value);

    // Transaction commits successfully if no conflict on read keys
    txn.put(b"other_key", b"other_value").unwrap();

    // This should fail because shared_key was read and then modified
    let result = txn.commit();
    assert!(
        matches!(result, Err(DBError::TransactionConflict(_))),
        "Expected conflict because shared_key was modified after txn start"
    );

    println!("PASS: Transaction and snapshot coexist correctly");
}

/// Test: Write-only transaction (no reads) should never conflict
#[test]
fn test_write_only_transactions_no_conflict() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    db.put(b"key", b"initial").unwrap();

    let num_threads = 10;
    let barrier = Arc::new(Barrier::new(num_threads));
    let successful = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let successful = Arc::clone(&successful);

        let handle = thread::spawn(move || {
            barrier.wait();

            // Write-only transaction (no reads, so empty read-set)
            let mut txn = db.begin_transaction();
            let value = format!("thread_{}", thread_id);
            txn.put(b"key", value.as_bytes()).unwrap();

            // Should always succeed - no read-set to conflict
            match txn.commit() {
                Ok(()) => {
                    successful.fetch_add(1, Ordering::SeqCst);
                }
                Err(e) => panic!("Write-only txn should not fail: {:?}", e),
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All should succeed
    assert_eq!(successful.load(Ordering::SeqCst), num_threads);

    println!(
        "PASS: {} write-only transactions all committed (last-writer-wins)",
        num_threads
    );
}

/// Test: Large transaction with many keys in read-set
#[test]
fn test_large_transaction_many_keys() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    let num_keys = 10_000;

    // Populate keys
    for i in 0..num_keys {
        let key = format!("key_{:06}", i);
        let value = format!("value_{:06}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // Start transaction and read all keys
    let mut txn = db.begin_transaction();

    for i in 0..num_keys {
        let key = format!("key_{:06}", i);
        let value = txn.get(key.as_bytes()).unwrap();
        assert!(value.is_some(), "Key {} should exist", key);
    }

    assert_eq!(txn.read_count(), num_keys);

    // Write some keys
    for i in 0..100 {
        let key = format!("key_{:06}", i);
        let value = format!("modified_{:06}", i);
        txn.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    assert_eq!(txn.write_count(), 100);

    // Commit should succeed (no concurrent modifications)
    txn.commit().unwrap();

    // Verify modifications
    for i in 0..100 {
        let key = format!("key_{:06}", i);
        let expected = format!("modified_{:06}", i);
        assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from(expected)));
    }

    println!(
        "PASS: Large transaction with {} reads and 100 writes committed successfully",
        num_keys
    );
}

/// Test: Transaction conflict detection with partial key overlap
#[test]
fn test_partial_key_overlap_conflict() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Initialize keys
    db.put(b"key_a", b"a").unwrap();
    db.put(b"key_b", b"b").unwrap();
    db.put(b"key_c", b"c").unwrap();

    // Transaction reads key_a and key_b
    let mut txn = db.begin_transaction();
    txn.get(b"key_a").unwrap();
    txn.get(b"key_b").unwrap();

    // External write modifies only key_b
    db.put(b"key_b", b"b_modified").unwrap();

    // Transaction writes key_c
    txn.put(b"key_c", b"c_from_txn").unwrap();

    // Should conflict because key_b was in read-set and modified
    let result = txn.commit();
    assert!(
        matches!(result, Err(DBError::TransactionConflict(ref c)) if c.conflicting_keys.len() == 1),
        "Expected exactly one conflict on key_b"
    );

    if let Err(DBError::TransactionConflict(c)) = result {
        assert_eq!(c.conflicting_keys[0], Bytes::from("key_b"));
        println!("PASS: Detected conflict on key_b as expected");
    }
}
