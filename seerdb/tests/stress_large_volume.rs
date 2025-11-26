// Large Volume Stress Tests
// Tests database stability with large operation counts
// Critical for production scale validation
// Added Nov 14, 2025 for production validation

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_500k_sequential_operations() {
    // Test 500K sequential write + read operations
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    println!("Writing 250K operations...");
    let write_count = 250_000;

    // Write phase
    for i in 0..write_count {
        let key = format!("large_volume_{:010}", i);
        let value = format!("value_{:010}", i);

        db.put(key.as_bytes(), value.as_bytes()).unwrap();

        if i % 50_000 == 0 && i > 0 {
            println!("  Wrote {} operations", i);
        }
    }

    println!("Reading 250K operations...");

    // Read phase
    let mut found = 0;
    let mut not_found = 0;

    for i in 0..write_count {
        let key = format!("large_volume_{:010}", i);

        match db.get(key.as_bytes()).unwrap() {
            Some(value) => {
                let expected = format!("value_{:010}", i);
                assert_eq!(value.as_ref(), expected.as_bytes());
                found += 1;
            }
            None => {
                not_found += 1;
            }
        }

        if i % 50_000 == 0 && i > 0 {
            println!("  Read {} operations ({} found)", i, found);
        }
    }

    assert_eq!(found, write_count, "All keys should be found");
    assert_eq!(not_found, 0, "No keys should be missing");

    println!("Large volume test completed: {} operations verified", found);
}

#[test]
fn test_many_batches() {
    // Test 10K batches of 50 operations each (500K total)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    println!("Writing 10K batches (50 ops each)...");

    for batch_id in 0..10_000 {
        let mut batch = db.batch();

        for i in 0..50 {
            let key = format!("batch_{:06}_item_{:03}", batch_id, i);
            let value = format!("value_{}", i);
            batch.put(key.as_bytes(), value.as_bytes());
        }

        batch.commit().unwrap();

        if batch_id % 2_000 == 0 && batch_id > 0 {
            println!("  Completed {} batches", batch_id);
        }
    }

    println!("Verifying batch atomicity...");

    // Verify all batches are complete (spot check)
    for batch_id in (0..10_000).step_by(100) {
        for i in 0..50 {
            let key = format!("batch_{:06}_item_{:03}", batch_id, i);
            assert!(
                db.get(key.as_bytes()).unwrap().is_some(),
                "Batch {} item {} should exist",
                batch_id,
                i
            );
        }
    }

    println!("All batches verified");
}

#[test]
fn test_many_flushes() {
    // Test multiple automatic flushes
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 5 * 1024 * 1024, // 5MB (triggers frequent flushes)
        background_flush: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    println!("Writing 200K operations to trigger multiple flushes...");

    let write_count = 200_000;

    for i in 0..write_count {
        let key = format!("flush_heavy_{:010}", i);
        let value = vec![b'x'; 500]; // 500 bytes

        db.put(key.as_bytes(), &value).unwrap();

        if i % 50_000 == 0 && i > 0 {
            println!("  Wrote {} operations", i);
            let mem = db.estimate_memory_usage();
            println!("  Current memory: {} MB", mem / 1024 / 1024);
        }
    }

    println!("Waiting for background flushes...");
    std::thread::sleep(std::time::Duration::from_secs(3));

    println!("Verifying data after flushes...");

    // Verify data survived flushes (spot check every 1000)
    for i in (0..write_count).step_by(1000) {
        let key = format!("flush_heavy_{:010}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_some(),
            "Key {} should exist after flush",
            key
        );
    }

    // Check stats
    let stats = db.stats();
    println!("Total SSTables: {}", stats.total_sstables);
    println!("Total flushes: {}", stats.total_flushes);

    assert!(
        stats.total_flushes > 0,
        "Should have triggered at least one flush"
    );

    println!("Flush heavy test completed successfully");
}

#[test]
fn test_large_keys_and_values() {
    // Test with larger keys and values
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        vlog_threshold: Some(4096), // Enable vlog for large values
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    println!("Writing 10K large operations...");

    let write_count = 10_000;

    for i in 0..write_count {
        // Large key (1KB)
        let key = format!("{:01000}", i); // Pad to 1000 chars

        // Large value (10KB)
        let value = vec![b'y'; 10_000];

        db.put(key.as_bytes(), &value).unwrap();

        if i % 2_000 == 0 && i > 0 {
            println!("  Wrote {} large operations", i);
        }
    }

    println!("Verifying large operations...");

    // Verify (spot check)
    for i in (0..write_count).step_by(100) {
        let key = format!("{:01000}", i);

        let value = db.get(key.as_bytes()).unwrap().unwrap();
        assert_eq!(value.len(), 10_000, "Large value should be 10KB");
    }

    println!("Large key/value test completed");
}

#[test]
fn test_mixed_operations_at_scale() {
    // Test mixed puts, gets, deletes at scale
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    println!("Running mixed operations at scale...");

    let total_ops = 300_000;
    let mut puts = 0;
    let mut gets = 0;
    let mut deletes = 0;

    for i in 0..total_ops {
        match i % 3 {
            0 => {
                // Put
                let key = format!("mixed_{:010}", i);
                let value = format!("value_{}", i);
                db.put(key.as_bytes(), value.as_bytes()).unwrap();
                puts += 1;
            }
            1 => {
                // Get
                let key = format!("mixed_{:010}", (i / 3) * 3); // Get previously written key
                let _ = db.get(key.as_bytes());
                gets += 1;
            }
            2 => {
                // Delete (delete half the keys)
                if i % 6 == 2 {
                    let key = format!("mixed_{:010}", (i / 6) * 6);
                    db.delete(key.as_bytes()).unwrap();
                    deletes += 1;
                }
            }
            _ => unreachable!(),
        }

        if i % 60_000 == 0 && i > 0 {
            println!("  Completed {} operations", i);
        }
    }

    println!(
        "Mixed operations completed: {} puts, {} gets, {} deletes",
        puts, gets, deletes
    );

    // Verify some keys exist and some are deleted
    assert!(db.get(b"mixed_0000000003").unwrap().is_some()); // Should exist (not deleted)
    assert!(db.get(b"mixed_0000000000").unwrap().is_none()); // Should be deleted

    db.flush().unwrap();
}

#[test]
fn test_reopens_at_scale() {
    // Test database reopens after large amounts of data
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    println!("Phase 1: Write 100K operations");

    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        for i in 0..100_000 {
            let key = format!("reopen_test_{:010}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        db.flush().unwrap();
        // Close
    }

    println!("Phase 2: Reopen and write 100K more");

    {
        let opts = DBOptions {
            data_dir: data_dir.clone(),
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        // Verify old data
        assert!(db.get(b"reopen_test_0000000000").unwrap().is_some());
        assert!(db.get(b"reopen_test_0000099999").unwrap().is_some());

        // Write more
        for i in 100_000..200_000 {
            let key = format!("reopen_test_{:010}", i);
            db.put(key.as_bytes(), b"value").unwrap();
        }

        db.flush().unwrap();
        // Close
    }

    println!("Phase 3: Final reopen and verify all 200K");

    {
        let opts = DBOptions {
            data_dir,
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        // Verify all data (spot check every 10000)
        for i in (0..200_000).step_by(10_000) {
            let key = format!("reopen_test_{:010}", i);
            assert!(
                db.get(key.as_bytes()).unwrap().is_some(),
                "Key {} should exist after reopens",
                key
            );
        }

        println!("All 200K operations verified after multiple reopens");
    }
}
