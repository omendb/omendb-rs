// Concurrent Stress Tests
// Tests thread safety under heavy concurrent load
// Critical for production multi-threaded workloads
// Added Nov 14, 2025 for production validation

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

#[test]
fn test_concurrent_20_writers() {
    // Test 20 concurrent writer threads
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());
    let barrier = Arc::new(Barrier::new(20));

    let mut handles = vec![];

    println!("Spawning 20 concurrent writer threads...");

    for thread_id in 0..20 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            // Wait for all threads to be ready
            barrier_clone.wait();

            // Each thread writes 5K operations
            for i in 0..5000 {
                let key = format!("thread_{:02}_key_{:06}", thread_id, i);
                let value = format!("value_{:06}", i);

                db_clone.put(key.as_bytes(), value.as_bytes()).unwrap();
            }

            println!("Thread {} completed 5K writes", thread_id);
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    println!("All 20 threads completed (100K total operations)");

    // Verify all data is present and correct
    for thread_id in 0..20 {
        for i in (0..5000).step_by(500) {
            let key = format!("thread_{:02}_key_{:06}", thread_id, i);
            let expected_value = format!("value_{:06}", i);

            let value = db
                .get(key.as_bytes())
                .unwrap()
                .expect(&format!("Key {} should exist", key));

            assert_eq!(
                value.as_ref(),
                expected_value.as_bytes(),
                "Value mismatch for key {}",
                key
            );
        }
    }

    println!("All data verified correctly!");
}

#[test]
fn test_concurrent_mixed_workload() {
    // Test mixed readers and writers
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate some data
    println!("Pre-populating database...");
    for i in 0..10000 {
        let key = format!("initial_{:06}", i);
        db.put(key.as_bytes(), b"initial_value").unwrap();
    }

    let barrier = Arc::new(Barrier::new(20));
    let mut handles = vec![];

    println!("Spawning mixed workload: 10 writers + 10 readers");

    // 10 writer threads
    for thread_id in 0..10 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            for i in 0..3000 {
                let key = format!("writer_{:02}_key_{:06}", thread_id, i);
                let value = format!("value_{}", i);
                db_clone.put(key.as_bytes(), value.as_bytes()).unwrap();
            }
        });

        handles.push(handle);
    }

    // 10 reader threads
    for thread_id in 0..10 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let mut found = 0;
            let mut not_found = 0;

            for i in 0..5000 {
                let key = format!("initial_{:06}", i % 10000);

                match db_clone.get(key.as_bytes()).unwrap() {
                    Some(_) => found += 1,
                    None => not_found += 1,
                }
            }

            println!(
                "Reader {} completed: {} found, {} not found",
                thread_id, found, not_found
            );
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    println!("Mixed workload completed successfully");

    // Verify database is still consistent
    db.flush().unwrap();
    assert!(db.get(b"initial_000000").unwrap().is_some());
}

#[test]
fn test_concurrent_hot_keys() {
    // Test many threads updating the same keys (contention test)
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());
    let barrier = Arc::new(Barrier::new(20));

    println!("Testing hot key contention: 20 threads updating 100 keys");

    let mut handles = vec![];

    for thread_id in 0..20 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // All threads update the same 100 keys
            for i in 0..1000 {
                let key = format!("hot_key_{:03}", i % 100);
                let value = format!("thread_{}_value_{}", thread_id, i);

                db_clone.put(key.as_bytes(), value.as_bytes()).unwrap();
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Hot key test completed without deadlock");

    // Verify all hot keys exist (last write wins)
    for i in 0..100 {
        let key = format!("hot_key_{:03}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_some(),
            "Hot key {} should exist",
            key
        );
    }
}

#[test]
fn test_concurrent_deletes() {
    // Test concurrent writes and deletes
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate
    for i in 0..5000 {
        let key = format!("deletable_{:06}", i);
        db.put(key.as_bytes(), b"will_be_deleted").unwrap();
    }

    let barrier = Arc::new(Barrier::new(10));
    let mut handles = vec![];

    println!("Testing concurrent deletes: 5 writers + 5 deleters");

    // 5 writer threads
    for thread_id in 0..5 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            for i in 0..2000 {
                let key = format!("new_{:02}_{:06}", thread_id, i);
                db_clone.put(key.as_bytes(), b"new_value").unwrap();
            }
        });

        handles.push(handle);
    }

    // 5 deleter threads
    for thread_id in 0..5 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            for i in 0..1000 {
                let key = format!("deletable_{:06}", thread_id * 1000 + i);
                db_clone.delete(key.as_bytes()).unwrap();
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Concurrent delete test completed");

    // Verify deletions worked
    for i in 0..1000 {
        let key = format!("deletable_{:06}", i);
        assert!(
            db.get(key.as_bytes()).unwrap().is_none(),
            "Key {} should be deleted",
            key
        );
    }

    // Verify new writes worked
    assert!(db.get(b"new_00_000000").unwrap().is_some());
}

#[test]
fn test_concurrent_flushes() {
    // Test concurrent operations during flush
    let temp_dir = TempDir::new().unwrap();
    let data_dir = PathBuf::from(temp_dir.path());

    let opts = DBOptions {
        data_dir,
        memtable_capacity: 5 * 1024 * 1024, // 5MB (small to trigger flushes)
        background_flush: true,
        ..Default::default()
    };

    let db = Arc::new(DB::open(opts).unwrap());
    let barrier = Arc::new(Barrier::new(15));
    let mut handles = vec![];

    println!("Testing concurrent operations during flushes");

    // 10 writer threads (will trigger multiple flushes)
    for thread_id in 0..10 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            for i in 0..3000 {
                let key = format!("flush_test_{:02}_{:06}", thread_id, i);
                let value = vec![b'x'; 1000]; // 1KB values
                db_clone.put(key.as_bytes(), &value).unwrap();
            }
        });

        handles.push(handle);
    }

    // 5 reader threads (reading during flushes)
    for _thread_id in 0..5 {
        let db_clone = Arc::clone(&db);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            for i in 0..2000 {
                let key = format!("flush_test_{:02}_{:06}", i % 10, i);
                // May or may not find the key (race with writers)
                let _ = db_clone.get(key.as_bytes());
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Concurrent flush test completed");

    // Final flush and verification
    db.flush().unwrap();

    // Spot check
    assert!(db.get(b"flush_test_00_000000").unwrap().is_some());
    assert!(db.get(b"flush_test_09_002999").unwrap().is_some());
}
