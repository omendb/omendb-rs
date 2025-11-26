// Stress tests for new APIs: snapshots, iter(), prefix()
// Tests stability and correctness under load

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn test_snapshot_isolation_under_concurrent_writes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Initial data: 100 keys with value "v1"
    for i in 0..100 {
        db.put(format!("key_{:04}", i).as_bytes(), b"v1").unwrap();
    }
    db.flush().unwrap();

    // Create snapshot BEFORE writes
    let snapshot = db.snapshot_consistent().unwrap();

    // Now write "v2" to all keys concurrently
    let barrier = Arc::new(Barrier::new(5));
    let mut handles = vec![];

    for thread_id in 0..4 {
        let db = db.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for i in 0..25 {
                let key = format!("key_{:04}", thread_id * 25 + i);
                db.put(key.as_bytes(), b"v2").unwrap();
            }
        }));
    }

    // Snapshot reader (in parallel with writes)
    let snap_clone = snapshot;
    let barrier_clone = barrier.clone();
    let reader = thread::spawn(move || {
        barrier_clone.wait();
        let mut v1_count = 0;
        for i in 0..100 {
            let key = format!("key_{:04}", i);
            if let Ok(Some(val)) = snap_clone.get(key.as_bytes()) {
                if val.as_ref() == b"v1" {
                    v1_count += 1;
                }
            }
        }
        v1_count
    });

    // Wait for all threads
    for h in handles {
        h.join().unwrap();
    }
    let v1_count = reader.join().unwrap();

    // Snapshot must see ALL original values (100 keys with v1)
    assert_eq!(v1_count, 100, "Snapshot must see point-in-time state");

    // Current DB should see updated values
    let mut v2_count = 0;
    for i in 0..100 {
        let key = format!("key_{:04}", i);
        if let Ok(Some(val)) = db.get(key.as_bytes()) {
            if val.as_ref() == b"v2" {
                v2_count += 1;
            }
        }
    }
    assert_eq!(v2_count, 100, "DB should see updated values");
}

#[test]
fn test_multiple_snapshots_under_load() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Create multiple snapshots at different points
    let mut snapshots = vec![];

    for version in 1..=10 {
        // Write version-specific data
        for i in 0..50 {
            let key = format!("key_{:04}", i);
            let val = format!("v{}", version);
            db.put(key.as_bytes(), val.as_bytes()).unwrap();
        }
        db.flush().unwrap();

        // Capture snapshot
        let snap = db.snapshot_consistent().unwrap();
        println!(
            "Version {}: snapshot seq_num = {}",
            version,
            snap.sequence_number()
        );
        snapshots.push((version, snap));
    }

    // Verify each snapshot sees its point-in-time state
    for (expected_version, snapshot) in snapshots {
        let expected_val = format!("v{}", expected_version);
        let key = b"key_0000";
        let val = snapshot.get(key).unwrap().unwrap();
        println!(
            "Snapshot {} (seq_num {}): key_0000 = {:?} (expected {:?})",
            expected_version,
            snapshot.sequence_number(),
            String::from_utf8_lossy(&val),
            expected_val
        );
        assert_eq!(
            val.as_ref(),
            expected_val.as_bytes(),
            "Snapshot {} should see v{}",
            expected_version,
            expected_version
        );
    }
}

#[test]
fn test_iter_large_dataset() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Insert 10K keys
    let num_keys = 10_000;
    for i in 0..num_keys {
        let key = format!("key_{:06}", i);
        let val = format!("value_{:06}", i);
        db.put(key.as_bytes(), val.as_bytes()).unwrap();
    }
    db.flush().unwrap();

    // Test iter() returns all keys in sorted order
    let start = Instant::now();
    let iter = db.iter().unwrap();
    let mut count = 0;
    let mut prev_key: Option<Vec<u8>> = None;

    for result in iter {
        let (key, _) = result.unwrap();
        count += 1;

        // Verify sorted order
        if let Some(prev) = &prev_key {
            assert!(key.as_ref() > prev.as_slice(), "Keys must be sorted");
        }
        prev_key = Some(key.to_vec());
    }

    let elapsed = start.elapsed();
    assert_eq!(count, num_keys, "Must iterate all keys");
    println!("iter() 10K keys in {:?}", elapsed);
}

#[test]
fn test_prefix_scan_correctness() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Create keys with different prefixes
    for i in 0..100 {
        db.put(format!("user:{:04}", i).as_bytes(), b"user_data")
            .unwrap();
        db.put(format!("product:{:04}", i).as_bytes(), b"product_data")
            .unwrap();
        db.put(format!("order:{:04}", i).as_bytes(), b"order_data")
            .unwrap();
    }
    db.flush().unwrap();

    // Test prefix scan for each type
    let user_count = db.prefix(b"user:").unwrap().count();
    let product_count = db.prefix(b"product:").unwrap().count();
    let order_count = db.prefix(b"order:").unwrap().count();

    assert_eq!(user_count, 100, "Should find all user keys");
    assert_eq!(product_count, 100, "Should find all product keys");
    assert_eq!(order_count, 100, "Should find all order keys");

    // Test specific prefix scan values
    for result in db.prefix(b"user:").unwrap() {
        let (k, v) = result.unwrap();
        assert!(k.as_ref().starts_with(b"user:"));
        assert_eq!(v.as_ref(), b"user_data");
    }
}

#[test]
fn test_prefix_edge_cases() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Test with 0xFF bytes (edge case for increment_bytes)
    db.put(&[0xFF, 0xFF, 0x01], b"val1").unwrap();
    db.put(&[0xFF, 0xFF, 0x02], b"val2").unwrap();
    db.put(&[0xFF, 0xFF, 0xFF], b"val3").unwrap();
    db.put(&[0xFF, 0xFF, 0xFF, 0x01], b"val4").unwrap();

    // Prefix scan with 0xFF,0xFF should find all 4 keys
    let count = db.prefix(&[0xFF, 0xFF]).unwrap().count();
    assert_eq!(count, 4, "Should find all keys with 0xFF,0xFF prefix");

    // Prefix scan with empty prefix should iterate all keys
    let total = db.prefix(b"").unwrap().count();
    assert_eq!(total, 4, "Empty prefix should return all keys");
}

#[test]
fn test_snapshot_range_under_load() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Insert ordered data
    for i in 0..1000 {
        let key = format!("key_{:06}", i);
        db.put(key.as_bytes(), b"original").unwrap();
    }
    db.flush().unwrap();

    // Create snapshot
    let snapshot = db.snapshot_consistent().unwrap();

    // Modify every other key after snapshot
    for i in (0..1000).step_by(2) {
        let key = format!("key_{:06}", i);
        db.put(key.as_bytes(), b"modified").unwrap();
    }

    // Range scan on snapshot should see all original values
    let range_count = snapshot
        .range(b"key_", Some(b"key_z"))
        .unwrap()
        .filter(|r| {
            if let Ok((_, val)) = r {
                val.as_ref() == b"original"
            } else {
                false
            }
        })
        .count();

    assert_eq!(range_count, 1000, "Snapshot range should see all originals");
}

#[test]
fn test_many_concurrent_snapshots() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Initial data
    for i in 0..100 {
        db.put(format!("key_{:04}", i).as_bytes(), b"initial")
            .unwrap();
    }
    db.flush().unwrap();

    let barrier = Arc::new(Barrier::new(10));
    let mut handles = vec![];

    // 10 threads creating snapshots concurrently
    for _ in 0..10 {
        let db = db.clone();
        let barrier = barrier.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            let mut snapshots = vec![];
            for _ in 0..10 {
                let snap = db.snapshot().unwrap();
                snapshots.push(snap);
                thread::sleep(Duration::from_millis(1));
            }
            // Verify all snapshots work
            for snap in &snapshots {
                let _ = snap.get(b"key_0001").unwrap();
            }
            snapshots.len()
        }));
    }

    let mut total_snapshots = 0;
    for h in handles {
        total_snapshots += h.join().unwrap();
    }

    assert_eq!(total_snapshots, 100, "All snapshots should be created");
}

#[test]
fn test_iter_concurrent_with_writes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Initial data
    for i in 0..500 {
        db.put(format!("key_{:06}", i).as_bytes(), b"value")
            .unwrap();
    }
    db.flush().unwrap();

    let barrier = Arc::new(Barrier::new(3));

    // Writer thread
    let db_write = db.clone();
    let barrier_write = barrier.clone();
    let writer = thread::spawn(move || {
        barrier_write.wait();
        for i in 500..1000 {
            db_write
                .put(format!("key_{:06}", i).as_bytes(), b"new")
                .unwrap();
            if i % 100 == 0 {
                thread::sleep(Duration::from_millis(1));
            }
        }
    });

    // Iterator thread (uses consistent snapshot internally)
    let db_iter = db.clone();
    let barrier_iter = barrier.clone();
    let iterator = thread::spawn(move || {
        barrier_iter.wait();
        thread::sleep(Duration::from_millis(5));
        let count = db_iter.iter().unwrap().count();
        count
    });

    // Deleter thread
    let db_del = db.clone();
    let barrier_del = barrier.clone();
    let deleter = thread::spawn(move || {
        barrier_del.wait();
        for i in 0..100 {
            db_del.delete(format!("key_{:06}", i).as_bytes()).unwrap();
        }
    });

    writer.join().unwrap();
    let iter_count = iterator.join().unwrap();
    deleter.join().unwrap();

    // iter() may see any consistent snapshot state
    assert!(
        iter_count >= 400,
        "Should see at least initial keys minus deletes"
    );
    assert!(iter_count <= 1000, "Should not exceed total keys");
}

#[test]
fn test_snapshot_memory_pressure() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024 * 1024, // Small memtable
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Create many snapshots (tests memory management)
    let mut snapshots = vec![];

    for batch in 0..10 {
        // Write data
        for i in 0..100 {
            let key = format!("batch{}:key{}", batch, i);
            db.put(key.as_bytes(), b"data").unwrap();
        }

        // Create consistent snapshot (flushes to ensure data is captured)
        // Using snapshot_consistent() because snapshot() only captures SSTable state
        let snap = db.snapshot_consistent().unwrap();
        snapshots.push(snap);
    }

    // Verify all snapshots are still accessible
    for (i, snap) in snapshots.iter().enumerate() {
        let key = format!("batch{}:key0", i);
        assert!(
            snap.get(key.as_bytes()).unwrap().is_some(),
            "Snapshot {} should still work",
            i
        );
    }

    // Drop snapshots and verify no memory leaks (implicit via test completion)
    drop(snapshots);

    // DB should still work
    assert!(db.get(b"batch0:key0").unwrap().is_some());
}
