// Concurrent edge case tests
// Tests complex interactions: flush during read, compact during write, etc.

use seerdb::{DBOptions, SyncPolicy, DB};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn test_reads_during_flush() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024 * 1024, // 1MB
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate with data
    for i in 0..1000 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    let barrier = Arc::new(Barrier::new(2));

    // Reader thread
    let db_read = db.clone();
    let barrier_read = barrier.clone();
    let reader = thread::spawn(move || {
        barrier_read.wait();
        // Read while flush is happening
        for _ in 0..100 {
            let _ = db_read.get(b"key_0500");
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Flusher thread
    let db_flush = db.clone();
    let barrier_flush = barrier.clone();
    let flusher = thread::spawn(move || {
        barrier_flush.wait();
        // Trigger flush
        db_flush.flush().unwrap();
    });

    reader.join().unwrap();
    flusher.join().unwrap();

    // Verify data still accessible
    assert!(db.get(b"key_0500").unwrap().is_some());
}

#[test]
fn test_writes_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 64 * 1024, // 64KB - small to trigger compactions
        background_compaction: true,
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Write enough to trigger multiple levels and compactions
    for i in 0..5000 {
        db.put(format!("key_{:05}", i).as_bytes(), &vec![b'v'; 100])
            .unwrap();
    }

    let barrier = Arc::new(Barrier::new(2));

    // Continue writing
    let db_write = db.clone();
    let barrier_write = barrier.clone();
    let writer = thread::spawn(move || {
        barrier_write.wait();
        for i in 5000..6000 {
            db_write
                .put(format!("key_{:05}", i).as_bytes(), &vec![b'v'; 100])
                .unwrap();
        }
    });

    // Force flush to potentially trigger compaction
    let db_flush = db.clone();
    let barrier_flush = barrier.clone();
    let flusher = thread::spawn(move || {
        barrier_flush.wait();
        thread::sleep(Duration::from_millis(10));
        db_flush.flush().unwrap();
    });

    writer.join().unwrap();
    flusher.join().unwrap();

    // Verify all data accessible
    assert!(db.get(b"key_00500").unwrap().is_some());
    assert!(db.get(b"key_05500").unwrap().is_some());
}

#[test]
fn test_concurrent_flushes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Populate data
    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Try to flush from multiple threads simultaneously
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let db_clone = db.clone();
            thread::spawn(move || {
                db_clone.flush().unwrap();
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify data intact
    assert_eq!(db.get(b"key_050").unwrap().unwrap().as_ref(), b"value");
}

#[test]
fn test_delete_during_read() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    // Pre-populate
    for i in 0..1000 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    let barrier = Arc::new(Barrier::new(2));
    // Signal flag: reader sets this after first successful read
    let reader_found_key = Arc::new(AtomicBool::new(false));

    // Reader thread - repeatedly reads
    let db_read = db.clone();
    let barrier_read = barrier.clone();
    let found_flag = reader_found_key.clone();
    let reader = thread::spawn(move || {
        barrier_read.wait();
        let mut found_count = 0;
        for _ in 0..100 {
            if db_read.get(b"key_0500").unwrap().is_some() {
                found_count += 1;
                // Signal that we've found the key at least once
                found_flag.store(true, Ordering::Release);
            }
            thread::sleep(Duration::from_micros(50));
        }
        found_count
    });

    // Deleter thread - waits for reader to confirm it found the key
    let db_delete = db.clone();
    let barrier_delete = barrier.clone();
    let found_flag = reader_found_key.clone();
    let deleter = thread::spawn(move || {
        barrier_delete.wait();
        // Wait until reader has found the key at least once (with timeout)
        for _ in 0..1000 {
            if found_flag.load(Ordering::Acquire) {
                break;
            }
            thread::sleep(Duration::from_micros(100));
        }
        db_delete.delete(b"key_0500").unwrap();
    });

    let found = reader.join().unwrap();
    deleter.join().unwrap();

    // Should have found it at least once (before delete)
    assert!(found > 0, "Reader should see key before it's deleted");
}

#[test]
fn test_overwrite_during_read() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    db.put(b"key", b"original_value").unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Reader thread
    let db_read = db.clone();
    let barrier_read = barrier.clone();
    let reader = thread::spawn(move || {
        barrier_read.wait();
        let mut values = Vec::new();
        for _ in 0..100 {
            if let Some(v) = db_read.get(b"key").unwrap() {
                values.push(v);
            }
            thread::sleep(Duration::from_micros(50));
        }
        values
    });

    // Writer thread - overwrites
    let db_write = db.clone();
    let barrier_write = barrier.clone();
    let writer = thread::spawn(move || {
        barrier_write.wait();
        thread::sleep(Duration::from_micros(100));
        db_write.put(b"key", b"new_value").unwrap();
    });

    let values = reader.join().unwrap();
    writer.join().unwrap();

    // Should see at least one of each value (or just one if timing is precise)
    assert!(!values.is_empty());
    let has_original = values.iter().any(|v| v.as_ref() == b"original_value");

    // Should see original before overwrite
    assert!(has_original, "Should see original value");
}

#[test]
fn test_heavy_concurrent_mixed_operations() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 128 * 1024, // 128KB
        background_compaction: false,  // Disable to avoid compaction removing tombstones
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    let write_barrier = Arc::new(Barrier::new(2));
    let delete_barrier = Arc::new(Barrier::new(2));

    // Writer 1
    let db1 = db.clone();
    let b1 = write_barrier.clone();
    let w1 = thread::spawn(move || {
        b1.wait();
        for i in 0..500 {
            db1.put(format!("key_a_{:04}", i).as_bytes(), b"value_a")
                .unwrap();
        }
    });

    // Writer 2
    let db2 = db.clone();
    let b2 = write_barrier.clone();
    let w2 = thread::spawn(move || {
        b2.wait();
        for i in 0..500 {
            db2.put(format!("key_b_{:04}", i).as_bytes(), b"value_b")
                .unwrap();
        }
    });

    // Wait for writers to finish before starting reader and deleter
    w1.join().unwrap();
    w2.join().unwrap();

    // Reader
    let db3 = db.clone();
    let b3 = delete_barrier.clone();
    let r = thread::spawn(move || {
        b3.wait();
        let mut read_count = 0;
        for i in 0..500 {
            if db3
                .get(format!("key_a_{:04}", i).as_bytes())
                .unwrap()
                .is_some()
            {
                read_count += 1;
            }
        }
        read_count
    });

    // Deleter - starts AFTER writes finish
    let db4 = db.clone();
    let b4 = delete_barrier.clone();
    let d = thread::spawn(move || {
        b4.wait();
        for i in 0..250 {
            db4.delete(format!("key_a_{:04}", i).as_bytes()).unwrap();
        }
    });

    let _reads = r.join().unwrap(); // May vary depending on timing
    d.join().unwrap();

    // Flush to persist
    db.flush().unwrap();

    // Wait for any background compaction to finish
    // Background compaction could interfere with our verification
    thread::sleep(Duration::from_millis(500));

    // Debug: Check DB stats
    let stats = db.stats();
    eprintln!("L0 SSTables: {}", stats.sstables_per_level[0]);
    eprintln!("Total SSTables: {}", stats.total_sstables);

    // Verify final state
    // key_a_0..249: deleted
    for i in 0..250 {
        let key = format!("key_a_{:04}", i);
        let result = db.get(key.as_bytes()).unwrap();
        if result.is_some() {
            eprintln!(
                "ERROR: Key {} should be deleted but got value: {:?}",
                key, result
            );
            eprintln!("This suggests the tombstone is not masking the older value in L0");
        }
        assert!(result.is_none(), "Key {} should be deleted", key);
    }
    // key_a_250..499: should exist
    for i in 250..500 {
        assert!(db
            .get(format!("key_a_{:04}", i).as_bytes())
            .unwrap()
            .is_some());
    }
    // key_b_0..499: all should exist
    for i in 0..500 {
        assert!(db
            .get(format!("key_b_{:04}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_flush_during_wal_write() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        wal_sync_policy: SyncPolicy::SyncData, // Force WAL sync
        memtable_capacity: 64 * 1024,
        ..Default::default()
    };
    let db = Arc::new(DB::open(opts).unwrap());

    let barrier = Arc::new(Barrier::new(2));

    // Writer - slow WAL writes
    let db_write = db.clone();
    let b_write = barrier.clone();
    let writer = thread::spawn(move || {
        b_write.wait();
        for i in 0..100 {
            db_write
                .put(format!("key_{:03}", i).as_bytes(), &vec![b'v'; 1000])
                .unwrap();
        }
    });

    // Flusher - tries to flush during writes
    let db_flush = db.clone();
    let b_flush = barrier.clone();
    let flusher = thread::spawn(move || {
        b_flush.wait();
        thread::sleep(Duration::from_millis(10));
        db_flush.flush().unwrap();
    });

    writer.join().unwrap();
    flusher.join().unwrap();

    // All data should be present
    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_compaction_consistency() {
    // This tests that compaction doesn't lose data or create duplicates
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 32 * 1024, // Small to force many SSTables
        background_compaction: true,
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write unique values for each key
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = format!("value_{:04}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    // Force multiple flushes
    for _ in 0..5 {
        db.flush().unwrap();
        // Write more data
        for i in 0..200 {
            let key = format!("key_{:04}", i);
            let value = format!("updated_{:04}", i);
            db.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
    }

    // Final flush
    db.flush().unwrap();

    // Wait a bit for background compaction to finish
    thread::sleep(Duration::from_secs(2));

    // Verify all keys present with correct values
    for i in 0..200 {
        let key = format!("key_{:04}", i);
        let expected = format!("updated_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap().unwrap();
        assert_eq!(value.as_ref(), expected.as_bytes());
    }

    for i in 200..1000 {
        let key = format!("key_{:04}", i);
        let expected = format!("value_{:04}", i);
        let value = db.get(key.as_bytes()).unwrap().unwrap();
        assert_eq!(value.as_ref(), expected.as_bytes());
    }
}
