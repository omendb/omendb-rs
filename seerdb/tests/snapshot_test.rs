use seerdb::{DBOptions, DB};
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;

#[test]
fn test_snapshot_isolation_basic() {
    let dir = tempdir().unwrap();
    let mut opts = DBOptions::default();
    opts.data_dir = dir.path().to_path_buf();

    let db = DB::open(opts).unwrap();

    // 1. Write initial data
    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();

    // 2. Take snapshot
    let snapshot = db.snapshot().unwrap();

    // 3. Modify data after snapshot
    db.put(b"key1", b"value1_updated").unwrap();
    db.delete(b"key2").unwrap();
    db.put(b"key3", b"value3_new").unwrap();

    // 4. Verify snapshot sees original state
    assert_eq!(
        snapshot.get(b"key1").unwrap(),
        Some(bytes::Bytes::from("value1"))
    );
    assert_eq!(
        snapshot.get(b"key2").unwrap(),
        Some(bytes::Bytes::from("value2"))
    );
    assert_eq!(snapshot.get(b"key3").unwrap(), None);

    // 5. Verify DB sees new state
    assert_eq!(
        db.get(b"key1").unwrap(),
        Some(bytes::Bytes::from("value1_updated"))
    );
    assert_eq!(db.get(b"key2").unwrap(), None);
    assert_eq!(
        db.get(b"key3").unwrap(),
        Some(bytes::Bytes::from("value3_new"))
    );
}

#[test]
fn test_snapshot_range_iteration() {
    let dir = tempdir().unwrap();
    let mut opts = DBOptions::default();
    opts.data_dir = dir.path().to_path_buf();

    let db = DB::open(opts).unwrap();

    // 1. Write data: key1, key3
    db.put(b"key1", b"value1").unwrap();
    db.put(b"key3", b"value3").unwrap();

    // 2. Take snapshot
    let snapshot = db.snapshot().unwrap();

    // 3. Insert key2 (between key1 and key3)
    db.put(b"key2", b"value2").unwrap();

    // 4. Iterate snapshot - should only see key1, key3
    let items: Vec<_> = snapshot.range(b"", None).unwrap().collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].as_ref().unwrap().0, bytes::Bytes::from("key1"));
    assert_eq!(items[1].as_ref().unwrap().0, bytes::Bytes::from("key3"));

    // 5. Iterate DB - should see key1, key2, key3
    let db_items: Vec<_> = db.range(b"", None).unwrap().collect();
    assert_eq!(db_items.len(), 3);
}

#[test]
fn test_concurrent_snapshot() {
    let dir = tempdir().unwrap();
    let mut opts = DBOptions::default();
    opts.data_dir = dir.path().to_path_buf();
    opts.background_flush = true; // Enable background flush to stress test locking

    let db = Arc::new(DB::open(opts).unwrap());

    let db_clone = db.clone();
    let handle = thread::spawn(move || {
        for i in 0..1000 {
            db_clone
                .put(format!("key{}", i).as_bytes(), b"value")
                .unwrap();
            if i % 100 == 0 {
                // Take snapshot periodically
                let _snap = db_clone.snapshot().unwrap();
            }
        }
    });

    handle.join().unwrap();

    // Just ensuring no panics/deadlocks
    assert!(true);
}
