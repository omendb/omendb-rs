// Iterator and range scan tests
// Tests: basic iteration, range scans, seeking, edge cases, concurrent modifications

use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_empty_iteration() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Empty database should have no entries
    let mut iter = db.iter().unwrap();
    assert!(iter.next().is_none());
}

#[test]
fn test_single_item_iteration() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();

    let mut iter = db.iter().unwrap();
    let (k, v) = iter.next().unwrap().unwrap();
    assert_eq!(k, b"key1".as_slice());
    assert_eq!(v, b"value1".as_slice());
    assert!(iter.next().is_none());
}

#[test]
fn test_ordered_iteration() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Insert out of order
    db.put(b"key3", b"value3").unwrap();
    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();

    // Iterator should return in sorted order
    let items: Vec<_> = db.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0].0, b"key1".as_slice());
    assert_eq!(items[1].0, b"key2".as_slice());
    assert_eq!(items[2].0, b"key3".as_slice());
}

#[test]
fn test_range_scan() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Insert keys a-z
    for i in b'a'..=b'z' {
        let key = vec![i];
        let value = vec![i; 10];
        db.put(&key, &value).unwrap();
    }

    // Scan range [d, g)
    let items: Vec<_> = db
        .range(b"d", Some(b"g"))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(items.len(), 3); // d, e, f
    assert_eq!(items[0].0, b"d".as_slice());
    assert_eq!(items[2].0, b"f".as_slice());
}

/*
#[test]
fn test_seek_to_key() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key3", b"value3").unwrap();
    db.put(b"key5", b"value5").unwrap();

    // Seek to key3
    // TODO: Uncomment when seek API is public
    // let mut iter = db.iter();
    // iter.seek(b"key3");
    // let (k, v) = iter.next().unwrap();
    // assert_eq!(k, b"key3");
}

#[test]
fn test_seek_to_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key5", b"value5").unwrap();

    // Seek to key3 (doesn't exist) - should position at key5
    // TODO: Uncomment when seek API is public
    // let mut iter = db.iter();
    // iter.seek(b"key3");
    // let (k, v) = iter.next().unwrap();
    // assert_eq!(k, b"key5"); // Next key >= key3
}
*/

#[test]
fn test_iteration_across_sstables() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024, // Small memtable to force multiple SSTables
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write enough data to span multiple SSTables
    for i in 0..1000 {
        let key = format!("key_{:04}", i);
        let value = vec![b'v'; 100];
        db.put(key.as_bytes(), &value).unwrap();
    }

    db.flush().unwrap();

    // Iterator should work across all SSTables
    let count = db.iter().unwrap().count();
    assert_eq!(count, 1000);
}

#[test]
fn test_iteration_includes_memtable() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write to SSTable
    db.put(b"key1", b"value1").unwrap();
    db.flush().unwrap();

    // Write to memtable (not flushed)
    db.put(b"key2", b"value2").unwrap();

    // Iterator should see both
    let items: Vec<_> = db.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(items.len(), 2);
}

#[test]
fn test_iteration_with_deletes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();
    db.put(b"key3", b"value3").unwrap();

    db.delete(b"key2").unwrap();

    // Iterator should skip deleted keys
    let items: Vec<_> = db.iter().unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(items.len(), 2); // Only key1 and key3
    assert_eq!(items[0].0, b"key1".as_slice());
    assert_eq!(items[1].0, b"key3".as_slice());
}

#[test]
fn test_concurrent_iteration_and_writes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Pre-populate
    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    let iter = db.iter().unwrap();

    // Write new data while iterating
    // Note: This test is racy if we expect iter to NOT see new writes.
    // Snapshot isolation (when implemented) will guarantee this.
    // For now, we just ensure it doesn't crash.
    for i in 100..200 {
        db.put(format!("key_{:03}", i).as_bytes(), b"new_value")
            .unwrap();
    }

    // Iterator behavior: may or may not see new writes (snapshot semantics)
    let count = iter.count();
    assert!(count >= 100); // At least original data
}

#[test]
fn test_iteration_during_compaction() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024,
        background_compaction: true,
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write enough to trigger compaction
    for i in 0..5000 {
        db.put(format!("key_{:05}", i).as_bytes(), &vec![b'v'; 100])
            .unwrap();
    }

    // Iterate while compaction may be running
    let count = db.iter().unwrap().count();
    assert_eq!(count, 5000);
}

/*
#[test]
fn test_reverse_iteration() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();
    db.put(b"key2", b"value2").unwrap();
    db.put(b"key3", b"value3").unwrap();

    // Reverse iteration
    // TODO: Uncomment when reverse iterator API is public
    // let items: Vec<_> = db.iter().rev().collect();
    // assert_eq!(items[0].0, b"key3");
    // assert_eq!(items[2].0, b"key1");
}
*/

#[test]
fn test_iterator_invalidation_on_flush() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    db.put(b"key1", b"value1").unwrap();

    let mut iter = db.iter().unwrap();

    // Flush changes SSTable structure
    db.flush().unwrap();

    // Iterator should still work (snapshot semantics)
    let (k, v) = iter.next().unwrap().unwrap();
    assert_eq!(k, b"key1".as_slice());
    assert_eq!(v, b"value1".as_slice());
}
