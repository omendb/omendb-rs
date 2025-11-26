// Configuration edge case tests
// Tests unusual/extreme configuration values
// Critical for stability: handle all valid configs safely

use seerdb::{DBOptions, SyncPolicy, DB};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_tiny_memtable_capacity() {
    // Test with very small memtable (1KB)
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024, // 1KB
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Should trigger frequent flushes
    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), &vec![b'v'; 100])
            .unwrap();
    }

    // Verify all data present
    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_large_memtable_capacity() {
    // Test with very large memtable (1GB)
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024 * 1024 * 1024, // 1GB
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write data that fits in memtable
    for i in 0..1000 {
        db.put(format!("key_{:04}", i).as_bytes(), b"value")
            .unwrap();
    }

    // Should all be in memtable (no flush)
    for i in 0..1000 {
        assert!(db
            .get(format!("key_{:04}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_base_level_size_extreme_values() {
    // Test with tiny base level
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        base_level_size: 1024, // 1KB
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    // Verify data
    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_size_ratio_extreme_values() {
    // Test with large size ratio
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        size_ratio: 100, // Very large
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_single_level() {
    // Test with only 1 level
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        num_levels: 1,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_many_levels() {
    // Test with many levels
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        num_levels: 20,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_vlog_threshold_zero() {
    // Test with vlog threshold = 0 (all values go to vlog)
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        vlog_threshold: Some(0),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    db.put(b"key", b"value").unwrap();
    db.flush().unwrap();

    assert_eq!(db.get(b"key").unwrap().unwrap().as_ref(), b"value");
}

#[test]
fn test_vlog_threshold_very_large() {
    // Test with very large threshold (nothing goes to vlog)
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        vlog_threshold: Some(1024 * 1024 * 1024), // 1GB
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    db.put(b"key", &vec![b'v'; 1000]).unwrap();
    db.flush().unwrap();

    assert_eq!(db.get(b"key").unwrap().unwrap().len(), 1000);
}

#[test]
fn test_all_sync_policies() {
    // Test all sync policies work
    let policies = vec![SyncPolicy::None, SyncPolicy::SyncData, SyncPolicy::SyncAll];

    for policy in policies {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            wal_sync_policy: policy,
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        db.put(b"key", b"value").unwrap();
        assert_eq!(db.get(b"key").unwrap().unwrap().as_ref(), b"value");
    }
}

#[test]
fn test_background_compaction_disabled() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        background_compaction: false,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_background_compaction_enabled() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        background_compaction: true,
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    for i in 0..100 {
        db.put(format!("key_{:03}", i).as_bytes(), b"value")
            .unwrap();
    }

    db.flush().unwrap();

    // Wait for potential background compaction
    std::thread::sleep(std::time::Duration::from_secs(1));

    for i in 0..100 {
        assert!(db
            .get(format!("key_{:03}", i).as_bytes())
            .unwrap()
            .is_some());
    }
}

#[test]
fn test_empty_database() {
    // Test opening database with no data
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Should handle empty DB gracefully
    assert!(db.get(b"nonexistent").unwrap().is_none());

    // Flush empty memtable should be safe
    db.flush().unwrap();

    assert!(db.get(b"nonexistent").unwrap().is_none());
}

#[test]
fn test_single_key() {
    // Test database with only one key
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    db.put(b"only_key", b"only_value").unwrap();
    db.flush().unwrap();

    assert_eq!(
        db.get(b"only_key").unwrap().unwrap().as_ref(),
        b"only_value"
    );
    assert!(db.get(b"other").unwrap().is_none());
}

#[test]
fn test_many_small_values() {
    // Stress test with many small values
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write 10k small values
    for i in 0..10000 {
        db.put(format!("k{:05}", i).as_bytes(), b"v").unwrap();
    }

    db.flush().unwrap();

    // Verify random sample
    for i in (0..10000).step_by(100) {
        assert_eq!(
            db.get(format!("k{:05}", i).as_bytes())
                .unwrap()
                .unwrap()
                .as_ref(),
            b"v"
        );
    }
}

#[test]
fn test_few_large_values() {
    // Test with few very large values
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        vlog_threshold: Some(1024), // Use vlog for large values
        ..Default::default()
    };

    let db = DB::open(opts).unwrap();

    // Write 10 large values (100KB each)
    for i in 0..10 {
        let value = vec![b'v'; 100 * 1024];
        db.put(format!("large_{}", i).as_bytes(), &value).unwrap();
    }

    db.flush().unwrap();

    // Verify
    for i in 0..10 {
        let value = db.get(format!("large_{}", i).as_bytes()).unwrap().unwrap();
        assert_eq!(value.len(), 100 * 1024);
    }
}
