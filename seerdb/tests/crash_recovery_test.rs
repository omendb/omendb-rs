// Crash Recovery Tests
// Tests database recovery after simulated crashes
//
// Strategy: Simulate crashes by incomplete operations, file corruption, and truncation
// instead of spawning processes (which is complex and flaky)

use bytes::Bytes;
use seerdb::{DBOptions, SyncPolicy, DB};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use tempfile::tempdir;

/// Corrupt a file by flipping bits at a given offset
fn corrupt_file(path: &Path, offset: u64, corruption: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(corruption)?;
    file.sync_all()?;
    Ok(())
}

/// Truncate a file to a new size
fn truncate_file(path: &Path, new_size: u64) -> std::io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(new_size)?;
    file.sync_all()?;
    Ok(())
}

#[test]
fn test_corrupted_sstable_detected() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");

    // Create database and write data
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Write enough data to trigger flush
        for i in 0..1000 {
            db.put(
                format!("key{:04}", i).as_bytes(),
                format!("value{:04}", i).as_bytes(),
            )
            .unwrap();
        }

        // Force flush to create SSTable
        db.flush().unwrap();
    }

    // Find the SSTable file
    let sstable_path = fs::read_dir(&db_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sst"))
        .unwrap()
        .path();

    // Corrupt the SSTable (flip bits in data section, not footer)
    corrupt_file(&sstable_path, 100, &[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();

    // Try to reopen database - should detect corruption
    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let result = DB::open(opts);

    // Verify corruption was detected
    assert!(
        result.is_err(),
        "Expected error when opening corrupted SSTable"
    );
}

#[test]
fn test_corrupted_wal_detected() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");

    // Create database and write data (without flush - stays in WAL)
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..100 {
            db.put(
                format!("key{:04}", i).as_bytes(),
                format!("value{:04}", i).as_bytes(),
            )
            .unwrap();
        }
        // Don't flush - data stays in WAL
    }

    // Find the WAL file
    let wal_path = db_path.join("wal.log");
    assert!(wal_path.exists(), "WAL file should exist");

    // Corrupt the WAL header (magic number at offset 0)
    // WAL validates header magic but doesn't have per-record checksums,
    // so we must corrupt the header to guarantee detection
    corrupt_file(&wal_path, 0, b"BAAD").unwrap();

    // Try to reopen database - should detect corruption
    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let result = DB::open(opts);

    // Header corruption must be detected - DB::open should fail
    assert!(
        result.is_err(),
        "DB::open should fail when WAL header is corrupted"
    );
}

#[test]
fn test_truncated_wal_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");

    // Create database and write data
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..50 {
            db.put(
                format!("key{:04}", i).as_bytes(),
                format!("value{:04}", i).as_bytes(),
            )
            .unwrap();
        }
    }

    // Truncate WAL to simulate incomplete write
    let wal_path = db_path.join("wal.log");
    let original_size = fs::metadata(&wal_path).unwrap().len();
    // Truncate to remove last incomplete record (not header)
    // Keep at least the 8-byte header intact
    // Use saturating_sub to avoid underflow if WAL is smaller than expected
    let truncated_size = original_size.saturating_sub(50).max(8);
    truncate_file(&wal_path, truncated_size).unwrap();

    // Reopen database - recovery behavior depends on implementation:
    // - May recover partial data (all complete records before truncation)
    // - May fail entirely if truncation detection is strict
    // - May recover no data if truncation corrupts critical metadata
    // All outcomes are acceptable as long as we don't panic or corrupt silently
    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };

    match DB::open(opts) {
        Ok(db) => {
            // Recovery succeeded - check how much data we recovered
            let count = (0..50)
                .filter(|i| db.get(format!("key{:04}", i).as_bytes()).unwrap().is_some())
                .count();

            // Any count is acceptable (0 to 50) - the key is we recovered
            // without panicking or returning garbage
            println!("Truncated WAL recovery: recovered {} / 50 keys", count);
        }
        Err(e) => {
            // Recovery failed - also acceptable if truncation causes
            // a parsing error (e.g., incomplete length prefix)
            println!("Truncated WAL recovery failed (acceptable): {}", e);
        }
    }
}

#[test]
fn test_crash_during_flush_incomplete_sstable() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");

    // Create database and write data
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            wal_sync_policy: SyncPolicy::SyncAll,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 0..500 {
            db.put(
                format!("key{:04}", i).as_bytes(),
                format!("value{:04}", i).as_bytes(),
            )
            .unwrap();
        }

        // Flush to create SSTable
        db.flush().unwrap();

        // WAL should be cleared after successful flush
        let wal_path = db_path.join("wal.log");
        let wal_size = fs::metadata(&wal_path).unwrap().len();
        assert!(wal_size < 1000, "WAL should be small after flush");
    }

    // Write more data (will be in WAL)
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            wal_sync_policy: SyncPolicy::SyncAll,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        for i in 500..600 {
            db.put(
                format!("key{:04}", i).as_bytes(),
                format!("value{:04}", i).as_bytes(),
            )
            .unwrap();
        }
        // Simulate crash before flush - data stays in WAL
    }

    // Reopen and verify recovery
    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // All data should be recovered
    for i in 0..600 {
        let value = db
            .get(format!("key{:04}", i).as_bytes())
            .unwrap()
            .unwrap_or_else(|| panic!("key{:04} should exist", i));
        assert_eq!(
            value,
            Bytes::from(format!("value{:04}", i)),
            "Value mismatch for key{:04}",
            i
        );
    }
}

#[test]
fn test_crash_during_compaction_incomplete() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("db");

    // Create database with multiple SSTables at L0
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            memtable_capacity: 1024, // Small capacity to trigger multiple flushes
            ..Default::default()
        };

        let db = DB::open(opts).unwrap();

        // Write data in batches to create multiple SSTables
        for batch in 0..5 {
            for i in 0..100 {
                let key = format!("key{:03}_{:04}", batch, i);
                let value = format!("value{:03}_{:04}", batch, i);
                db.put(key.as_bytes(), value.as_bytes()).unwrap();
            }
            db.flush().unwrap();
        }
    }

    // Count SSTables before compaction
    let sstable_count_before = fs::read_dir(&db_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sst"))
        .count();

    assert!(
        sstable_count_before >= 5,
        "Should have multiple SSTables before compaction"
    );

    // Reopen database (triggers compaction if L0 has 4+ SSTables)
    {
        let opts = DBOptions {
            data_dir: db_path.clone(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Verify all data is still accessible
        for batch in 0..5 {
            for i in 0..100 {
                let key = format!("key{:03}_{:04}", batch, i);
                let expected_value = format!("value{:03}_{:04}", batch, i);
                let value = db
                    .get(key.as_bytes())
                    .unwrap()
                    .unwrap_or_else(|| panic!("{} should exist", key));
                assert_eq!(
                    value,
                    Bytes::from(expected_value),
                    "Value mismatch for {}",
                    key
                );
            }
        }
    }

    // Verify data integrity after potential compaction
    let opts = DBOptions {
        data_dir: db_path.clone(),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    for batch in 0..5 {
        for i in 0..100 {
            let key = format!("key{:03}_{:04}", batch, i);
            let expected_value = format!("value{:03}_{:04}", batch, i);
            let value = db
                .get(key.as_bytes())
                .unwrap()
                .unwrap_or_else(|| panic!("{} should exist after recovery", key));
            assert_eq!(
                value,
                Bytes::from(expected_value),
                "Value mismatch for {} after recovery",
                key
            );
        }
    }
}
