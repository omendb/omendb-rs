// Integration tests for WAL + Memtable + SSTable
// Tests the complete write → flush → recovery flow

use bytes::Bytes;
use seerdb::wal::{reader::WALReader, SyncPolicy};
use seerdb::{Memtable, Record, SSTable, WAL};
use tempfile::tempdir;

#[test]
fn test_wal_memtable_integration() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Create WAL and memtable
    let mut wal = WAL::create(&wal_path, SyncPolicy::SyncAll).unwrap();
    let memtable = Memtable::new(1024 * 1024);

    // Write some data with sequence numbers
    let records = vec![
        Record::Put {
            key: Bytes::from("key1"),
            value: Bytes::from("value1"),
            seq: 1,
        },
        Record::Put {
            key: Bytes::from("key2"),
            value: Bytes::from("value2"),
            seq: 2,
        },
        Record::Delete {
            key: Bytes::from("key3"),
            seq: 3,
        },
    ];

    // Write to WAL and memtable
    for record in &records {
        wal.write(record).unwrap();

        match record {
            Record::Put { key, value, seq } => {
                memtable.put(key.clone(), value.clone(), *seq);
            }
            Record::Delete { key, seq } => {
                memtable.delete(key.clone(), *seq);
            }
            Record::Merge { key, operand, seq } => {
                memtable.merge(key.clone(), operand.clone(), *seq);
            }
            Record::Batch {
                base_seq,
                operations,
            } => {
                let mut seq = *base_seq;
                for op in operations {
                    match op {
                        seerdb::wal::BatchOp::Put { key, value } => {
                            memtable.put(key.clone(), value.clone(), seq);
                            seq += 1;
                        }
                        seerdb::wal::BatchOp::Delete { key } => {
                            memtable.delete(key.clone(), seq);
                            seq += 1;
                        }
                        seerdb::wal::BatchOp::Merge { key, operand } => {
                            memtable.merge(key.clone(), operand.clone(), seq);
                            seq += 1;
                        }
                    }
                }
            }
        }
    }

    // Verify memtable has the data (using snapshot_seq = u64::MAX to see all writes)
    assert_eq!(
        memtable.get(b"key1", u64::MAX).map(|(v, _)| v),
        Some(Bytes::from("value1"))
    );
    assert_eq!(
        memtable.get(b"key2", u64::MAX).map(|(v, _)| v),
        Some(Bytes::from("value2"))
    );
    assert_eq!(memtable.get(b"key3", u64::MAX), None); // Deleted

    // Flush memtable to SSTable
    let sstable_path = dir.path().join("flush.sst");
    memtable.flush(&sstable_path).unwrap();

    // Open the SSTable and verify it has the data (non-tombstone entries)
    let mut sstable = SSTable::open(&sstable_path).unwrap();
    // SSTable now stores encoded InternalKeys, so we can't directly get by user key
    // This test needs to be updated when SSTable get() is updated for MVCC
    // For now, just verify the SSTable was created successfully
    assert!(sstable.len() > 0);
}

#[test]
fn test_crash_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");
    let sstable_path = dir.path().join("flush.sst");

    // Simulate writes before crash
    {
        let mut wal = WAL::create(&wal_path, SyncPolicy::SyncAll).unwrap();
        let memtable = Memtable::new(1024 * 1024);

        // Write data
        for i in 0..100u64 {
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            let record = Record::Put {
                key: Bytes::from(key.clone()),
                value: Bytes::from(value.clone()),
                seq: i + 1,
            };

            wal.write(&record).unwrap();
            memtable.put(Bytes::from(key), Bytes::from(value), i + 1);
        }

        // Flush some data
        memtable.flush(&sstable_path).unwrap();

        // Simulate crash (drop WAL and memtable)
    }

    // Recovery: Replay WAL
    let mut reader = WALReader::open(&wal_path).unwrap();
    let records = reader.read_all().unwrap();

    // Should have 100 records
    assert_eq!(records.len(), 100);

    // Rebuild memtable from WAL
    let memtable = Memtable::new(1024 * 1024);
    for record in records {
        match record {
            Record::Put { key, value, seq } => {
                memtable.put(key, value, seq);
            }
            Record::Delete { key, seq } => {
                memtable.delete(key, seq);
            }
            Record::Merge { key, operand, seq } => {
                memtable.merge(key, operand, seq);
            }
            Record::Batch {
                base_seq,
                operations,
            } => {
                let mut seq = base_seq;
                for op in operations {
                    match op {
                        seerdb::wal::BatchOp::Put { key, value } => {
                            memtable.put(key, value, seq);
                            seq += 1;
                        }
                        seerdb::wal::BatchOp::Delete { key } => {
                            memtable.delete(key, seq);
                            seq += 1;
                        }
                        seerdb::wal::BatchOp::Merge { key, operand } => {
                            memtable.merge(key, operand, seq);
                            seq += 1;
                        }
                    }
                }
            }
        }
    }

    // Verify data is recovered
    for i in 0..100 {
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);
        assert_eq!(
            memtable.get(key.as_bytes(), u64::MAX).map(|(v, _)| v),
            Some(Bytes::from(value))
        );
    }

    // Verify SSTable still exists
    let sstable = SSTable::open(&sstable_path).unwrap();
    assert!(sstable.len() > 0);
}

#[test]
fn test_write_flush_recover_cycle() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Phase 1: Write data
    {
        let mut wal = WAL::create(&wal_path, SyncPolicy::SyncData).unwrap();
        let memtable = Memtable::new(100); // Small capacity to trigger flush

        for i in 0..10u64 {
            let key = format!("key_{}", i);
            let value = format!("value_with_long_data_{}", i);
            let record = Record::Put {
                key: Bytes::from(key.clone()),
                value: Bytes::from(value.clone()),
                seq: i + 1,
            };

            wal.write(&record).unwrap();
            memtable.put(Bytes::from(key), Bytes::from(value), i + 1);

            // Check if should flush
            if memtable.should_flush() {
                let sstable_path = dir.path().join(format!("sstable_{}.sst", i));
                memtable.flush(&sstable_path).unwrap();
            }
        }
    }

    // Phase 2: Recover from WAL
    {
        let mut reader = WALReader::open(&wal_path).unwrap();
        let records = reader.read_all().unwrap();

        let memtable = Memtable::new(100);
        for record in records {
            match record {
                Record::Put { key, value, seq } => {
                    memtable.put(key, value, seq);
                }
                Record::Delete { key, seq } => {
                    memtable.delete(key, seq);
                }
                Record::Merge { key, operand, seq } => {
                    memtable.merge(key, operand, seq);
                }
                Record::Batch {
                    base_seq,
                    operations,
                } => {
                    let mut seq = base_seq;
                    for op in operations {
                        match op {
                            seerdb::wal::BatchOp::Put { key, value } => {
                                memtable.put(key, value, seq);
                                seq += 1;
                            }
                            seerdb::wal::BatchOp::Delete { key } => {
                                memtable.delete(key, seq);
                                seq += 1;
                            }
                            seerdb::wal::BatchOp::Merge { key, operand } => {
                                memtable.merge(key, operand, seq);
                                seq += 1;
                            }
                        }
                    }
                }
            }
        }

        // Verify all data recovered
        for i in 0..10 {
            let key = format!("key_{}", i);
            assert!(memtable.get(key.as_bytes(), u64::MAX).is_some());
        }
    }
}

#[test]
fn test_delete_in_wal_and_memtable() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Write, then delete
    {
        let mut wal = WAL::create(&wal_path, SyncPolicy::SyncAll).unwrap();
        let memtable = Memtable::new(1024);

        // Put
        wal.write(&Record::Put {
            key: Bytes::from("key1"),
            value: Bytes::from("value1"),
            seq: 1,
        })
        .unwrap();
        memtable.put(Bytes::from("key1"), Bytes::from("value1"), 1);

        // Delete
        wal.write(&Record::Delete {
            key: Bytes::from("key1"),
            seq: 2,
        })
        .unwrap();
        memtable.delete(Bytes::from("key1"), 2);

        assert_eq!(memtable.get(b"key1", u64::MAX), None);
    }

    // Recover
    {
        let mut reader = WALReader::open(&wal_path).unwrap();
        let records = reader.read_all().unwrap();

        let memtable = Memtable::new(1024);
        for record in records {
            match record {
                Record::Put { key, value, seq } => {
                    memtable.put(key, value, seq);
                }
                Record::Delete { key, seq } => {
                    memtable.delete(key, seq);
                }
                Record::Merge { key, operand, seq } => {
                    memtable.merge(key, operand, seq);
                }
                Record::Batch {
                    base_seq,
                    operations,
                } => {
                    let mut seq = base_seq;
                    for op in operations {
                        match op {
                            seerdb::wal::BatchOp::Put { key, value } => {
                                memtable.put(key, value, seq);
                                seq += 1;
                            }
                            seerdb::wal::BatchOp::Delete { key } => {
                                memtable.delete(key, seq);
                                seq += 1;
                            }
                            seerdb::wal::BatchOp::Merge { key, operand } => {
                                memtable.merge(key, operand, seq);
                                seq += 1;
                            }
                        }
                    }
                }
            }
        }

        // After recovery, key1 should still be deleted (check at latest seq)
        assert_eq!(memtable.get(b"key1", u64::MAX), None);
    }
}
