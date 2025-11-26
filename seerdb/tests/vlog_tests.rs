//! Comprehensive tests for VLog implementation
//!
//! Tests cover:
//! - Corruption detection (CRC validation)
//! - Truncation handling (partial writes)
//! - Header validation (magic number, version)
//! - Concurrent reads (thread safety)
//! - Edge cases (empty values, large values, boundary conditions)

use bytes::Bytes;
use seerdb::vlog::{VLog, VLogError, VLogRecord};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

// VLog constants (from vlog/mod.rs)
const MAGIC: u32 = 0x564C4F47; // "VLOG"
const VERSION: u32 = 0x00000001;
const HEADER_SIZE: u64 = 8;

// Corruption detection tests

#[test]
fn test_corrupted_crc_detected() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    let pointer = vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Corrupt the value bytes (flip last byte)
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .open(&vlog_path)
        .unwrap();

    // Seek to value position and corrupt it
    file.seek(SeekFrom::Start(pointer.offset + pointer.length as u64 - 1))
        .unwrap();
    file.write_all(&[0xFF]).unwrap();
    file.sync_all().unwrap();
    drop(file);

    // Try to read corrupted value
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let result = vlog.read_record(HEADER_SIZE);

    // Should detect CRC mismatch
    assert!(matches!(result, Err(VLogError::CrcMismatch { .. })));
}

#[test]
fn test_corrupted_key_detected() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Corrupt the key bytes (flip a byte in the key)
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .open(&vlog_path)
        .unwrap();

    // Key starts at offset: HEADER_SIZE + 4 (key_len)
    file.seek(SeekFrom::Start(HEADER_SIZE + 4)).unwrap();
    file.write_all(&[0xFF]).unwrap();
    file.sync_all().unwrap();
    drop(file);

    // Try to read corrupted record
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let result = vlog.read_record(HEADER_SIZE);

    // Should detect CRC mismatch
    assert!(matches!(result, Err(VLogError::CrcMismatch { .. })));
}

#[test]
fn test_valid_crc_passes() {
    // Sanity check: valid data should pass CRC check
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Read should succeed (no corruption)
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let (record, _) = vlog.read_record(HEADER_SIZE).unwrap();

    assert_eq!(record.key, Bytes::from("key1"));
    assert_eq!(record.value, Bytes::from("value1"));
}

#[test]
fn test_corrupted_length_field() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Corrupt key_len field (make it huge)
    let mut file = OpenOptions::new()
        .write(true)
        .read(true)
        .open(&vlog_path)
        .unwrap();

    file.seek(SeekFrom::Start(HEADER_SIZE)).unwrap();
    file.write_all(&0xFFFFFFFFu32.to_le_bytes()).unwrap();
    file.sync_all().unwrap();
    drop(file);

    // Try to read - should fail (invalid format or CRC mismatch)
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let result = vlog.read_record(HEADER_SIZE);

    assert!(result.is_err());
}

// Truncation handling tests

#[test]
fn test_truncated_record_detected() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Truncate file to cut off CRC bytes
    let file = OpenOptions::new().write(true).open(&vlog_path).unwrap();

    // Truncate to remove last 2 bytes (incomplete CRC)
    let current_size = file.metadata().unwrap().len();
    file.set_len(current_size - 2).unwrap();
    drop(file);

    // Try to read truncated record
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let result = vlog.read_record(HEADER_SIZE);

    // Should fail (not enough bytes to read CRC)
    assert!(result.is_err());
}

#[test]
fn test_truncated_value_detected() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();
    vlog.append(b"key1", b"value_long").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Truncate file to cut off half the value
    let file = OpenOptions::new().write(true).open(&vlog_path).unwrap();

    // Truncate to remove last 5 bytes of value + CRC
    let current_size = file.metadata().unwrap().len();
    file.set_len(current_size - 9).unwrap(); // 5 bytes of "value_long" + 4 CRC
    drop(file);

    // Try to read truncated record
    let mut vlog = VLog::open(&vlog_path).unwrap();
    let result = vlog.read_record(HEADER_SIZE);

    // Should fail (not enough bytes)
    assert!(result.is_err());
}

#[test]
fn test_partial_write_recovery() {
    // Simulate crash during write (file has incomplete record at end)
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Write first record (complete)
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();

    // Write second record but don't sync (simulate crash)
    vlog.append(b"key2", b"value2").unwrap();
    // No sync - buffered write
    drop(vlog);

    // Truncate file to simulate partial write
    let file = OpenOptions::new().write(true).open(&vlog_path).unwrap();

    // Truncate to remove last few bytes (incomplete second record)
    let current_size = file.metadata().unwrap().len();
    file.set_len(current_size - 5).unwrap();
    drop(file);

    // Reopen vLog
    let mut vlog = VLog::open(&vlog_path).unwrap();

    // First record should be readable
    let (record1, _) = vlog.read_record(HEADER_SIZE).unwrap();
    assert_eq!(record1.key, Bytes::from("key1"));
    assert_eq!(record1.value, Bytes::from("value1"));

    // Second record should fail (truncated)
    // Note: head position is set to file size, so we can continue appending
    assert!(vlog.head() < current_size);
}

// Header validation tests

#[test]
fn test_invalid_magic_number() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    // Create file with wrong magic number
    let mut file = File::create(&vlog_path).unwrap();
    file.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap(); // Wrong magic
    file.write_all(&VERSION.to_le_bytes()).unwrap();
    file.sync_all().unwrap();
    drop(file);

    // Try to open
    let result = VLog::open(&vlog_path);

    assert!(matches!(result, Err(VLogError::InvalidFormat)));
}

#[test]
fn test_invalid_version() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    // Create file with wrong version
    let mut file = File::create(&vlog_path).unwrap();
    file.write_all(&MAGIC.to_le_bytes()).unwrap();
    file.write_all(&0xFFFFFFFFu32.to_le_bytes()).unwrap(); // Wrong version
    file.sync_all().unwrap();
    drop(file);

    // Try to open
    let result = VLog::open(&vlog_path);

    assert!(matches!(result, Err(VLogError::InvalidFormat)));
}

#[test]
fn test_valid_header_accepted() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    // Create vLog (should write valid header)
    let vlog = VLog::create(&vlog_path).unwrap();
    drop(vlog);

    // Reopen should succeed
    let vlog = VLog::open(&vlog_path);
    assert!(vlog.is_ok());
}

#[test]
fn test_truncated_header() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    // Create file with incomplete header (only 4 bytes)
    let mut file = File::create(&vlog_path).unwrap();
    file.write_all(&MAGIC.to_le_bytes()).unwrap();
    // Missing version bytes
    drop(file);

    // Try to open
    let result = VLog::open(&vlog_path);

    assert!(result.is_err());
}

#[test]
fn test_empty_file() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    // Create empty file
    File::create(&vlog_path).unwrap();

    // Try to open
    let result = VLog::open(&vlog_path);

    assert!(result.is_err());
}

// Concurrent reads tests

#[test]
fn test_concurrent_reads_same_value() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Write values
    let p1 = vlog.append(b"key1", b"value1").unwrap();
    let p2 = vlog.append(b"key2", b"value2").unwrap();
    vlog.sync().unwrap();
    drop(vlog);

    // Spawn 4 threads reading same values
    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let vlog_path = vlog_path.clone();
            let barrier = barrier.clone();
            let p1_copy = p1;
            let p2_copy = p2;

            thread::spawn(move || {
                barrier.wait();

                let mut vlog = VLog::open(&vlog_path).unwrap();

                // Each thread reads both values 100 times
                for _ in 0..100 {
                    let v1 = vlog.read(p1_copy).unwrap();
                    let v2 = vlog.read(p2_copy).unwrap();

                    assert_eq!(v1, Bytes::from("value1"));
                    assert_eq!(v2, Bytes::from("value2"));
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_reads_different_values() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Write 100 values
    let mut pointers = Vec::new();
    for i in 0..100 {
        let key = format!("key_{:03}", i);
        let value = format!("value_{:03}", i);
        let pointer = vlog.append(key.as_bytes(), value.as_bytes()).unwrap();
        pointers.push((i, pointer));
    }
    vlog.sync().unwrap();
    drop(vlog);

    let pointers = Arc::new(pointers);
    let barrier = Arc::new(Barrier::new(4));

    // Spawn 4 threads, each reading different ranges
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let vlog_path = vlog_path.clone();
            let pointers = pointers.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                barrier.wait();

                let mut vlog = VLog::open(&vlog_path).unwrap();

                // Each thread reads its assigned range
                let start = thread_id * 25;
                let end = start + 25;

                for &(i, pointer) in &pointers[start..end] {
                    let value = vlog.read(pointer).unwrap();
                    let expected = format!("value_{:03}", i);
                    assert_eq!(value, Bytes::from(expected));
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_read_record() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Write records
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);
        vlog.append(key.as_bytes(), value.as_bytes()).unwrap();
    }
    vlog.sync().unwrap();
    drop(vlog);

    let barrier = Arc::new(Barrier::new(4));

    // Spawn 4 threads reading records sequentially
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let vlog_path = vlog_path.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                barrier.wait();

                let mut vlog = VLog::open(&vlog_path).unwrap();

                // Each thread reads all records
                let mut offset = HEADER_SIZE;
                for i in 0..10 {
                    let (record, next_offset) = vlog.read_record(offset).unwrap();
                    let expected_key = format!("key_{}", i);
                    let expected_value = format!("value_{}", i);

                    assert_eq!(record.key, Bytes::from(expected_key));
                    assert_eq!(record.value, Bytes::from(expected_value));

                    offset = next_offset;
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

// Edge cases

#[test]
fn test_empty_key() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Empty key is valid
    let pointer = vlog.append(b"", b"value1").unwrap();
    vlog.sync().unwrap();

    let value = vlog.read(pointer).unwrap();
    assert_eq!(value, Bytes::from("value1"));

    // Read record should also work
    let (record, _) = vlog.read_record(HEADER_SIZE).unwrap();
    assert_eq!(record.key, Bytes::from(""));
    assert_eq!(record.value, Bytes::from("value1"));
}

#[test]
fn test_empty_value() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Empty value is valid (tombstone use case)
    let pointer = vlog.append(b"key1", b"").unwrap();
    assert_eq!(pointer.length, 0);
    vlog.sync().unwrap();

    let value = vlog.read(pointer).unwrap();
    assert_eq!(value, Bytes::from(""));

    // Read record should also work
    let (record, _) = vlog.read_record(HEADER_SIZE).unwrap();
    assert_eq!(record.key, Bytes::from("key1"));
    assert_eq!(record.value, Bytes::from(""));
}

#[test]
fn test_very_large_value() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // 1MB value (stress test)
    let large_value = vec![b'x'; 1024 * 1024];
    let pointer = vlog.append(b"key1", &large_value).unwrap();
    vlog.sync().unwrap();

    let value = vlog.read(pointer).unwrap();
    assert_eq!(value.len(), 1024 * 1024);
    assert_eq!(value, Bytes::from(large_value));
}

#[test]
fn test_many_small_values() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Write 1000 small values
    let mut pointers = Vec::new();
    for i in 0..1000 {
        let key = format!("k{}", i);
        let value = format!("v{}", i);
        let pointer = vlog.append(key.as_bytes(), value.as_bytes()).unwrap();
        pointers.push((i, pointer));
    }
    vlog.sync().unwrap();

    // Read all values back
    for (i, pointer) in pointers {
        let value = vlog.read(pointer).unwrap();
        let expected = format!("v{}", i);
        assert_eq!(value, Bytes::from(expected));
    }
}

#[test]
fn test_vlog_head_tail_tracking() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Initial state
    assert_eq!(vlog.head(), HEADER_SIZE);
    assert_eq!(vlog.tail(), HEADER_SIZE);

    // After append
    vlog.append(b"key1", b"value1").unwrap();
    assert!(vlog.head() > HEADER_SIZE);
    assert_eq!(vlog.tail(), HEADER_SIZE); // Tail doesn't move on append

    // Set tail (GC simulation)
    vlog.set_tail(vlog.head());
    assert_eq!(vlog.tail(), vlog.head());
}

#[test]
fn test_vlog_size_tracking() {
    let dir = TempDir::new().unwrap();
    let vlog_path = dir.path().join("test.vlog");

    let mut vlog = VLog::create(&vlog_path).unwrap();

    // Initial size (header only)
    assert_eq!(vlog.size().unwrap(), HEADER_SIZE);

    // After append
    vlog.append(b"key1", b"value1").unwrap();
    vlog.sync().unwrap();

    let size_after = vlog.size().unwrap();
    assert!(size_after > HEADER_SIZE);

    // Size should match head position
    assert_eq!(size_after, vlog.head());
}

// VLogRecord encode/decode edge cases

#[test]
fn test_record_decode_invalid_length() {
    // Record too short
    let short_data = Bytes::from(vec![0u8; 4]); // Only key_len, no data
    let result = VLogRecord::decode(short_data);
    assert!(matches!(result, Err(VLogError::InvalidRecordFormat)));
}

#[test]
fn test_record_decode_truncated_key() {
    // key_len says 100 bytes but data is only 20 bytes
    let mut data = vec![0u8; 20];
    data[0..4].copy_from_slice(&100u32.to_le_bytes()); // key_len = 100

    let result = VLogRecord::decode(Bytes::from(data));
    assert!(matches!(result, Err(VLogError::InvalidRecordFormat)));
}

#[test]
fn test_record_decode_truncated_value() {
    let record = VLogRecord {
        key: Bytes::from("key"),
        value: Bytes::from("value"),
    };

    let encoded = record.encode();

    // Truncate off CRC bytes
    let truncated = encoded.slice(0..encoded.len() - 4);
    let result = VLogRecord::decode(truncated);

    assert!(matches!(result, Err(VLogError::InvalidRecordFormat)));
}
