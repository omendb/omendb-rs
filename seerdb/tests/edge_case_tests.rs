use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_empty_value() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Empty value should be valid
    db.put(b"key", b"").unwrap();
    let result = db.get(b"key").unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b""[..]));

    // Flush and reopen
    db.flush().unwrap();
    drop(db);

    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();
    let result = db.get(b"key").unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b""[..]));
}

#[test]
fn test_single_byte_key() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Single byte key should work
    db.put(b"k", b"value").unwrap();
    let result = db.get(b"k").unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b"value"[..]));
}

#[test]
fn test_large_key() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // 64KB key (maximum recommended size)
    let large_key = vec![b'k'; 64 * 1024];
    db.put(&large_key, b"value").unwrap();
    let result = db.get(&large_key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b"value"[..]));
}

#[test]
fn test_large_value() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // 1MB value
    let large_value = vec![b'v'; 1024 * 1024];
    db.put(b"key", &large_value).unwrap();
    let result = db.get(b"key").unwrap();
    assert_eq!(result.as_ref().map(|v| v.len()), Some(1024 * 1024));
}

#[test]
fn test_binary_data_with_null_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Key and value with null bytes
    let key = b"key\x00with\x00nulls";
    let value = b"value\x00\x00\x00with\x00nulls";

    db.put(key, value).unwrap();
    let result = db.get(key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&value[..]));

    // Flush and reopen to test persistence
    db.flush().unwrap();
    drop(db);

    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();
    let result = db.get(key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&value[..]));
}

#[test]
fn test_special_characters() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Various special characters
    let test_cases = vec![
        (
            b"key\nwith\nnewlines" as &[u8],
            b"value\nwith\nnewlines" as &[u8],
        ),
        (b"key\twith\ttabs", b"value\twith\ttabs"),
        (b"key\rwith\rcarriage", b"value\rwith\rcarriage"),
        (b"key with spaces", b"value with spaces"),
        (b"key!@#$%^&*()", b"value!@#$%^&*()"),
        (b"key\x01\x02\x03", b"value\x01\x02\x03"), // Control characters
    ];

    for (key, value) in test_cases {
        db.put(key, value).unwrap();
        let result = db.get(key).unwrap();
        assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(value));
    }
}

#[test]
fn test_utf8_and_non_utf8() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Valid UTF-8
    let utf8_key = "键🔑".as_bytes();
    let utf8_value = "值📊".as_bytes();
    db.put(utf8_key, utf8_value).unwrap();
    let result = db.get(utf8_key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(utf8_value));

    // Invalid UTF-8 (random bytes)
    let invalid_utf8 = vec![0xFF, 0xFE, 0xFD, 0xFC];
    db.put(&invalid_utf8, &invalid_utf8).unwrap();
    let result = db.get(&invalid_utf8).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&invalid_utf8[..]));
}

#[test]
fn test_all_zero_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    let zero_key = vec![0u8; 32];
    let zero_value = vec![0u8; 64];

    db.put(&zero_key, &zero_value).unwrap();
    let result = db.get(&zero_key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.len()), Some(64));
}

#[test]
fn test_all_max_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    let max_key = vec![0xFFu8; 32];
    let max_value = vec![0xFFu8; 64];

    db.put(&max_key, &max_value).unwrap();
    let result = db.get(&max_key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.len()), Some(64));
}

#[test]
fn test_sequential_bytes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Test all possible byte values
    let key: Vec<u8> = (0..=255).collect();
    let value: Vec<u8> = (0..=255).rev().collect();

    db.put(&key, &value).unwrap();
    let result = db.get(&key).unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&value[..]));
}

#[test]
fn test_many_small_kvs() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // 10,000 small key-value pairs
    for i in 0..10_000 {
        let key = format!("k{:05}", i);
        let value = format!("v{:05}", i);
        db.put(key.as_bytes(), value.as_bytes()).unwrap();
    }

    db.flush().unwrap();

    // Verify all keys
    for i in 0..10_000 {
        let key = format!("k{:05}", i);
        let expected = format!("v{:05}", i);
        let result = db.get(key.as_bytes()).unwrap();
        assert_eq!(
            result.as_ref().map(|v| std::str::from_utf8(v).unwrap()),
            Some(expected.as_str())
        );
    }
}

#[test]
fn test_overwrite_many_times() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Overwrite the same key 1000 times
    for i in 0..1000 {
        db.put(b"key", format!("value{}", i).as_bytes()).unwrap();
    }

    let result = db.get(b"key").unwrap();
    assert_eq!(
        result.as_ref().map(|v| std::str::from_utf8(v).unwrap()),
        Some("value999")
    );
}

#[test]
fn test_delete_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Deleting a non-existent key should not error
    db.delete(b"nonexistent").unwrap();

    // Get should still return None
    let result = db.get(b"nonexistent").unwrap();
    assert_eq!(result, None);
}

#[test]
fn test_delete_then_reinsert() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Put, delete, put again
    db.put(b"key", b"value1").unwrap();
    db.delete(b"key").unwrap();
    db.put(b"key", b"value2").unwrap();

    let result = db.get(b"key").unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b"value2"[..]));

    // Flush and verify persistence
    db.flush().unwrap();
    drop(db);

    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();
    let result = db.get(b"key").unwrap();
    assert_eq!(result.as_ref().map(|v| v.as_ref()), Some(&b"value2"[..]));
}

#[test]
fn test_rapid_put_delete_cycles() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Rapidly cycle put/delete 100 times
    for i in 0..100 {
        db.put(b"key", format!("value{}", i).as_bytes()).unwrap();
        if i % 2 == 0 {
            db.delete(b"key").unwrap();
        }
    }

    // Key should exist (100 is even, so last operation was put)
    let result = db.get(b"key").unwrap();
    assert_eq!(
        result.as_ref().map(|v| std::str::from_utf8(v).unwrap()),
        Some("value99")
    );
}

#[test]
fn test_boundary_flush_sizes() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        memtable_capacity: 1024, // Very small memtable
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Write enough to trigger multiple flushes
    for i in 0..100 {
        let value = vec![b'v'; 100]; // 100 bytes per value
        db.put(format!("key{:03}", i).as_bytes(), &value).unwrap();
    }

    // All keys should be retrievable
    for i in 0..100 {
        let result = db.get(format!("key{:03}", i).as_bytes()).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 100);
    }
}

#[test]
fn test_identical_keys_different_cases() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Keys are case-sensitive (binary comparison)
    db.put(b"Key", b"value1").unwrap();
    db.put(b"key", b"value2").unwrap();
    db.put(b"KEY", b"value3").unwrap();

    assert_eq!(
        db.get(b"Key").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"value1"[..])
    );
    assert_eq!(
        db.get(b"key").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"value2"[..])
    );
    assert_eq!(
        db.get(b"KEY").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"value3"[..])
    );
}

#[test]
fn test_similar_keys() {
    let temp_dir = TempDir::new().unwrap();
    let opts = DBOptions {
        data_dir: PathBuf::from(temp_dir.path()),
        ..Default::default()
    };
    let db = DB::open(opts).unwrap();

    // Keys that differ by one character
    db.put(b"user:1", b"alice").unwrap();
    db.put(b"user:2", b"bob").unwrap();
    db.put(b"user:10", b"charlie").unwrap();
    db.put(b"user:11", b"dave").unwrap();

    assert_eq!(
        db.get(b"user:1").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"alice"[..])
    );
    assert_eq!(
        db.get(b"user:2").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"bob"[..])
    );
    assert_eq!(
        db.get(b"user:10").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"charlie"[..])
    );
    assert_eq!(
        db.get(b"user:11").unwrap().as_ref().map(|v| v.as_ref()),
        Some(&b"dave"[..])
    );
}
