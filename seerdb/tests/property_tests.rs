use bytes::Bytes;
use proptest::prelude::*;
use seerdb::{DBOptions, DB};
use std::path::PathBuf;
use tempfile::TempDir;

proptest! {
    /// Property: Put then Get returns the same value
    ///
    /// For any key-value pair, if we put(k,v) then get(k) should return Some(v).
    /// This is the fundamental correctness property of a key-value store.
    #[test]
    fn prop_put_then_get_returns_same_value(
        key in prop::collection::vec(any::<u8>(), 1..64),
        value in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(&key, &value).unwrap();
        let result = db.get(&key).unwrap();

        prop_assert_eq!(result, Some(Bytes::from(value)));
    }

    /// Property: Delete then Get returns None
    ///
    /// After deleting a key, get(k) should return None.
    /// Tests that delete actually removes the key.
    #[test]
    fn prop_delete_then_get_returns_none(
        key in prop::collection::vec(any::<u8>(), 1..64),
        value in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Put the key first
        db.put(&key, &value).unwrap();

        // Delete it
        db.delete(&key).unwrap();

        // Should return None
        let result = db.get(&key).unwrap();
        prop_assert_eq!(result, None);
    }

    /// Property: Get non-existent key returns None
    ///
    /// Getting a key that was never put should return None.
    #[test]
    fn prop_get_nonexistent_returns_none(
        key in prop::collection::vec(any::<u8>(), 1..64),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let result = db.get(&key).unwrap();
        prop_assert_eq!(result, None);
    }

    /// Property: Overwrite preserves latest value
    ///
    /// If we put(k,v1) then put(k,v2), get(k) should return v2.
    /// Tests that overwrites work correctly.
    #[test]
    fn prop_overwrite_preserves_latest(
        key in prop::collection::vec(any::<u8>(), 1..64),
        value1 in prop::collection::vec(any::<u8>(), 0..512),
        value2 in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(&key, &value1).unwrap();
        db.put(&key, &value2).unwrap();

        let result = db.get(&key).unwrap();
        prop_assert_eq!(result, Some(Bytes::from(value2)));
    }

    /// Property: Flush preserves all data
    ///
    /// After flushing, all previously written keys should still be readable.
    #[test]
    fn prop_flush_preserves_data(
        keys in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..64),
            1..20
        ),
        values in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 0..512),
            1..20
        ),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Put all key-value pairs
        for (key, value) in keys.iter().zip(values.iter()) {
            db.put(key, value).unwrap();
        }

        // Flush to disk
        db.flush().unwrap();

        // All keys should still be readable
        for (key, value) in keys.iter().zip(values.iter()) {
            let result = db.get(key).unwrap();
            prop_assert_eq!(result, Some(Bytes::from(value.clone())));
        }
    }

    /// Property: Reopen preserves persisted data
    ///
    /// After closing and reopening the database, all flushed data should still exist.
    /// This tests durability and recovery.
    #[test]
    fn prop_reopen_preserves_persisted_data(
        keys in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 1..64),
            1..10
        ),
        values in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 0..512),
            1..10
        ),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = PathBuf::from(temp_dir.path());

        {
            let opts = DBOptions {
                data_dir: db_path.clone(),
                ..Default::default()
            };
            let db = DB::open(opts).unwrap();

            // Put all key-value pairs
            for (key, value) in keys.iter().zip(values.iter()) {
                db.put(key, value).unwrap();
            }

            // Flush to ensure data is persisted
            db.flush().unwrap();

            // Drop database (close it)
        }

        // Reopen database
        let opts = DBOptions {
            data_dir: db_path,
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // All keys should still exist
        for (key, value) in keys.iter().zip(values.iter()) {
            let result = db.get(key).unwrap();
            prop_assert_eq!(result, Some(Bytes::from(value.clone())));
        }
    }

    /// Property: Empty value is valid
    ///
    /// Putting an empty value should work correctly.
    #[test]
    fn prop_empty_value_is_valid(
        key in prop::collection::vec(any::<u8>(), 1..64),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(&key, []).unwrap();
        let result = db.get(&key).unwrap();

        prop_assert_eq!(result, Some(Bytes::from(vec![])));
    }

    /// Property: Binary data (non-UTF8) works
    ///
    /// Keys and values don't need to be UTF-8 strings.
    /// Arbitrary binary data should work.
    #[test]
    fn prop_binary_data_works(
        key in prop::collection::vec(any::<u8>(), 1..64),
        value in prop::collection::vec(any::<u8>(), 0..512),
    ) {
        let temp_dir = TempDir::new().unwrap();
        let opts = DBOptions {
            data_dir: PathBuf::from(temp_dir.path()),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(&key, &value).unwrap();
        let result = db.get(&key).unwrap();

        prop_assert_eq!(result, Some(Bytes::from(value)));
    }
}
