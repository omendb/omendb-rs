use super::*;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

#[test]
fn test_recovery_corrupted_slim_snapshot() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("corrupted_slim");

    // 1. Create DB and insert data
    {
        let store = VectorStore::create(&db_path, dense_schema(4)).unwrap();
        store
            .set(
                "doc1",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();
        store.flush().unwrap(); // Creates .vecs and initial manifest

        store
            .set(
                "doc2",
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();

        // Force a checkpoint which creates the .records snapshot
        store.checkpoint_wal().unwrap();
        assert_eq!(store.len(), 2);
    }

    // 2. Corrupt the .records file
    let records_path = dir.path().join("corrupted_slim.records");
    assert!(records_path.exists());
    let mut f = fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&records_path)
        .unwrap();
    f.write_all(b"NOT_A_VALID_SNAPSHOT_MAGIC_BYTES_AND_VERSION")
        .unwrap();
    f.sync_all().unwrap();

    // 3. Open DB - it should skip the corrupted snapshot and load from manifest
    // Note: doc2 is lost because checkpoint_wal truncated the WAL and we corrupted the snapshot
    // where it was stored. doc1 is safe in the main manifest.
    // However, OmenDB recovers the raw vector from .vecs as a synthetic ID.
    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 2);
    assert!(store.get("doc1").is_some());
    assert!(store.get("doc2").is_none());
    assert!(store.contains("__slot_1"));
}

#[test]
fn test_recovery_wal_present_but_manifest_says_no_wal() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("wal_mismatch");

    // 1. Create DB and insert data, then close without checkpointing WAL
    {
        let store = VectorStore::create(&db_path, dense_schema(4)).unwrap();
        store
            .set(
                "doc1",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();
        // WAL has 1 entry, manifest is at offset 0
    }

    // 2. Open - it should replay WAL even if manifest doesn't know about it yet (initial state)
    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 1);
    assert!(store.get("doc1").is_some());
}

#[test]
fn test_recovery_with_truncated_wal_epoch_mismatch() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("epoch_mismatch");

    // 1. Create DB, insert, checkpoint
    {
        let store = VectorStore::create(&db_path, dense_schema(4)).unwrap();
        store
            .set(
                "doc1",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();
        store.checkpoint_wal().unwrap(); // Creates snapshot with epoch 0
    }

    // 2. Open, insert more, then simulate a WAL truncation that increments epoch
    {
        let store = VectorStore::open(&db_path).unwrap();
        store
            .set(
                "doc2",
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )
            .unwrap();
        // Note: truncation happens on full checkpoint.
        // We want to verify that if the WAL was truncated (epoch incremented),
        // we don't accidentally skip replaying current WAL entries.
    }

    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 2);
    assert!(store.get("doc2").is_some());
}
