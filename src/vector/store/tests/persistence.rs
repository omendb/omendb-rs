use super::super::*;
use super::random_vector;
use crate::text::{TextSearchConfig, TokenizerPreset};
use crate::vector::sparse::SparseVector;
use crate::vector::store::edge_store::EdgeDirection;

#[test]
fn test_open_new_database() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test-oadb");

    // Open new database
    let store = VectorStore::open(&db_path).unwrap();
    assert!(store.is_persistent());
    assert_eq!(store.len(), 0);

    // Insert some vectors
    store
        .set(
            "doc1",
            random_vector(128, 0),
            serde_json::json!({"title": "Doc 1"}),
        )
        .unwrap();

    store
        .set(
            "doc2",
            random_vector(128, 1),
            serde_json::json!({"title": "Doc 2"}),
        )
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(store.get("doc1").is_some());
}

#[test]
fn test_persistent_roundtrip() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("roundtrip-oadb");

    // Create and populate store
    {
        let store = VectorStore::open(&db_path).unwrap();

        store
            .set(
                "vec1",
                random_vector(128, 10),
                serde_json::json!({"category": "A", "score": 0.95}),
            )
            .unwrap();

        store
            .set(
                "vec2",
                random_vector(128, 20),
                serde_json::json!({"category": "B", "score": 0.85}),
            )
            .unwrap();

        store
            .set(
                "vec3",
                random_vector(128, 30),
                serde_json::json!({"category": "A", "score": 0.75}),
            )
            .unwrap();

        // Flush to ensure data is on disk
        store.flush().unwrap();
    }

    // Reopen and verify data
    {
        let store = VectorStore::open(&db_path).unwrap();

        assert_eq!(store.len(), 3);

        // Verify vec1
        let (vec1, meta1) = store.get("vec1").unwrap();
        assert_eq!(vec1.data, random_vector(128, 10).data);
        assert_eq!(meta1["category"], "A");
        assert_eq!(meta1["score"], 0.95);

        // Verify vec2
        let (vec2, meta2) = store.get("vec2").unwrap();
        assert_eq!(vec2.data, random_vector(128, 20).data);
        assert_eq!(meta2["category"], "B");

        // Verify vec3
        assert!(store.get("vec3").is_some());
    }
}

#[test]
fn test_persistent_delete() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("delete-oadb");

    // Create, populate, and delete
    {
        let store = VectorStore::open(&db_path).unwrap();

        store
            .set("keep", random_vector(128, 1), serde_json::json!({}))
            .unwrap();
        store
            .set("delete_me", random_vector(128, 2), serde_json::json!({}))
            .unwrap();

        assert_eq!(store.len(), 2);

        // Delete one
        store.delete("delete_me").unwrap();
        assert!(store.get("delete_me").is_none());

        store.flush().unwrap();
    }

    // Reopen and verify deletion persisted
    {
        let store = VectorStore::open(&db_path).unwrap();

        // Only "keep" should be accessible
        assert!(store.get("keep").is_some());
        assert!(store.get("delete_me").is_none());
    }
}

#[test]
fn test_persistent_search() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("search-oadb");

    // Create and populate
    {
        let store = VectorStore::open(&db_path).unwrap();

        for i in 0..100 {
            store
                .set(
                    &format!("vec{i}"),
                    random_vector(128, i),
                    serde_json::json!({"index": i}),
                )
                .unwrap();
        }

        store.flush().unwrap();
    }

    // Reopen and search
    {
        let store = VectorStore::open(&db_path).unwrap();

        assert_eq!(store.len(), 100);

        // Search should work
        let query = random_vector(128, 50);
        let results = store.knn_search(&query, 10).unwrap();

        // Verify we get results
        assert_eq!(results.len(), 10, "Should return 10 results");

        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i].1 >= results[i - 1].1,
                "Results should be sorted by distance"
            );
        }
    }
}

#[test]
fn test_text_tokenizer_config_persists_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("text-tokenizer-persist");

    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .enable_text_search_with_config(Some(TextSearchConfig {
                writer_buffer_mb: 20,
                tokenizer: TokenizerPreset::Code,
            }))
            .unwrap();
        store
            .set_with_text(
                "doc1",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                "getUserProfile HTTPClient",
                serde_json::json!({"kind": "code"}),
            )
            .unwrap();
        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&db_path).unwrap();
        let results = store.search_text("user", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1");

        let results = store.search_text("client", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1");
    }
}

mod incremental_tests {
    use super::super::super::*;

    #[test]
    fn test_incremental_set_batch() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        // Single item inserts
        store
            .set_batch(vec![(
                "vec1".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        store
            .set_batch(vec![(
                "vec2".to_string(),
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Batch insert
        store
            .set_batch(vec![
                (
                    "vec3".to_string(),
                    Vector::new(vec![0.0, 0.0, 1.0, 0.0]),
                    serde_json::json!({}),
                ),
                (
                    "vec4".to_string(),
                    Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
                    serde_json::json!({}),
                ),
            ])
            .unwrap();

        // Another batch
        store
            .set_batch(vec![
                (
                    "vec5".to_string(),
                    Vector::new(vec![0.5, 0.5, 0.0, 0.0]),
                    serde_json::json!({}),
                ),
                (
                    "vec6".to_string(),
                    Vector::new(vec![0.0, 0.5, 0.5, 0.0]),
                    serde_json::json!({}),
                ),
            ])
            .unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(
            results.len(),
            6,
            "Incremental inserts must all be searchable"
        );
    }

    #[test]
    fn test_interleaved_insert_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        let mut total_inserted = 0;

        // Insert 10 batches of 10 vectors, searching after each batch
        for batch in 0..10 {
            let vectors: Vec<_> = (0..10)
                .map(|i| {
                    let id = batch * 10 + i;
                    let mut v = vec![0.0; 4];
                    v[id % 4] = 1.0 + (id as f32 * 0.01);
                    (format!("vec{id}"), Vector::new(v), serde_json::json!({}))
                })
                .collect();

            store.set_batch(vectors).unwrap();
            total_inserted += 10;

            // Search after each batch
            let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
            let results = store.knn_search(&query, total_inserted + 10).unwrap();
            assert_eq!(
                results.len(),
                total_inserted,
                "After batch {}, expected {} results but got {}",
                batch,
                total_inserted,
                results.len()
            );
        }

        // Final verification
        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 200).unwrap();
        assert_eq!(results.len(), 100, "All 100 vectors must be searchable");
    }

    #[test]
    fn test_batch_then_single_insert() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        // Batch insert
        let batch: Vec<_> = (0..50)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                (format!("batch{i}"), Vector::new(v), serde_json::json!({}))
            })
            .collect();
        store.set_batch(batch).unwrap();

        // Search to "activate" the index
        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let results = store.knn_search(&query, 100).unwrap();
        assert_eq!(results.len(), 50, "Batch vectors must be searchable");

        // Single insert after search
        store
            .set_batch(vec![(
                "single".to_string(),
                Vector::new(vec![0.99, 0.01, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search again - new vector must be reachable
        let results = store.knn_search(&query, 100).unwrap();
        assert_eq!(
            results.len(),
            51,
            "New vector after search must be reachable"
        );

        // The new vector should appear in search results
        // Index 50 is the single insert (0-49 were batch)
        let found = results.iter().any(|(idx, _)| *idx == 50);
        assert!(found, "Newly inserted vector must appear in search results");
    }

    #[test]
    fn test_insert_search_cycle_from_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

        let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

        // Search empty index
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 0, "Empty index should return no results");

        // First insert
        store
            .set_batch(vec![(
                "first".to_string(),
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find first vector
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 1, "Should find first vector");

        // Second insert
        store
            .set_batch(vec![(
                "second".to_string(),
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find both
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 2, "Should find both vectors");

        // Third insert
        store
            .set_batch(vec![(
                "third".to_string(),
                Vector::new(vec![0.5, 0.5, 0.0, 0.0]),
                serde_json::json!({}),
            )])
            .unwrap();

        // Search should find all three
        let results = store.knn_search(&query, 10).unwrap();
        assert_eq!(results.len(), 3, "Should find all three vectors");
    }
}

#[test]
fn test_set_writes_to_wal() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_wal_write");

    // Create and insert
    {
        let store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "vec1",
                Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
                serde_json::json!({"key": "value"}),
            )
            .unwrap();
        // No flush - just drop
    }

    // Check WAL file
    let wal_path = db_path.with_extension("wal");
    let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    println!("WAL file size: {wal_size} bytes");
    assert!(wal_size > 0, "WAL file should not be empty after insert");

    // Reopen and verify
    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(store.len(), 1, "Should have 1 vector after WAL replay");
    }
}

#[test]
fn test_recovery_skips_stale_wal_after_full_flush_publish() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("stale_wal_after_publish");
    let wal_path = db_path.with_extension("wal");
    let wal_meta_path = db_path.with_extension("wal.meta");

    let stale_wal = {
        let store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "vec1",
                Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
                serde_json::json!({"version": 1}),
            )
            .unwrap();
        let bytes = std::fs::read(&wal_path).unwrap();
        store.flush().unwrap();
        bytes
    };

    std::fs::write(&wal_path, stale_wal).unwrap();
    let _ = std::fs::remove_file(&wal_meta_path);

    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 1);
    assert_eq!(store.records.slot_count(), 1);
    assert_eq!(store.records.get_slot("vec1"), Some(0));

    let (vec, meta) = store.get("vec1").unwrap();
    assert_eq!(vec.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(meta["version"], 1);
}

#[test]
fn test_recovery_replays_wal_if_manifest_publish_did_not_happen() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("manifest_publish_recovery");
    let omen_path = db_path.with_extension("omen");
    let wal_path = db_path.with_extension("wal");
    let wal_meta_path = db_path.with_extension("wal.meta");

    {
        let store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "base",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();
    }

    let old_manifest = std::fs::read(&omen_path).unwrap();

    let (pending_wal, pending_meta) = {
        let store = VectorStore::open(&db_path).unwrap();
        store
            .set(
                "new_doc",
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({"phase": "wal"}),
            )
            .unwrap();
        let wal_bytes = std::fs::read(&wal_path).unwrap();
        let meta_bytes = std::fs::read(&wal_meta_path).unwrap();
        store.flush().unwrap();
        (wal_bytes, meta_bytes)
    };

    std::fs::write(&omen_path, old_manifest).unwrap();
    std::fs::write(&wal_path, pending_wal).unwrap();
    std::fs::write(&wal_meta_path, pending_meta).unwrap();

    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 2);
    assert!(store.get("base").is_some());
    assert!(store.get("new_doc").is_some());
    assert!(store.records.get_slot("new_doc").is_some());
}

#[test]
fn test_vector_only_checkpoint_recovers_legitimate_zero_vector() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("zero_vector_recovery");

    {
        let store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "base",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();

        store
            .set(
                "zero",
                Vector::new(vec![0.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "zero"}),
            )
            .unwrap();
        store.checkpoint_wal().unwrap();
    }

    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.len(), 2);

    let (vector, metadata) = store.get("zero").unwrap();
    assert_eq!(vector.data, vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(metadata["phase"], "zero");
}

#[test]
fn test_vector_only_checkpoint_recovers_sparse_only_record() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("sparse_placeholder_recovery");

    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "base",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();

        store
            .set_sparse(
                "sparse_doc",
                SparseVector::from_pairs(vec![(42, 1.5), (99, 0.5)]).unwrap(),
                serde_json::json!({"phase": "sparse"}),
            )
            .unwrap();
        store.checkpoint_wal().unwrap();
    }

    let store = VectorStore::open(&db_path).unwrap();
    assert!(store.contains("sparse_doc"));
    assert!(store.get("sparse_doc").is_none());
}

#[test]
fn test_vector_only_checkpoint_preserves_sparse_state() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("sparse_state_recovery");

    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "base",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();

        store
            .set_sparse(
                "sparse_doc",
                SparseVector::from_pairs(vec![(42, 1.5), (99, 0.5)]).unwrap(),
                serde_json::json!({"phase": "sparse"}),
            )
            .unwrap();
        store.checkpoint_wal().unwrap();
    }

    let store = VectorStore::open(&db_path).unwrap();
    assert!(store.has_sparse());

    let query = SparseVector::from_pairs(vec![(42, 1.0)]).unwrap();
    let results = store.sparse_search(&query, 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "sparse_doc");
}

#[test]
fn test_sparse_wal_recovery_without_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("sparse_wal_recovery");

    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "base",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();

        store
            .set_sparse(
                "sparse_doc",
                SparseVector::from_pairs(vec![(42, 1.5), (99, 0.5)]).unwrap(),
                serde_json::json!({"phase": "wal"}),
            )
            .unwrap();
    }

    let store = VectorStore::open(&db_path).unwrap();
    let query = SparseVector::from_pairs(vec![(42, 1.0)]).unwrap();
    let results = store.sparse_search(&query, 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "sparse_doc");
    assert_eq!(results[0].metadata["phase"], "wal");
}

#[test]
fn test_vector_only_checkpoint_preserves_edges() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("edge_state_recovery");

    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "a",
                Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store
            .set(
                "b",
                Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
                serde_json::json!({"phase": "base"}),
            )
            .unwrap();
        store.flush().unwrap();

        store
            .add_edge(
                "a",
                "b",
                "link",
                0.7,
                Some(serde_json::json!({"phase": "edge"})),
            )
            .unwrap();
        store
            .set(
                "pad",
                Vector::new(vec![0.0, 0.0, 1.0, 0.0]),
                serde_json::json!({"phase": "pad"}),
            )
            .unwrap();
        store.checkpoint_wal().unwrap();
    }

    let store = VectorStore::open(&db_path).unwrap();
    assert_eq!(store.edge_count(), 1);

    let edges = store.get_edges("a", EdgeDirection::Outgoing, None);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].to_id, "b");
    assert_eq!(edges[0].edge_type, "link");
    assert_eq!(edges[0].weight, 0.7);
    assert_eq!(
        edges[0].metadata,
        Some(serde_json::json!({"phase": "edge"}))
    );
}
