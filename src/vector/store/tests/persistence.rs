use super::super::*;
use super::random_vector;

#[test]
fn test_open_new_database() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test-oadb");

    // Open new database
    let mut store = VectorStore::open(&db_path).unwrap();
    assert!(store.is_persistent());
    assert_eq!(store.len(), 0);

    // Insert some vectors
    store
        .set(
            "doc1".to_string(),
            random_vector(128, 0),
            serde_json::json!({"title": "Doc 1"}),
        )
        .unwrap();

    store
        .set(
            "doc2".to_string(),
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
        let mut store = VectorStore::open(&db_path).unwrap();

        store
            .set(
                "vec1".to_string(),
                random_vector(128, 10),
                serde_json::json!({"category": "A", "score": 0.95}),
            )
            .unwrap();

        store
            .set(
                "vec2".to_string(),
                random_vector(128, 20),
                serde_json::json!({"category": "B", "score": 0.85}),
            )
            .unwrap();

        store
            .set(
                "vec3".to_string(),
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
        let mut store = VectorStore::open(&db_path).unwrap();

        store
            .set(
                "keep".to_string(),
                random_vector(128, 1),
                serde_json::json!({}),
            )
            .unwrap();
        store
            .set(
                "delete_me".to_string(),
                random_vector(128, 2),
                serde_json::json!({}),
            )
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
        let mut store = VectorStore::open(&db_path).unwrap();

        for i in 0..100 {
            store
                .set(
                    format!("vec{i}"),
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

mod incremental_tests {
    use super::super::super::*;
    use super::super::random_vector;

    #[test]
    fn test_incremental_set_batch() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

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
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

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
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

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
        let mut store = VectorStore::open_with_dimensions(dir.path(), 4).unwrap();

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
        let mut store = VectorStore::open_with_dimensions(&db_path, 4).unwrap();
        store
            .set(
                "vec1".to_string(),
                Vector::new(vec![1.0, 2.0, 3.0, 4.0]),
                serde_json::json!({"key": "value"}),
            )
            .unwrap();
        // No flush - just drop
    }

    // Check WAL file
    let wal_path = db_path.with_extension("wal");
    let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    println!("WAL file size: {} bytes", wal_size);
    assert!(wal_size > 0, "WAL file should not be empty after insert");

    // Reopen and verify
    {
        let store = VectorStore::open(&db_path).unwrap();
        assert_eq!(store.len(), 1, "Should have 1 vector after WAL replay");
    }
}
