use super::*;

fn random_vector(dim: usize, seed: usize) -> Vector {
    let data: Vec<f32> = (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect();
    Vector::new(data)
}

#[test]
fn test_vector_store_insert() {
    let mut store = VectorStore::new(128);

    let v1 = random_vector(128, 0);
    let v2 = random_vector(128, 1);

    let id1 = store.insert(v1).unwrap();
    let id2 = store.insert(v2).unwrap();

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(store.len(), 2);
}

#[test]
fn test_vector_store_knn_with_hnsw() {
    let mut store = VectorStore::new(128);

    // Insert some vectors
    for i in 0..100 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Query for nearest neighbors (uses HNSW)
    let query = random_vector(128, 50);
    let results = store.knn_search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
}

#[test]
fn test_vector_store_brute_force() {
    let mut store = VectorStore::new(128);

    // Insert some vectors
    for i in 0..100 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Query using brute-force
    let query = random_vector(128, 50);
    let results = store.knn_search_brute_force(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
}

#[test]
fn test_dimension_mismatch() {
    let mut store = VectorStore::new(128);
    let wrong_dim = Vector::new(vec![1.0; 64]);

    assert!(store.insert(wrong_dim).is_err());
}

#[test]
fn test_ef_search_tuning() {
    let mut store = VectorStore::new(128);

    // Insert vectors to initialize HNSW
    for i in 0..10 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Check default ef_search (fixed default: M=16, ef_construction=100, ef_search=100)
    assert_eq!(store.get_ef_search(), Some(100));

    // Tune ef_search
    store.set_ef_search(600);
    assert_eq!(store.get_ef_search(), Some(600));
}

#[test]
fn test_rebuild_index() {
    let mut store = VectorStore::new(128);

    // Insert vectors
    for i in 0..100 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Verify index exists (segments or hnsw_index)
    assert!(store.segments.is_some() || store.hnsw_index.is_some());

    // Clear the index
    store.segments = None;
    store.hnsw_index = None;
    assert!(store.segments.is_none() && store.hnsw_index.is_none());

    // Rebuild index
    store.rebuild_index().unwrap();

    // Verify index is rebuilt
    assert!(store.segments.is_some());

    // Verify search works
    let query = random_vector(128, 50);
    let results = store.knn_search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_compact_basic() {
    let mut store = VectorStore::new(128);

    // Insert 100 vectors
    for i in 0..100 {
        store
            .set(
                format!("vec{i}"),
                random_vector(128, i),
                serde_json::json!({"idx": i}),
            )
            .unwrap();
    }
    assert_eq!(store.len(), 100);

    // Delete 30 vectors
    for i in 0..30 {
        store.delete(&format!("vec{i}")).unwrap();
    }
    assert_eq!(store.len(), 70);

    // Compact - should remove 30 tombstones
    let removed = store.compact().unwrap();
    assert_eq!(removed, 30);
    assert_eq!(store.len(), 70);

    // Verify search still works
    let query = random_vector(128, 50);
    let results = store.knn_search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Verify remaining vectors accessible by ID
    for i in 30..100 {
        assert!(store.contains(&format!("vec{i}")));
    }

    // Verify deleted vectors gone
    for i in 0..30 {
        assert!(!store.contains(&format!("vec{i}")));
    }
}

#[test]
fn test_compact_empty() {
    let mut store = VectorStore::new(128);

    // Insert some vectors but don't delete any
    for i in 0..10 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Compact with no deletions - should return 0
    let removed = store.compact().unwrap();
    assert_eq!(removed, 0);
    assert_eq!(store.len(), 10);
}

#[test]
fn test_compact_all_deleted() {
    let mut store = VectorStore::new(128);

    // Insert and delete all
    for i in 0..10 {
        store
            .set(
                format!("vec{i}"),
                random_vector(128, i),
                serde_json::json!({}),
            )
            .unwrap();
    }
    for i in 0..10 {
        store.delete(&format!("vec{i}")).unwrap();
    }
    assert_eq!(store.len(), 0);

    // Compact - should remove all tombstones
    let removed = store.compact().unwrap();
    assert_eq!(removed, 10);
    assert_eq!(store.len(), 0);
}

#[test]
fn test_quantization_insert() {
    use crate::vector::QuantizationMode;

    // Create store with SQ8 quantization
    let mut store = VectorStore::new_with_quantization(128, QuantizationMode::SQ8);

    // Insert vectors
    for i in 0..50 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Verify vectors stored and quantization is enabled
    assert_eq!(store.len(), 50);
    assert!(store.segments.as_ref().is_some_and(|s| s.is_quantized()));
}

#[test]
fn test_quantization_search_accuracy() {
    use crate::vector::QuantizationMode;

    // Create store with SQ8 quantization
    let mut store = VectorStore::new_with_quantization(128, QuantizationMode::SQ8);

    // Insert vectors
    for i in 0..100 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Search with quantization (uses asymmetric HNSW)
    let query = random_vector(128, 50);
    let results = store.knn_search(&query, 10).unwrap();

    // Should still get 10 results
    assert_eq!(results.len(), 10);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
}

#[test]
fn test_quantization_batch_insert() {
    use crate::vector::QuantizationMode;

    // Create store with SQ8 quantization
    let mut store = VectorStore::new_with_quantization(128, QuantizationMode::SQ8);

    // Batch insert vectors
    let vectors: Vec<Vector> = (0..100).map(|i| random_vector(128, i)).collect();
    let ids = store.batch_insert(vectors).unwrap();

    // Verify all vectors were created and quantization is enabled
    assert_eq!(ids.len(), 100);
    assert_eq!(store.len(), 100);
    assert!(store.segments.as_ref().is_some_and(|s| s.is_quantized()));
}

#[test]
fn test_new_with_params_functional() {
    // Verify new_with_params works functionally
    let mut store = VectorStore::new_with_params(128, 16, 100, 100, Metric::L2).unwrap();

    // Insert vectors
    for i in 0..100 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Search
    let query = random_vector(128, 50);
    let results = store.knn_search(&query, 10).unwrap();

    assert_eq!(results.len(), 10);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
}

// Tests for metadata support

#[test]
fn test_insert_with_metadata() {
    let mut store = VectorStore::new(128);

    let metadata = serde_json::json!({
        "title": "Test Document",
        "author": "Alice",
        "year": 2024
    });

    let index = store
        .insert_with_metadata("doc1".to_string(), random_vector(128, 0), metadata.clone())
        .unwrap();

    assert_eq!(index, 0);
    assert!(store.contains("doc1"));
    assert_eq!(store.get_metadata_by_id("doc1"), Some(&metadata));
}

#[test]
fn test_set_insert() {
    let mut store = VectorStore::new(128);

    let metadata = serde_json::json!({"title": "Doc 1"});

    // First set should insert
    let index = store
        .set("doc1".to_string(), random_vector(128, 0), metadata.clone())
        .unwrap();

    assert_eq!(index, 0);
    assert_eq!(store.len(), 1);
}

#[test]
fn test_set_update() {
    let mut store = VectorStore::new(128);

    // Insert initial document
    store
        .set(
            "doc1".to_string(),
            random_vector(128, 0),
            serde_json::json!({"title": "Original"}),
        )
        .unwrap();

    // Upsert with same ID - creates new slot (to maintain slot == HNSW node ID)
    let index = store
        .set(
            "doc1".to_string(),
            random_vector(128, 1),
            serde_json::json!({"title": "Updated"}),
        )
        .unwrap();

    assert_eq!(index, 1); // New slot (old slot 0 marked deleted)
    assert_eq!(store.len(), 1); // Still only 1 live vector
    assert_eq!(
        store
            .get_metadata_by_id("doc1")
            .unwrap()
            .get("title")
            .unwrap(),
        "Updated"
    );
}

#[test]
fn test_delete() {
    let mut store = VectorStore::new(128);

    store
        .insert_with_metadata(
            "doc1".to_string(),
            random_vector(128, 0),
            serde_json::json!({"title": "Doc 1"}),
        )
        .unwrap();

    // Delete the document
    store.delete("doc1").unwrap();

    // Should be marked as deleted
    assert!(!store.contains("doc1"));

    // get should return None for deleted
    assert!(store.get("doc1").is_none());
}

#[test]
fn test_update() {
    let mut store = VectorStore::new(128);

    store
        .insert_with_metadata(
            "doc1".to_string(),
            random_vector(128, 0),
            serde_json::json!({"title": "Original"}),
        )
        .unwrap();

    // Update metadata only
    store
        .update(
            "doc1",
            None,
            Some(serde_json::json!({"title": "Updated", "author": "Bob"})),
        )
        .unwrap();

    let (_, metadata) = store.get("doc1").unwrap();
    assert_eq!(metadata.get("title").unwrap(), "Updated");
    assert_eq!(metadata.get("author").unwrap(), "Bob");
}

#[test]
fn test_metadata_filter_eq() {
    let filter = MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice"));

    let metadata1 = serde_json::json!({"author": "Alice"});
    let metadata2 = serde_json::json!({"author": "Bob"});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
}

#[test]
fn test_metadata_filter_gte() {
    let filter = MetadataFilter::Gte("year".to_string(), 2020.0);

    let metadata1 = serde_json::json!({"year": 2024});
    let metadata2 = serde_json::json!({"year": 2019});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
}

#[test]
fn test_metadata_filter_and() {
    let filter = MetadataFilter::And(vec![
        MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice")),
        MetadataFilter::Gte("year".to_string(), 2020.0),
    ]);

    let metadata1 = serde_json::json!({"author": "Alice", "year": 2024});
    let metadata2 = serde_json::json!({"author": "Alice", "year": 2019});
    let metadata3 = serde_json::json!({"author": "Bob", "year": 2024});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
    assert!(!filter.matches(&metadata3));
}

#[test]
fn test_search_with_filter() {
    let mut store = VectorStore::new(128);

    // Insert vectors with metadata
    store
        .set(
            "doc1".to_string(),
            random_vector(128, 0),
            serde_json::json!({"author": "Alice", "year": 2024}),
        )
        .unwrap();

    store
        .set(
            "doc2".to_string(),
            random_vector(128, 1),
            serde_json::json!({"author": "Bob", "year": 2023}),
        )
        .unwrap();

    store
        .set(
            "doc3".to_string(),
            random_vector(128, 2),
            serde_json::json!({"author": "Alice", "year": 2022}),
        )
        .unwrap();

    // Search with filter for Alice's documents
    let filter = MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice"));
    let query = random_vector(128, 0);
    let results = store.knn_search_with_filter(&query, 10, &filter).unwrap();

    // Should only return Alice's documents (doc1 and doc3)
    assert_eq!(results.len(), 2);
    for result in &results {
        assert_eq!(result.metadata.get("author").unwrap(), "Alice");
    }
}

#[test]
fn test_get() {
    let mut store = VectorStore::new(128);

    let vector = random_vector(128, 0);
    let metadata = serde_json::json!({"title": "Test"});

    store
        .insert_with_metadata("doc1".to_string(), vector.clone(), metadata.clone())
        .unwrap();

    // Get by ID
    let (retrieved_vector, retrieved_metadata) = store.get("doc1").unwrap();

    assert_eq!(retrieved_vector.data, vector.data);
    assert_eq!(retrieved_metadata, metadata);

    // Non-existent ID should return None
    assert!(store.get("nonexistent").is_none());
}

// Tests for persistent storage

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
    use super::*;

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

    /// INC-2: Interleave inserts and searches
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

    /// INC-3: Insert batch, search, single insert, search
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

    /// INC-4: Empty index -> insert -> search -> insert -> search cycle
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

// ============================================================================
// Text Search / Hybrid Search Tests
// ============================================================================

#[test]
fn test_enable_text_search() {
    let mut store = VectorStore::new(4);

    assert!(!store.has_text_search());

    store.enable_text_search().unwrap();

    assert!(store.has_text_search());

    // Enabling again should be a no-op
    store.enable_text_search().unwrap();
    assert!(store.has_text_search());
}

#[test]
fn test_set_with_text() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    let idx = store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning is awesome",
            serde_json::json!({"type": "article"}),
        )
        .unwrap();

    // Flush to commit text index changes
    store.flush().unwrap();

    assert_eq!(idx, 0);
    assert_eq!(store.len(), 1);

    // Text search should find it
    let results = store.text_search("machine", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_set_with_text_requires_enabled() {
    let mut store = VectorStore::new(4);

    // Should fail without enabling text search
    let result = store.set_with_text(
        "doc1".to_string(),
        Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
        "test text",
        serde_json::json!({}),
    );

    assert!(result.is_err());
}

#[test]
fn test_text_search_bm25() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // Add documents with different term frequencies
    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "rust programming language",
            serde_json::json!({}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2".to_string(),
            Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
            "rust rust systems programming",
            serde_json::json!({}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    // Search for "rust" - doc2 should rank higher (higher term frequency)
    let results = store.text_search("rust", 10).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "doc2"); // Higher BM25 score
    assert_eq!(results[1].0, "doc1");
}

#[test]
fn test_hybrid_search() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // doc1: similar vector, relevant text
    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: different vector, relevant text
    store
        .set_with_text(
            "doc2".to_string(),
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning models",
            serde_json::json!({}),
        )
        .unwrap();

    // doc3: similar vector, irrelevant text
    store
        .set_with_text(
            "doc3".to_string(),
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "cooking recipes",
            serde_json::json!({}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    // Query: similar to doc1/doc3 vectors, text matches doc1/doc2
    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let results = store
        .hybrid_search(&query, "machine learning", 3, None)
        .unwrap();

    assert!(!results.is_empty());

    // doc1 should rank highest (both vector similarity and text match)
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_hybrid_search_with_filter() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2024}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2".to_string(),
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2023}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let filter = MetadataFilter::Eq("year".to_string(), serde_json::json!(2024));

    let results = store
        .hybrid_search_with_filter(&query, "machine", 10, &filter, None)
        .unwrap();

    // Only doc1 should match the filter
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_text_search_options_builder() {
    let store = VectorStoreOptions::default()
        .dimensions(4)
        .text_search(true)
        .build()
        .unwrap();

    assert!(store.has_text_search());
}

#[test]
fn test_hybrid_search_empty_text() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test content",
            serde_json::json!({}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

    // Empty text query should still return vector search results
    let results = store.hybrid_search(&query, "", 10, None).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hybrid_search_alpha_weighting() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // doc1: closest vector, weak text match
    store
        .set_with_text(
            "vec_winner".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "unrelated words here",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: far vector, strong text match
    store
        .set_with_text(
            "text_winner".to_string(),
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

    // alpha=1.0: vector only - vec_winner should win
    let results = store
        .hybrid_search(&query, "machine learning", 2, Some(1.0))
        .unwrap();
    assert_eq!(results[0].0, "vec_winner");

    // alpha=0.0: text only - text_winner should win
    let results = store
        .hybrid_search(&query, "machine learning", 2, Some(0.0))
        .unwrap();
    assert_eq!(results[0].0, "text_winner");
}

#[test]
fn test_hybrid_search_k_zero() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test document",
            serde_json::json!({}),
        )
        .unwrap();
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    // k=0 should return an error (HNSW requires k > 0)
    let result = store.hybrid_search(&query, "test", 0, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("k=0"));
}

#[test]
fn test_hybrid_search_dimension_mismatch() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test document",
            serde_json::json!({}),
        )
        .unwrap();
    store.flush().unwrap();

    // Query with wrong dimension (8 instead of 4)
    let wrong_query = Vector::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = store.hybrid_search(&wrong_query, "test", 10, None);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension 8 does not match store dimension 4"));
}

#[test]
fn test_hybrid_search_large_k() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // Insert only 3 documents
    for i in 0..3 {
        store
            .set_with_text(
                format!("doc{i}"),
                Vector::new(vec![1.0, 0.0, 0.0, i as f32]),
                &format!("document {i}"),
                serde_json::json!({}),
            )
            .unwrap();
    }
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    // Request more results than available
    let results = store.hybrid_search(&query, "document", 100, None).unwrap();
    // Should return at most 3 (the number of documents)
    assert!(results.len() <= 3);
}

#[test]
fn test_hybrid_search_without_text_enabled() {
    let mut store = VectorStore::new(4);
    // Don't enable text search

    store
        .set(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            serde_json::json!({}),
        )
        .unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let result = store.hybrid_search(&query, "test", 10, None);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Text search not enabled"));
}

#[test]
fn test_hybrid_search_with_subscores() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // doc1: matches both vector and text
    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: matches text only (very different vector)
    store
        .set_with_text(
            "doc2".to_string(),
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning models",
            serde_json::json!({}),
        )
        .unwrap();

    // doc3: matches vector only (no matching text)
    store
        .set_with_text(
            "doc3".to_string(),
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "cooking recipes",
            serde_json::json!({}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let results = store
        .hybrid_search_with_subscores(&query, "machine learning", 3, None, None)
        .unwrap();

    assert_eq!(results.len(), 3);

    // doc1 should have both scores
    let doc1 = results.iter().find(|(r, _)| r.id == "doc1").unwrap();
    assert!(
        doc1.0.keyword_score.is_some(),
        "doc1 should have keyword_score"
    );
    assert!(
        doc1.0.semantic_score.is_some(),
        "doc1 should have semantic_score"
    );

    // doc2 should have keyword but possibly no semantic (if not in vector top-k)
    let doc2 = results.iter().find(|(r, _)| r.id == "doc2").unwrap();
    assert!(
        doc2.0.keyword_score.is_some(),
        "doc2 should have keyword_score"
    );

    // doc3 should have semantic but no keyword (text doesn't match "machine learning")
    let doc3 = results.iter().find(|(r, _)| r.id == "doc3").unwrap();
    assert!(
        doc3.0.semantic_score.is_some(),
        "doc3 should have semantic_score"
    );
    assert!(
        doc3.0.keyword_score.is_none(),
        "doc3 should not have keyword_score"
    );

    // doc1 should rank highest (both vector similarity and text match)
    assert_eq!(results[0].0.id, "doc1");
}

#[test]
fn test_hybrid_search_with_filter_subscores() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1".to_string(),
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2024}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2".to_string(),
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2023}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let filter = MetadataFilter::Gte("year".to_string(), 2024.0);

    let results = store
        .hybrid_search_with_filter_subscores(&query, "machine learning", 10, &filter, None, None)
        .unwrap();

    // Only doc1 should match (year >= 2024)
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "doc1");
    assert!(results[0].0.keyword_score.is_some());
    assert!(results[0].0.semantic_score.is_some());
}

// ============================================================================
// Property-Based Tests
// ============================================================================

mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Verify HNSW parameters roundtrip through save/load
        #[test]
        fn params_roundtrip(
            m in 16usize..64,
            ef_construction in 100usize..500,
            ef_search in 100usize..500,
            dimensions in 8usize..128
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.omen");

            // Create store with specific params using VectorStoreOptions
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(dimensions)
                    .m(m)
                    .ef_construction(ef_construction)
                    .ef_search(ef_search)
                    .open(&path)
                    .unwrap();

                // Insert some vectors to trigger HNSW creation
                for i in 0..10 {
                    let v = Vector::new((0..dimensions).map(|j| (i * j) as f32 * 0.1).collect());
                    let id = format!("vec_{i}");
                    store.set(id, v, serde_json::json!({})).unwrap();
                }

                store.flush().unwrap();
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert_eq!(loaded.hnsw_m, m);
            prop_assert_eq!(loaded.hnsw_ef_construction, ef_construction);
            prop_assert_eq!(loaded.hnsw_ef_search, ef_search);
            prop_assert_eq!(loaded.len(), 10);
        }

        /// Verify ID mappings stay consistent after insert/delete operations
        #[test]
        fn id_mapping_consistency(
            num_inserts in 10usize..100,
            num_deletes in 0usize..10
        ) {
            let mut store = VectorStore::new(8);

            // Insert vectors
            let mut ids = Vec::new();
            for i in 0..num_inserts {
                let id = format!("vec_{i}");
                let v = Vector::new((0..8).map(|j| (i * j) as f32 * 0.1).collect());
                store.set(id.clone(), v, serde_json::json!({})).unwrap();
                ids.push(id);
            }

            // Delete some
            let to_delete = num_deletes.min(ids.len());
            for id in ids.iter().take(to_delete) {
                store.delete(id).unwrap();
            }

            // Verify consistency: count matches
            let expected_count = num_inserts - to_delete;
            prop_assert_eq!(store.len(), expected_count);

            // Verify remaining IDs are accessible
            for id in ids.iter().skip(to_delete) {
                prop_assert!(store.contains(id));
            }
        }

        /// Verify non-quantized mode persists correctly
        #[test]
        fn non_quantized_roundtrip(
            num_vectors in 10usize..30
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("nonquant.omen");

            // Create non-quantized store
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(64)
                    .open(&path)
                    .unwrap();

                // Insert some vectors
                for i in 0..num_vectors {
                    let v = Vector::new((0..64).map(|j| (i * j) as f32 * 0.01).collect());
                    let id = format!("vec_{i}");
                    store.set(id, v, serde_json::json!({})).unwrap();
                }

                store.flush().unwrap();
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert_eq!(loaded.len(), num_vectors);
        }

        /// Verify SQ8 quantized mode persists ID mappings correctly
        ///
        /// This tests the fix for the SQ8 ID mapping corruption bug where
        /// multiple batches would overwrite previous IDs because vectors.len()
        /// was used instead of next_index counter.
        #[test]
        fn sq8_quantized_roundtrip(
            num_vectors in 10usize..30
        ) {
            use crate::vector::QuantizationMode;

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("sq8quant.omen");

            // Create SQ8 quantized store
            {
                let mut store = VectorStoreOptions::default()
                    .dimensions(64)
                    .quantization(QuantizationMode::SQ8)
                    .open(&path)
                    .unwrap();

                // Insert vectors in batches to trigger the bug
                for batch in 0..3 {
                    let batch_vectors: Vec<_> = (0..num_vectors / 3)
                        .map(|i| {
                            let idx = batch * (num_vectors / 3) + i;
                            let v = Vector::new((0..64).map(|j| (idx * j + batch) as f32 * 0.01).collect());
                            let id = format!("vec_{idx}");
                            (id, v, serde_json::json!({"batch": batch}))
                        })
                        .collect();
                    store.set_batch(batch_vectors).unwrap();
                }

                store.flush().unwrap();

                // Verify count before close
                prop_assert!(store.len() > 0, "Store should not be empty");
            }

            // Load and verify
            let loaded = VectorStore::open(&path).unwrap();
            prop_assert!(loaded.len() > 0, "Loaded store should not be empty");

            // Verify all IDs are searchable
            for batch in 0..3 {
                for i in 0..(num_vectors / 3) {
                    let idx = batch * (num_vectors / 3) + i;
                    let id = format!("vec_{idx}");
                    prop_assert!(
                        loaded.contains(&id),
                        "ID '{}' not found after reload",
                        id
                    );
                }
            }
        }
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

// ============================================================================
// Persistence Round-Trip Property Tests (tk-xvf9)
// ============================================================================

mod persistence_proptest {
    use super::*;
    use proptest::prelude::*;

    /// Generate random f32 vector data
    fn arb_vector_data(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0f32, dim)
    }

    /// Generate valid ID strings (alphanumeric, no special chars that could break parsing)
    #[allow(dead_code)]
    fn arb_id() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_]{0,15}".prop_map(|s| s)
    }

    proptest! {
        /// WAL recovery without flush - data survives via WAL replay
        #[test]
        fn wal_recovery_no_flush(
            num_vectors in 1usize..20,
            dim in 4usize..32
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wal_recovery.omen");

            let mut expected_vectors = Vec::new();

            // Insert without flush
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for i in 0..num_vectors {
                    let data: Vec<f32> = (0..dim).map(|j| (i * 100 + j) as f32 * 0.01).collect();
                    let id = format!("v{}", i);
                    store.set(id.clone(), Vector::new(data.clone()), serde_json::json!({"i": i})).unwrap();
                    expected_vectors.push((id, data, i));
                }
                // NO flush - data should survive via WAL
            }

            // Reopen and verify WAL recovery
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), num_vectors, "WAL recovery should restore all vectors");

                for (id, expected_data, idx) in &expected_vectors {
                    prop_assert!(store.contains(id), "ID not found after WAL recovery");
                    let (vec, meta) = store.get(id).unwrap();
                    prop_assert_eq!(&vec.data, expected_data, "Vector data mismatch");
                    prop_assert_eq!(meta["i"].as_u64().unwrap() as usize, *idx, "Metadata mismatch");
                }
            }
        }

        /// Mixed insert + delete with WAL recovery
        #[test]
        fn wal_recovery_with_deletes(
            num_inserts in 5usize..30,
            delete_ratio in 0.1f64..0.5
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("wal_delete.omen");

            let dim = 8;
            let num_deletes = ((num_inserts as f64) * delete_ratio) as usize;

            // Insert then delete without flush
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for i in 0..num_inserts {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
                    store.set(format!("v{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }

                // Delete first N vectors
                for i in 0..num_deletes {
                    store.delete(&format!("v{}", i)).unwrap();
                }
                // NO flush
            }

            // Reopen and verify
            {
                let store = VectorStore::open(&path).unwrap();
                let expected_count = num_inserts - num_deletes;
                prop_assert_eq!(store.len(), expected_count);

                // Deleted vectors should not exist
                for i in 0..num_deletes {
                    let id = format!("v{}", i);
                    prop_assert!(!store.contains(&id), "Deleted vector should not exist");
                }

                // Remaining vectors should exist
                for i in num_deletes..num_inserts {
                    let id = format!("v{}", i);
                    prop_assert!(store.contains(&id), "Vector should exist");
                }
            }
        }

        /// Flush + WAL recovery: data persisted via checkpoint, then more via WAL
        #[test]
        fn checkpoint_plus_wal(
            checkpoint_count in 5usize..20,
            wal_count in 1usize..10
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("checkpoint_wal.omen");
            let dim = 8;

            // Insert, flush, insert more (no flush)
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                // First batch - will be checkpointed
                for i in 0..checkpoint_count {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
                    store.set(format!("cp{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }
                store.flush().unwrap();

                // Second batch - only in WAL
                for i in 0..wal_count {
                    let data: Vec<f32> = (0..dim).map(|j| (i + j + 1000) as f32).collect();
                    store.set(format!("wal{}", i), Vector::new(data), serde_json::json!({})).unwrap();
                }
                // NO flush for second batch
            }

            // Reopen - should have both checkpoint and WAL data
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), checkpoint_count + wal_count);

                for i in 0..checkpoint_count {
                    let id = format!("cp{}", i);
                    prop_assert!(store.contains(&id));
                }
                for i in 0..wal_count {
                    let id = format!("wal{}", i);
                    prop_assert!(store.contains(&id));
                }
            }
        }

        /// Vector data integrity - exact float values preserved
        #[test]
        fn vector_data_integrity(
            values in arb_vector_data(16)
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("data_integrity.omen");

            // Store exact values
            {
                let mut store = VectorStore::open_with_dimensions(&path, 16).unwrap();
                store.set("test".to_string(), Vector::new(values.clone()), serde_json::json!({})).unwrap();
                store.flush().unwrap();
            }

            // Verify exact match after reload
            {
                let store = VectorStore::open(&path).unwrap();
                let (vec, _) = store.get("test").unwrap();
                prop_assert_eq!(vec.data.len(), values.len());
                for (i, (got, expected)) in vec.data.iter().zip(values.iter()).enumerate() {
                    prop_assert!(
                        (got - expected).abs() < f32::EPSILON,
                        "Float mismatch at index {i}: got {got}, expected {expected}"
                    );
                }
            }
        }

        /// Metadata types integrity - various JSON values roundtrip
        #[test]
        fn metadata_types_roundtrip(
            int_val in -1000i64..1000,
            float_val in -100.0f64..100.0,
            bool_val in proptest::bool::ANY,
            str_len in 1usize..20
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("meta_types.omen");

            let string_val: String = (0..str_len).map(|i| ((i % 26) as u8 + b'a') as char).collect();

            let metadata = serde_json::json!({
                "int": int_val,
                "float": float_val,
                "bool": bool_val,
                "string": string_val,
                "null": null,
                "array": [1, 2, 3],
                "nested": {"a": 1, "b": "two"}
            });

            // Store
            {
                let mut store = VectorStore::open_with_dimensions(&path, 4).unwrap();
                store.set("test".to_string(), Vector::new(vec![1.0, 2.0, 3.0, 4.0]), metadata.clone()).unwrap();
                store.flush().unwrap();
            }

            // Verify
            {
                let store = VectorStore::open(&path).unwrap();
                let (_, loaded_meta) = store.get("test").unwrap();
                prop_assert_eq!(loaded_meta["int"].as_i64().unwrap(), int_val);
                prop_assert_eq!(loaded_meta["bool"].as_bool().unwrap(), bool_val);
                prop_assert_eq!(loaded_meta["string"].as_str().unwrap(), string_val);
                prop_assert!(loaded_meta["null"].is_null());
                prop_assert_eq!(loaded_meta["array"].as_array().unwrap().len(), 3);
                prop_assert_eq!(loaded_meta["nested"]["a"].as_i64().unwrap(), 1);
                prop_assert_eq!(loaded_meta["nested"]["b"].as_str().unwrap(), "two");
            }
        }

        /// Upsert semantics - overwriting vector persists correctly
        #[test]
        fn upsert_persistence(
            num_updates in 2usize..5
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("upsert.omen");
            let dim = 4;
            let last_update = num_updates - 1;

            // Expected: last update's data (update index = num_updates - 1)
            let final_data: Vec<f32> = (0..dim).map(|i| (last_update * 100 + i) as f32).collect();

            // Insert then update multiple times
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for update in 0..num_updates {
                    let data: Vec<f32> = (0..dim).map(|i| (update * 100 + i) as f32).collect();
                    store.set("same_id".to_string(), Vector::new(data), serde_json::json!({"version": update})).unwrap();
                }
                store.flush().unwrap();
            }

            // Should have latest version
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), 1);
                let (vec, meta) = store.get("same_id").unwrap();
                prop_assert_eq!(vec.data, final_data);
                prop_assert_eq!(meta["version"].as_u64().unwrap() as usize, last_update);
            }
        }

        /// Batch insert persistence
        #[test]
        fn batch_persistence(
            num_batches in 2usize..5,
            batch_size in 5usize..15
        ) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("batch.omen");
            let dim = 8;

            let total = num_batches * batch_size;

            // Insert in batches
            {
                let mut store = VectorStore::open_with_dimensions(&path, dim).unwrap();

                for batch_idx in 0..num_batches {
                    let items: Vec<_> = (0..batch_size).map(|i| {
                        let idx = batch_idx * batch_size + i;
                        let data: Vec<f32> = (0..dim).map(|j| (idx * 10 + j) as f32).collect();
                        (format!("b{}_i{}", batch_idx, i), Vector::new(data), serde_json::json!({"batch": batch_idx, "i": i}))
                    }).collect();
                    store.set_batch(items).unwrap();
                }
                store.flush().unwrap();
            }

            // Verify all
            {
                let store = VectorStore::open(&path).unwrap();
                prop_assert_eq!(store.len(), total);

                for batch_idx in 0..num_batches {
                    for i in 0..batch_size {
                        let id = format!("b{}_i{}", batch_idx, i);
                        prop_assert!(store.contains(&id), "Missing ID");
                    }
                }
            }
        }
    }
}
