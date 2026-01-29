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

    // Verify segments exist
    assert!(store.segments.is_some());

    // Clear the index
    store.segments = None;
    assert!(store.segments.is_none());

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
    let mut store = VectorStore::new_with_params(128, 16, 100, 100, Metric::L2);

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

// ============================================================================
// MUVERA (Multi-Vector) Tests
// ============================================================================

mod muvera_tests {
    use super::*;
    use crate::vector::muvera::MuveraConfig;

    /// Config for tests using small token dimensions (< 16).
    /// Disables d_proj to avoid d_proj > token_dim panic.
    fn small_dim_config() -> MuveraConfig {
        MuveraConfig {
            d_proj: None,
            ..Default::default()
        }
    }

    fn random_tokens(num_tokens: usize, dim: usize, seed: usize) -> Vec<Vec<f32>> {
        (0..num_tokens)
            .map(|i| {
                (0..dim)
                    .map(|j| ((seed + i * dim + j) as f32 * 0.01) - 0.5)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_multi_vector_creates_encoder() {
        let config = MuveraConfig::default(); // d_proj=16
        let store = VectorStore::multi_vector_with(128, config);

        assert!(store.is_multi_vector());
        assert_eq!(store.token_dimension(), Some(128));
        assert_eq!(store.encoded_dimension(), Some(2048)); // 8 * 16 * 16 (d_proj)
    }

    #[test]
    fn test_regular_store_not_muvera() {
        let store = VectorStore::new(128);

        assert!(!store.is_multi_vector());
        assert_eq!(store.token_dimension(), None);
        assert_eq!(store.encoded_dimension(), None);
    }

    #[test]
    fn test_set_multi_basic() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        let tokens = random_tokens(10, 4, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        store
            .set_multi("doc1", &token_refs, serde_json::json!({"title": "test"}))
            .unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains("doc1"));
    }

    #[test]
    fn test_set_multi_multiple_docs() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        for i in 0..10 {
            let tokens = random_tokens(5 + i, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(
                    &format!("doc{}", i),
                    &token_refs,
                    serde_json::json!({"i": i}),
                )
                .unwrap();
        }

        assert_eq!(store.len(), 10);
    }

    #[test]
    fn test_set_multi_error_on_regular_store() {
        let mut store = VectorStore::new(128);

        let tokens = random_tokens(10, 128, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let result = store.set_multi("doc1", &token_refs, serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not configured"));
    }

    #[test]
    fn test_set_multi_error_on_empty_tokens() {
        let mut store = VectorStore::multi_vector_with(128, MuveraConfig::default());

        let tokens: Vec<&[f32]> = vec![];
        let result = store.set_multi("doc1", &tokens, serde_json::json!({}));

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_set_multi_error_on_wrong_dimension() {
        let mut store = VectorStore::multi_vector_with(128, MuveraConfig::default());

        let tokens = random_tokens(10, 64, 0); // Wrong dimension
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let result = store.set_multi("doc1", &token_refs, serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dimension"));
    }

    #[test]
    fn test_search_multi_approx_basic() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        // Insert 100 documents
        for i in 0..100 {
            let tokens = random_tokens(10, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(
                    &format!("doc{}", i),
                    &token_refs,
                    serde_json::json!({"i": i}),
                )
                .unwrap();
        }

        // Search - use exact same tokens as a document to ensure it's found
        let query_tokens = random_tokens(10, 4, 0); // Same as doc0
        let query_refs: Vec<&[f32]> = query_tokens.iter().map(|t| t.as_slice()).collect();

        let results = store.search_multi_approx(&query_refs, 10).unwrap();

        assert_eq!(results.len(), 10);
        // All results should have valid IDs
        for result in &results {
            assert!(result.id.starts_with("doc"));
        }
    }

    #[test]
    fn test_search_multi_approx_returns_correct_metadata() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        let tokens = random_tokens(10, 4, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
        store
            .set_multi(
                "doc1",
                &token_refs,
                serde_json::json!({"title": "Test Document"}),
            )
            .unwrap();

        let results = store.search_multi_approx(&token_refs, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[0].metadata["title"], "Test Document");
    }

    #[test]
    fn test_search_multi_approx_error_on_regular_store() {
        let store = VectorStore::new(128);

        let tokens = random_tokens(10, 128, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let result = store.search_multi_approx(&token_refs, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_multi_approx_error_on_empty_query() {
        let mut store = VectorStore::multi_vector_with(128, MuveraConfig::default());

        // Insert a document first
        let tokens = random_tokens(10, 128, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
        store
            .set_multi("doc1", &token_refs, serde_json::json!({}))
            .unwrap();

        // Empty query
        let empty: Vec<&[f32]> = vec![];
        let result = store.search_multi_approx(&empty, 10);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_set_multi_batch_basic() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // Create batch of documents
        let batch: Vec<(&str, Vec<Vec<f32>>, serde_json::Value)> = (0..100)
            .map(|i| {
                let tokens = random_tokens(10 + (i % 10), 4, i);
                (
                    format!("doc{}", i).leak() as &str,
                    tokens,
                    serde_json::json!({"i": i}),
                )
            })
            .collect();

        store.set_multi_batch(batch).unwrap();

        assert_eq!(store.len(), 100);
        for i in 0..100 {
            assert!(store.contains(&format!("doc{}", i)));
        }
    }

    #[test]
    fn test_set_multi_batch_searchable() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // Batch insert
        let batch: Vec<(&str, Vec<Vec<f32>>, serde_json::Value)> = (0..50)
            .map(|i| {
                let tokens = random_tokens(10, 4, i);
                (
                    format!("doc{}", i).leak() as &str,
                    tokens,
                    serde_json::json!({"i": i}),
                )
            })
            .collect();

        store.set_multi_batch(batch).unwrap();

        // Search
        let query = random_tokens(5, 4, 25);
        let query_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();
        let results = store.search_multi_approx(&query_refs, 10).unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_set_multi_batch_empty() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        let batch: Vec<(&str, Vec<Vec<f32>>, serde_json::Value)> = vec![];
        store.set_multi_batch(batch).unwrap();

        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_set_multi_batch_error_on_invalid_doc() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // One valid doc, one with wrong dimension
        let batch: Vec<(&str, Vec<Vec<f32>>, serde_json::Value)> = vec![
            ("doc1", random_tokens(10, 4, 0), serde_json::json!({})),
            ("doc2", random_tokens(10, 8, 1), serde_json::json!({})), // Wrong dim
        ];

        let result = store.set_multi_batch(batch);
        assert!(result.is_err());
        // Store should not have any documents (validation failed before insertion)
        assert_eq!(store.len(), 0);
    }

    // ========================================================================
    // MUV-16: search_multi tests (with MaxSim reranking)
    // ========================================================================

    #[test]
    fn test_search_multi_basic() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        // Insert 100 documents
        for i in 0..100 {
            let tokens = random_tokens(10, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(
                    &format!("doc{}", i),
                    &token_refs,
                    serde_json::json!({"i": i}),
                )
                .unwrap();
        }

        // Search with reranking
        let query_tokens = random_tokens(10, 4, 0); // Same as doc0
        let query_refs: Vec<&[f32]> = query_tokens.iter().map(|t| t.as_slice()).collect();

        let results = store.search_multi(&query_refs, 10).unwrap();

        assert_eq!(results.len(), 10);
        // All results should have valid IDs
        for result in &results {
            assert!(result.id.starts_with("doc"));
        }
    }

    #[test]
    fn test_search_multi_improves_ordering() {
        // This test verifies that reranking improves result quality
        // by checking that similar documents score higher after reranking
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        // Insert 20 documents with varying similarity to query
        for i in 0..20 {
            let tokens = random_tokens(10, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(&format!("doc{}", i), &token_refs, serde_json::json!({}))
                .unwrap();
        }

        let query = random_tokens(5, 4, 0); // Similar to doc0
        let query_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();

        // Get results with reranking
        let results = store.search_multi(&query_refs, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Verify results are sorted by MaxSim score (descending)
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance >= results[i].distance,
                "Results should be sorted by MaxSim score"
            );
        }

        // Verify scores are reasonable (MaxSim with 5 query tokens should be positive)
        for result in &results {
            assert!(
                result.distance >= 0.0,
                "MaxSim scores should be non-negative"
            );
        }
    }

    #[test]
    fn test_search_multi_returns_maxsim_scores() {
        let config = small_dim_config();
        let mut store = VectorStore::multi_vector_with(4, config);

        // Insert document with known tokens
        let doc_tokens: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(|t| t.as_slice()).collect();
        store
            .set_multi("doc1", &doc_refs, serde_json::json!({}))
            .unwrap();

        // Query matching first token exactly
        let query_tokens: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let query_refs: Vec<&[f32]> = query_tokens.iter().map(|t| t.as_slice()).collect();

        let results = store.search_multi(&query_refs, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        // MaxSim score should be 1.0 (perfect match with first doc token)
        assert!(
            (results[0].distance - 1.0).abs() < 0.01,
            "MaxSim score should be ~1.0, got {}",
            results[0].distance
        );
    }

    #[test]
    fn test_search_multi_error_on_regular_store() {
        let store = VectorStore::new(128);

        let tokens = random_tokens(10, 128, 0);
        let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        let result = store.search_multi(&token_refs, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_multi_empty_store() {
        let store = VectorStore::multi_vector_with(4, small_dim_config());

        let query = random_tokens(5, 4, 0);
        let query_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();

        let results = store.search_multi(&query_refs, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_multi_custom_factor() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // Insert 20 documents
        for i in 0..20 {
            let tokens = random_tokens(10, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(&format!("doc{}", i), &token_refs, serde_json::json!({}))
                .unwrap();
        }

        let query = random_tokens(5, 4, 0);
        let query_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();

        // With k=5, rerank_factor=2 -> fetches 10 candidates
        let results = store.search_multi_rerank(&query_refs, 5, 2).unwrap();
        assert_eq!(results.len(), 5);

        // With k=5, rerank_factor=8 -> fetches 40 candidates (but only 20 exist)
        let results = store.search_multi_rerank(&query_refs, 5, 8).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_multi_scores_are_ordered() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // Insert 50 documents
        for i in 0..50 {
            let tokens = random_tokens(10, 4, i);
            let token_refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
            store
                .set_multi(&format!("doc{}", i), &token_refs, serde_json::json!({}))
                .unwrap();
        }

        let query = random_tokens(5, 4, 0);
        let query_refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();

        let results = store.search_multi(&query_refs, 10).unwrap();

        // Results should be ordered by descending MaxSim score
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance >= results[i].distance,
                "Results should be ordered by MaxSim score (descending)"
            );
        }
    }
}

// ============================================================================
// Unified API tests
// ============================================================================

mod unified_api_tests {
    use super::*;
    use crate::vector::muvera::MuveraConfig;

    fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect()
    }

    fn random_tokens(num_tokens: usize, dim: usize, seed: usize) -> Vec<Vec<f32>> {
        (0..num_tokens)
            .map(|i| {
                (0..dim)
                    .map(|j| ((seed + i * dim + j) as f32 * 0.01) - 0.5)
                    .collect()
            })
            .collect()
    }

    fn small_dim_config() -> MuveraConfig {
        MuveraConfig {
            d_proj: None,
            ..Default::default()
        }
    }

    // === Regular store tests ===

    #[test]
    fn test_store_single_vector() {
        let mut store = VectorStore::new(4);

        store
            .store(
                "doc1",
                vec![1.0, 2.0, 3.0, 4.0],
                serde_json::json!({"title": "test"}),
            )
            .unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains("doc1"));
    }

    #[test]
    fn test_store_single_vector_slice() {
        let mut store = VectorStore::new(4);
        let vec = [1.0, 2.0, 3.0, 4.0];

        store
            .store("doc1", vec.as_slice(), serde_json::json!({}))
            .unwrap();

        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_store_multi_in_regular_store_fails() {
        let mut store = VectorStore::new(4);
        let tokens = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        let result = store.store("doc1", tokens, serde_json::json!({}));

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot store token embeddings in regular store"));
    }

    #[test]
    fn test_query_single_vector() {
        let mut store = VectorStore::new(4);
        store
            .store("doc1", vec![1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();
        store
            .store("doc2", vec![0.0, 1.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();

        let results = store.query(&vec![1.0, 0.1, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1"); // Closer to query
    }

    #[test]
    fn test_query_multi_in_regular_store_fails() {
        let mut store = VectorStore::new(4);
        store
            .store("doc1", vec![1.0, 0.0, 0.0, 0.0], serde_json::json!({}))
            .unwrap();

        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let result = store.query(&query, 1);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot query regular store with token embeddings"));
    }

    #[test]
    fn test_get_vector() {
        let mut store = VectorStore::new(4);
        store
            .store(
                "doc1",
                vec![1.0, 2.0, 3.0, 4.0],
                serde_json::json!({"key": "value"}),
            )
            .unwrap();

        let (vec, meta) = store.get_vector("doc1").unwrap();

        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(meta["key"], "value");
    }

    #[test]
    fn test_get_data_single() {
        let mut store = VectorStore::new(4);
        store
            .store("doc1", vec![1.0, 2.0, 3.0, 4.0], serde_json::json!({}))
            .unwrap();

        let (data, _) = store.get_data("doc1").unwrap();

        assert!(data.is_single());
        assert_eq!(data.as_single().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_store_batch_single() {
        let mut store = VectorStore::new(4);

        store
            .store_batch(vec![
                ("doc1", vec![1.0, 2.0, 3.0, 4.0], serde_json::json!({})),
                ("doc2", vec![5.0, 6.0, 7.0, 8.0], serde_json::json!({})),
            ])
            .unwrap();

        assert_eq!(store.len(), 2);
        assert!(store.contains("doc1"));
        assert!(store.contains("doc2"));
    }

    // === Multi-vector store tests ===

    #[test]
    fn test_store_multi_vector() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());
        let tokens = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        store
            .store("doc1", tokens, serde_json::json!({"title": "test"}))
            .unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.contains("doc1"));
    }

    #[test]
    fn test_store_single_in_multi_store_fails() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        let result = store.store("doc1", vec![1.0, 2.0, 3.0, 4.0], serde_json::json!({}));

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot store single vector in multi-vector store"));
    }

    #[test]
    fn test_query_multi_vector() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        // Insert documents
        let doc1_tokens = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let doc2_tokens = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];

        store
            .store("doc1", doc1_tokens, serde_json::json!({}))
            .unwrap();
        store
            .store("doc2", doc2_tokens, serde_json::json!({}))
            .unwrap();

        // Query
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let results = store.query(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        // doc1 should be closer because query token matches one of its tokens
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_query_single_in_multi_store_fails() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());
        store
            .store(
                "doc1",
                vec![vec![1.0, 0.0, 0.0, 0.0]],
                serde_json::json!({}),
            )
            .unwrap();

        let result = store.query(&vec![1.0, 0.0, 0.0, 0.0], 1);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot query multi-vector store with single vector"));
    }

    #[test]
    fn test_get_tokens() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());
        let tokens = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        store
            .store("doc1", tokens.clone(), serde_json::json!({"key": "value"}))
            .unwrap();

        let (retrieved, meta) = store.get_tokens("doc1").unwrap();

        assert_eq!(retrieved, tokens);
        assert_eq!(meta["key"], "value");
    }

    #[test]
    fn test_get_data_multi() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());
        let tokens = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        store
            .store("doc1", tokens.clone(), serde_json::json!({}))
            .unwrap();

        let (data, _) = store.get_data("doc1").unwrap();

        assert!(data.is_multi());
        assert_eq!(data.as_multi().unwrap(), &tokens);
    }

    #[test]
    fn test_get_vector_on_multi_store_returns_none() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());
        store
            .store(
                "doc1",
                vec![vec![1.0, 2.0, 3.0, 4.0]],
                serde_json::json!({}),
            )
            .unwrap();

        assert!(store.get_vector("doc1").is_none());
    }

    #[test]
    fn test_get_tokens_on_regular_store_returns_none() {
        let mut store = VectorStore::new(4);
        store
            .store("doc1", vec![1.0, 2.0, 3.0, 4.0], serde_json::json!({}))
            .unwrap();

        assert!(store.get_tokens("doc1").is_none());
    }

    #[test]
    fn test_store_batch_multi() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        store
            .store_batch(vec![
                (
                    "doc1",
                    vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
                    serde_json::json!({}),
                ),
                (
                    "doc2",
                    vec![vec![9.0, 10.0, 11.0, 12.0]],
                    serde_json::json!({}),
                ),
            ])
            .unwrap();

        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_query_with_options_no_rerank() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        let doc1_tokens = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let doc2_tokens = vec![vec![0.0, 1.0, 0.0, 0.0]];

        store
            .store("doc1", doc1_tokens, serde_json::json!({}))
            .unwrap();
        store
            .store("doc2", doc2_tokens, serde_json::json!({}))
            .unwrap();

        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let options = SearchOptions::default().no_rerank();
        let results = store.query_with_options(&query, 2, &options).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_with_options_custom_rerank() {
        let mut store = VectorStore::multi_vector_with(4, small_dim_config());

        let doc1_tokens = vec![vec![1.0, 0.0, 0.0, 0.0]];
        store
            .store("doc1", doc1_tokens, serde_json::json!({}))
            .unwrap();

        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let options = SearchOptions::default().rerank(Rerank::Factor(10));
        let results = store.query_with_options(&query, 1, &options).unwrap();

        assert_eq!(results.len(), 1);
    }
}

// ============================================================================
// Multi-vector persistence tests (MUV-13)
// ============================================================================

mod multivec_persistence_tests {
    use super::*;
    use crate::vector::muvera::MuveraConfig;
    use tempfile::tempdir;

    fn small_dim_config() -> MuveraConfig {
        MuveraConfig {
            d_proj: None,
            ..Default::default()
        }
    }

    fn random_tokens(num_tokens: usize, dim: usize, seed: usize) -> Vec<Vec<f32>> {
        (0..num_tokens)
            .map(|i| {
                (0..dim)
                    .map(|j| ((seed + i * dim + j) as f32 * 0.01) - 0.5)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_multivec_persistence_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_multivec.omen");

        let token_dim = 8;

        // Create store, add documents, flush
        {
            let mut store = VectorStore::multi_vector_with(token_dim, small_dim_config());
            store = store.persist(&path).unwrap();

            let doc1_tokens = random_tokens(5, token_dim, 100);
            let doc2_tokens = random_tokens(3, token_dim, 200);

            store
                .store(
                    "doc1",
                    doc1_tokens.clone(),
                    serde_json::json!({"title": "first"}),
                )
                .unwrap();
            store
                .store(
                    "doc2",
                    doc2_tokens.clone(),
                    serde_json::json!({"title": "second"}),
                )
                .unwrap();

            store.flush().unwrap();

            // Verify before close
            assert!(store.is_multi_vector());
            assert_eq!(store.len(), 2);
        }

        // Reopen and verify
        {
            let store = VectorStore::open(&path).unwrap();

            // Should detect multi-vector from persisted config
            assert!(store.is_multi_vector());
            assert_eq!(store.len(), 2);
            assert_eq!(store.token_dimension(), Some(token_dim));

            // Verify documents exist
            assert!(store.contains("doc1"));
            assert!(store.contains("doc2"));
        }
    }

    #[test]
    fn test_multivec_persistence_empty_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_empty_multivec.omen");

        // Create empty multi-vector store and flush
        {
            let mut store = VectorStore::multi_vector(128);
            store = store.persist(&path).unwrap();
            store.flush().unwrap();
        }

        // Reopen - should detect multi-vector config
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(store.is_multi_vector());
            assert_eq!(store.len(), 0);
            assert_eq!(store.token_dimension(), Some(128));
        }
    }

    #[test]
    fn test_multivec_persistence_large_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_large_multivec.omen");

        let token_dim = 32;
        let num_docs = 100;

        // Create store with 100 documents
        {
            let mut store = VectorStore::multi_vector(token_dim);
            store = store.persist(&path).unwrap();

            for i in 0..num_docs {
                let num_tokens = (i % 5) + 1; // 1-5 tokens
                let tokens = random_tokens(num_tokens, token_dim, i * 1000);
                store
                    .store(&format!("doc{i}"), tokens, serde_json::json!({"idx": i}))
                    .unwrap();
            }

            store.flush().unwrap();
        }

        // Reopen and verify
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(store.is_multi_vector());
            assert_eq!(store.len(), num_docs);

            // Spot check a few documents
            for i in [0, 50, 99] {
                assert!(store.contains(&format!("doc{i}")));
            }
        }
    }

    #[test]
    fn test_multivec_rerank_after_reload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_rerank_reload.omen");

        let token_dim = 4;

        // Create store with documents that have different relevance patterns
        {
            let mut store = VectorStore::multi_vector_with(token_dim, small_dim_config());
            store = store.persist(&path).unwrap();

            // doc1: tokens aligned with query
            store
                .store(
                    "doc1",
                    vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
                    serde_json::json!({}),
                )
                .unwrap();

            // doc2: tokens less aligned
            store
                .store(
                    "doc2",
                    vec![vec![0.3, 0.3, 0.3, 0.0], vec![0.0, 0.0, 0.5, 0.5]],
                    serde_json::json!({}),
                )
                .unwrap();

            store.flush().unwrap();
        }

        // Reopen and search with reranking
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(store.is_multi_vector());

            // Query tokens aligned with doc1
            let query = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
            let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();

            let results = store.search_multi_rerank(&query_refs, 2, 10).unwrap();

            assert_eq!(results.len(), 2);
            // doc1 should rank higher (better MaxSim alignment)
            assert_eq!(results[0].id, "doc1");
        }
    }

    #[test]
    fn test_multivec_config_persisted_correctly() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_config.omen");

        use crate::vector::muvera::MultiVectorConfig;

        let token_dim = 64;
        let custom_config = MultiVectorConfig {
            repetitions: 10,
            partition_bits: 4,
            d_proj: Some(16),
            seed: 12345,
            pool_factor: None,
        };

        // Create with custom config
        {
            let mut store = VectorStore::multi_vector_with(token_dim, custom_config.clone());
            store = store.persist(&path).unwrap();

            store
                .store("doc1", vec![vec![0.1f32; token_dim]], serde_json::json!({}))
                .unwrap();

            store.flush().unwrap();
        }

        // Reopen and verify config
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(store.is_multi_vector());
            assert_eq!(store.token_dimension(), Some(token_dim));

            // Encoded dimension should match the custom config
            // encoded_dim = repetitions * 2^partition_bits * d_proj
            // = 10 * 16 * 16 = 2560
            let expected_encoded_dim = 10 * 16 * 16; // d_proj=16
            assert_eq!(store.encoded_dimension(), Some(expected_encoded_dim));
        }
    }

    #[test]
    fn test_regular_store_no_multivec_after_reload() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_regular.omen");

        // Create regular (non-multi-vector) store
        {
            let mut store = VectorStore::new(4);
            store = store.persist(&path).unwrap();

            store
                .store("doc1", vec![1.0, 2.0, 3.0, 4.0], serde_json::json!({}))
                .unwrap();

            store.flush().unwrap();
        }

        // Reopen - should NOT be multi-vector
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(!store.is_multi_vector());
            assert_eq!(store.len(), 1);
        }
    }

    #[test]
    fn test_pooling_reduces_stored_tokens() {
        use crate::vector::muvera::MultiVectorConfig;

        let token_dim = 64;

        // Config with pool_factor=2: 50% token reduction
        let config_with_pooling = MultiVectorConfig {
            pool_factor: Some(2),
            ..Default::default()
        };

        // Config without pooling
        let config_no_pooling = MultiVectorConfig {
            pool_factor: None,
            ..Default::default()
        };

        // Create 100 tokens
        let tokens: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..token_dim)
                    .map(|j| ((i * 64 + j) as f32).sin())
                    .collect()
            })
            .collect();

        // Store with pooling
        let mut store_pooled = VectorStore::multi_vector_with(token_dim, config_with_pooling);
        store_pooled
            .store("doc1", tokens.clone(), serde_json::json!({}))
            .unwrap();

        // Store without pooling
        let mut store_no_pool = VectorStore::multi_vector_with(token_dim, config_no_pooling);
        store_no_pool
            .store("doc1", tokens.clone(), serde_json::json!({}))
            .unwrap();

        // Get stored tokens
        let (pooled_tokens, _) = store_pooled.get_tokens("doc1").unwrap();
        let (no_pool_tokens, _) = store_no_pool.get_tokens("doc1").unwrap();

        // Pooled should have ~50 tokens (100 / 2)
        assert_eq!(pooled_tokens.len(), 50, "pool_factor=2 should halve tokens");
        assert_eq!(
            no_pool_tokens.len(),
            100,
            "no pooling should keep all tokens"
        );

        // Verify dimensions are preserved
        assert_eq!(pooled_tokens[0].len(), token_dim);
    }

    #[test]
    fn test_pooling_persists_and_reloads() {
        use crate::vector::muvera::MultiVectorConfig;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_pooling.omen");
        let token_dim = 32;

        let config = MultiVectorConfig {
            pool_factor: Some(2),
            ..Default::default()
        };

        // Create and persist with pooling
        {
            let mut store = VectorStore::multi_vector_with(token_dim, config);
            store = store.persist(&path).unwrap();

            let tokens: Vec<Vec<f32>> = (0..20)
                .map(|i| {
                    (0..token_dim)
                        .map(|j| ((i * 32 + j) as f32).cos())
                        .collect()
                })
                .collect();

            store.store("doc1", tokens, serde_json::json!({})).unwrap();
            store.flush().unwrap();

            // Verify pooling worked
            let (stored_tokens, _) = store.get_tokens("doc1").unwrap();
            assert_eq!(stored_tokens.len(), 10); // 20 / 2 = 10
        }

        // Reopen and verify
        {
            let store = VectorStore::open(&path).unwrap();
            assert!(store.is_multi_vector());

            let (stored_tokens, _) = store.get_tokens("doc1").unwrap();
            assert_eq!(stored_tokens.len(), 10);
            assert_eq!(stored_tokens[0].len(), token_dim);
        }
    }
}
