use super::super::*;
use super::random_vector;

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
fn test_update_vector_reindexes_hnsw() {
    let mut store = VectorStore::new(4);

    // Insert 50 vectors so HNSW is used
    for i in 0..50 {
        let v = Vector::new(vec![i as f32, 0.0, 0.0, 0.0]);
        store
            .insert_with_metadata(format!("v{i}"), v, serde_json::json!({}))
            .unwrap();
    }

    // Insert a target vector far from the query point
    let far_vec = Vector::new(vec![100.0, 100.0, 100.0, 100.0]);
    store
        .insert_with_metadata("target".to_string(), far_vec, serde_json::json!({}))
        .unwrap();

    // Search near origin -- "target" should NOT be in top 5
    let query = Vector::new(vec![0.0, 0.0, 0.0, 0.0]);
    let results = store.search(&query, 5, None).unwrap();
    assert!(
        !results.iter().any(|r| r.id == "target"),
        "target should be far from origin"
    );

    // Now update "target" vector to be near the origin
    let near_vec = Vector::new(vec![0.0, 0.0, 0.0, 0.0]);
    store.update("target", Some(near_vec), None).unwrap();

    // Search again -- "target" should now be in top 5
    let results = store.search(&query, 5, None).unwrap();
    assert!(
        results.iter().any(|r| r.id == "target"),
        "target should be near origin after update"
    );
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
