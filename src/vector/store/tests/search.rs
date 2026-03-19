use super::super::*;
use super::random_vector;

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
fn test_search_with_filter() {
    let mut store = VectorStore::new(128);

    // Insert vectors with metadata
    store
        .set(
            "doc1",
            random_vector(128, 0),
            serde_json::json!({"author": "Alice", "year": 2024}),
        )
        .unwrap();

    store
        .set(
            "doc2",
            random_vector(128, 1),
            serde_json::json!({"author": "Bob", "year": 2023}),
        )
        .unwrap();

    store
        .set(
            "doc3",
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
fn test_vector_store_cosine_zero_query_rejected_in_brute_force_path() {
    let mut store = VectorStore::new_with_params(3, 16, 100, 100, Metric::Cosine);
    store
        .records
        .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
        .unwrap();

    let err = store
        .search(&Vector::new(vec![0.0, 0.0, 0.0]), 1, None)
        .unwrap_err();

    assert!(err.to_string().contains("zero vector"));
}

#[test]
fn test_vector_store_search_with_options_rejects_non_finite_query() {
    let mut store = VectorStore::new(3);
    store
        .records
        .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
        .unwrap();

    let err = store
        .search_with_options(&Vector::new(vec![1.0, f32::NAN, 0.0]), 1, None, None, None)
        .unwrap_err();

    assert!(err.to_string().contains("NaN or Infinity"));
}

#[test]
fn test_vector_store_search_batch_reports_invalid_query() {
    let mut store = VectorStore::new_with_params(3, 16, 100, 100, Metric::Cosine);
    store
        .records
        .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
        .unwrap();

    let queries = vec![Vector::new(vec![0.0, 0.0, 0.0])];
    let results = store.search_batch(&queries, 1, None);

    assert_eq!(results.len(), 1);
    assert!(
        results[0]
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("zero vector")
    );
}

#[test]
fn test_vector_store_search_rejects_k_zero_without_segments() {
    let mut store = VectorStore::new(3);
    store
        .records
        .set("a".to_string(), vec![1.0, 0.0, 0.0], None)
        .unwrap();

    let err = store
        .search(&Vector::new(vec![1.0, 0.0, 0.0]), 0, None)
        .unwrap_err();

    assert!(err.to_string().contains("k=0"));
}
