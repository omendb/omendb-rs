use super::super::*;
use super::random_vector;

#[test]
fn test_quantization_insert() {
    let mut store = VectorStore::new_with_quantization(128);

    // Insert vectors
    for i in 0..50 {
        store.insert(random_vector(128, i)).unwrap();
    }

    // Verify vectors stored and quantization is enabled
    assert_eq!(store.len(), 50);
    assert!(store
        .segments
        .read()
        .as_ref()
        .is_some_and(|s| s.is_quantized()));
}

#[test]
fn test_quantization_search_accuracy() {
    let mut store = VectorStore::new_with_quantization(128);

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
    let mut store = VectorStore::new_with_quantization(128);

    // Batch insert vectors
    let vectors: Vec<Vector> = (0..100).map(|i| random_vector(128, i)).collect();
    let ids = store.batch_insert(vectors).unwrap();

    // Verify all vectors were created and quantization is enabled
    assert_eq!(ids.len(), 100);
    assert_eq!(store.len(), 100);
    assert!(store
        .segments
        .read()
        .as_ref()
        .is_some_and(|s| s.is_quantized()));
}
