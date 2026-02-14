use super::super::*;
use crate::vector::sparse::SparseVector;

#[test]
fn test_enable_sparse() {
    let mut store = VectorStore::new(4);
    assert!(!store.has_sparse());

    store.enable_sparse();
    assert!(store.has_sparse());

    // Enabling again is a no-op
    store.enable_sparse();
    assert!(store.has_sparse());
}

#[test]
fn test_set_sparse_auto_enables() {
    let mut store = VectorStore::new(0);
    assert!(!store.has_sparse());

    let sparse = SparseVector::from_pairs(vec![(10, 0.5), (42, 1.2)]);
    store
        .set_sparse("doc1", sparse, serde_json::json!({}))
        .unwrap();

    // set_sparse auto-enables the sparse index
    assert!(store.has_sparse());
}

#[test]
fn test_set_sparse_and_search() {
    let mut store = VectorStore::new(0);
    store.enable_sparse();

    // Insert documents with different sparse vectors
    let sv1 = SparseVector::from_pairs(vec![(10, 0.5), (42, 1.2), (100, 0.8)]);
    store
        .set_sparse("doc1", sv1, serde_json::json!({"title": "first"}))
        .unwrap();

    let sv2 = SparseVector::from_pairs(vec![(10, 0.3), (50, 2.0), (100, 0.1)]);
    store
        .set_sparse("doc2", sv2, serde_json::json!({"title": "second"}))
        .unwrap();

    let sv3 = SparseVector::from_pairs(vec![(42, 0.9), (50, 0.5)]);
    store
        .set_sparse("doc3", sv3, serde_json::json!({"title": "third"}))
        .unwrap();

    // Search for documents with dims 10 and 42
    let query = SparseVector::from_pairs(vec![(10, 1.0), (42, 1.0)]);
    let results = store.sparse_search(&query, 3, None).unwrap();

    assert_eq!(results.len(), 3);
    // doc1 has both dims 10 and 42 with high weights
    assert_eq!(results[0].id, "doc1");
}

#[test]
fn test_sparse_search_with_filter() {
    let mut store = VectorStore::new(0);
    store.enable_sparse();

    let sv1 = SparseVector::from_pairs(vec![(10, 1.0), (42, 1.0)]);
    store
        .set_sparse("doc1", sv1, serde_json::json!({"category": "A"}))
        .unwrap();

    let sv2 = SparseVector::from_pairs(vec![(10, 0.9), (42, 0.9)]);
    store
        .set_sparse("doc2", sv2, serde_json::json!({"category": "B"}))
        .unwrap();

    let sv3 = SparseVector::from_pairs(vec![(10, 0.1)]);
    store
        .set_sparse("doc3", sv3, serde_json::json!({"category": "A"}))
        .unwrap();

    // Search with filter for category B
    let query = SparseVector::from_pairs(vec![(10, 1.0), (42, 1.0)]);
    let filter = MetadataFilter::Eq("category".to_string(), serde_json::json!("B"));
    let results = store.sparse_search(&query, 10, Some(&filter)).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc2");
}

#[test]
fn test_set_hybrid_sparse() {
    let mut store = VectorStore::new(3);
    store.enable_sparse();

    let dense = Vector::new(vec![1.0, 0.0, 0.0]);
    let sparse = SparseVector::from_pairs(vec![(10, 0.5), (42, 1.2)]);
    store
        .set_hybrid_sparse("doc1", dense, sparse, serde_json::json!({"cat": "A"}))
        .unwrap();

    // Verify dense search works
    let dense_query = Vector::new(vec![1.0, 0.0, 0.0]);
    let dense_results = store
        .search_with_options(&dense_query, 1, None, None, None)
        .unwrap();
    assert_eq!(dense_results.len(), 1);
    assert_eq!(dense_results[0].id, "doc1");

    // Verify sparse search works
    let sparse_query = SparseVector::from_pairs(vec![(42, 1.0)]);
    let sparse_results = store.sparse_search(&sparse_query, 1, None).unwrap();
    assert_eq!(sparse_results.len(), 1);
    assert_eq!(sparse_results[0].id, "doc1");
}

#[test]
fn test_hybrid_sparse_search_rrf() {
    let mut store = VectorStore::new(3);
    store.enable_sparse();

    // doc1: close in dense, weak in sparse
    store
        .set_hybrid_sparse(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0]),
            SparseVector::from_pairs(vec![(10, 0.1)]),
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: far in dense, strong in sparse
    store
        .set_hybrid_sparse(
            "doc2",
            Vector::new(vec![0.0, 1.0, 0.0]),
            SparseVector::from_pairs(vec![(10, 5.0), (42, 3.0)]),
            serde_json::json!({}),
        )
        .unwrap();

    // doc3: moderate in both
    store
        .set_hybrid_sparse(
            "doc3",
            Vector::new(vec![0.8, 0.2, 0.0]),
            SparseVector::from_pairs(vec![(10, 1.0), (42, 1.0)]),
            serde_json::json!({}),
        )
        .unwrap();

    // Hybrid search: alpha=0.5 (equal weight)
    let dense_q = Vector::new(vec![1.0, 0.0, 0.0]);
    let sparse_q = SparseVector::from_pairs(vec![(10, 1.0), (42, 1.0)]);
    let results = store
        .hybrid_sparse_search(&dense_q, &sparse_q, 3, 0.5, None)
        .unwrap();

    assert_eq!(results.len(), 3);
    // All three should be present
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"doc1"));
    assert!(ids.contains(&"doc2"));
    assert!(ids.contains(&"doc3"));

    // Pure dense: doc1 should be first
    let dense_only = store
        .hybrid_sparse_search(&dense_q, &sparse_q, 3, 1.0, None)
        .unwrap();
    assert_eq!(dense_only[0].id, "doc1");

    // Pure sparse: doc2 should be first (highest sparse scores)
    let sparse_only = store
        .hybrid_sparse_search(&dense_q, &sparse_q, 3, 0.0, None)
        .unwrap();
    assert_eq!(sparse_only[0].id, "doc2");
}

#[test]
fn test_sparse_upsert() {
    let mut store = VectorStore::new(0);
    store.enable_sparse();

    let sv1 = SparseVector::from_pairs(vec![(10, 1.0)]);
    store
        .set_sparse("doc1", sv1, serde_json::json!({"v": 1}))
        .unwrap();

    // Upsert with new sparse vector
    let sv2 = SparseVector::from_pairs(vec![(42, 2.0)]);
    store
        .set_sparse("doc1", sv2, serde_json::json!({"v": 2}))
        .unwrap();

    // Old dim should not match
    let q1 = SparseVector::from_pairs(vec![(10, 1.0)]);
    let r1 = store.sparse_search(&q1, 1, None).unwrap();
    // doc1 should still appear but with 0 score on dim 10 (the sparse index replaces)
    // Actually the SparseIndex.insert does remove+insert, so dim 10 is gone
    // The search might return 0 results or doc1 with score 0
    // Since search only returns non-zero scores, it should be empty
    assert!(r1.is_empty() || r1[0].distance == 0.0);

    // New dim should match
    let q2 = SparseVector::from_pairs(vec![(42, 1.0)]);
    let r2 = store.sparse_search(&q2, 1, None).unwrap();
    assert_eq!(r2.len(), 1);
    assert_eq!(r2[0].id, "doc1");

    // Metadata should be updated
    assert_eq!(r2[0].metadata["v"], 2);
}

#[test]
fn test_sparse_preserves_dense() {
    let mut store = VectorStore::new(3);
    store.enable_sparse();

    // Insert dense first
    let dense = Vector::new(vec![1.0, 0.0, 0.0]);
    store
        .set("doc1", dense, serde_json::json!({"type": "dense"}))
        .unwrap();

    // Now add sparse to same doc
    let sparse = SparseVector::from_pairs(vec![(42, 1.0)]);
    store
        .set_sparse("doc1", sparse, serde_json::json!({"type": "hybrid"}))
        .unwrap();

    // Dense search should still find it
    let dq = Vector::new(vec![1.0, 0.0, 0.0]);
    let dr = store.search_with_options(&dq, 1, None, None, None).unwrap();
    assert_eq!(dr.len(), 1);
    assert_eq!(dr[0].id, "doc1");

    // Sparse search should also find it
    let sq = SparseVector::from_pairs(vec![(42, 1.0)]);
    let sr = store.sparse_search(&sq, 1, None).unwrap();
    assert_eq!(sr.len(), 1);
    assert_eq!(sr[0].id, "doc1");
}

#[test]
fn test_sparse_persistence_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.omen");

    // Create store, add sparse data, flush
    {
        let mut store = VectorStore::open(path.to_str().unwrap()).unwrap();
        store.enable_sparse();

        store
            .set_sparse(
                "doc1",
                SparseVector::from_pairs(vec![(10, 0.5), (42, 1.2)]),
                serde_json::json!({"title": "test"}),
            )
            .unwrap();
        store
            .set_sparse(
                "doc2",
                SparseVector::from_pairs(vec![(10, 0.3), (50, 2.0)]),
                serde_json::json!({"title": "test2"}),
            )
            .unwrap();

        store.flush().unwrap();
    }

    // Reopen and verify
    {
        let store = VectorStore::open(path.to_str().unwrap()).unwrap();
        assert!(store.has_sparse());

        let query = SparseVector::from_pairs(vec![(42, 1.0)]);
        let results = store.sparse_search(&query, 2, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");

        // Also check dim 50
        let q2 = SparseVector::from_pairs(vec![(50, 1.0)]);
        let r2 = store.sparse_search(&q2, 2, None).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].id, "doc2");
    }
}

#[test]
fn test_sparse_empty_search() {
    let mut store = VectorStore::new(0);
    store.enable_sparse();

    // Search with no data
    let query = SparseVector::from_pairs(vec![(10, 1.0)]);
    let results = store.sparse_search(&query, 10, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_sparse_score_convention() {
    let mut store = VectorStore::new(0);
    store.enable_sparse();

    store
        .set_sparse(
            "doc1",
            SparseVector::from_pairs(vec![(10, 2.0)]),
            serde_json::json!({}),
        )
        .unwrap();

    let query = SparseVector::from_pairs(vec![(10, 3.0)]);
    let results = store.sparse_search(&query, 1, None).unwrap();

    assert_eq!(results.len(), 1);
    // dot product = 2.0 * 3.0 = 6.0
    // distance = -dot_product = -6.0
    assert!((results[0].distance - (-6.0)).abs() < 1e-6);
}
