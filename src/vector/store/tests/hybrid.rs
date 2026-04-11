use super::super::*;
use crate::catalog::{CollectionSchema, SparseIndexKind, SparseSchema, TextSchema};
use crate::text::TokenizerPreset;

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
            "doc1",
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
    let results = store.search_text("machine", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_set_with_text_requires_enabled() {
    let mut store = VectorStore::new(4);

    // Should fail without enabling text search
    let result = store.set_with_text(
        "doc1",
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
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "rust programming language",
            serde_json::json!({}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2",
            Vector::new(vec![0.0, 1.0, 0.0, 0.0]),
            "rust rust systems programming",
            serde_json::json!({}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    // Search for "rust" - doc2 should rank higher (higher term frequency)
    let results = store.search_text("rust", 10).unwrap();
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
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: different vector, relevant text
    store
        .set_with_text(
            "doc2",
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning models",
            serde_json::json!({}),
        )
        .unwrap();

    // doc3: similar vector, irrelevant text
    store
        .set_with_text(
            "doc3",
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
        .search_hybrid(&query, "machine learning", 3, None, None, None)
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
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2024}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2",
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
        .search_hybrid(&query, "machine", 10, Some(&filter), None, None)
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
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test content",
            serde_json::json!({}),
        )
        .unwrap();

    // Commit text index changes
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

    // Empty text query should still return vector search results
    let results = store
        .search_hybrid(&query, "", 10, None, None, None)
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hybrid_search_alpha_weighting() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // doc1: closest vector, weak text match
    store
        .set_with_text(
            "vec_winner",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "unrelated words here",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: far vector, strong text match
    store
        .set_with_text(
            "text_winner",
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);

    // alpha=1.0: vector only - vec_winner should win
    let results = store
        .search_hybrid(&query, "machine learning", 2, None, Some(1.0), None)
        .unwrap();
    assert_eq!(results[0].0, "vec_winner");

    // alpha=0.0: text only - text_winner should win
    let results = store
        .search_hybrid(&query, "machine learning", 2, None, Some(0.0), None)
        .unwrap();
    assert_eq!(results[0].0, "text_winner");
}

#[test]
fn test_hybrid_search_requires_dense_schema() {
    let store = VectorStore::create_in_memory(CollectionSchema {
        name: "sparse-text".to_string(),
        metric: Metric::L2,
        dense: None,
        sparse: Some(SparseSchema {
            index_kind: SparseIndexKind::InvertedExact,
            max_nonzero: None,
        }),
        multi: None,
        text: Some(TextSchema {
            tokenizer: TokenizerPreset::Default,
            writer_buffer_mb: 16,
        }),
        graph: None,
    })
    .unwrap();

    let result = store.search_hybrid(&Vector::new(vec![1.0]), "test", 10, None, None, None);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires dense vectors")
    );
}

#[test]
fn test_hybrid_search_k_zero() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test document",
            serde_json::json!({}),
        )
        .unwrap();
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    // k=0 should return an error (HNSW requires k > 0)
    let result = store.search_hybrid(&query, "test", 0, None, None, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("k=0"));
}

#[test]
fn test_hybrid_search_dimension_mismatch() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    store
        .set_with_text(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "test document",
            serde_json::json!({}),
        )
        .unwrap();
    store.flush().unwrap();

    // Query with wrong dimension (8 instead of 4)
    let wrong_query = Vector::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let result = store.search_hybrid(&wrong_query, "test", 10, None, None, None);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("dimension 8 does not match store dimension 4")
    );
}

#[test]
fn test_hybrid_search_large_k() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // Insert only 3 documents
    for i in 0..3 {
        store
            .set_with_text(
                &format!("doc{i}"),
                Vector::new(vec![1.0, 0.0, 0.0, i as f32]),
                &format!("document {i}"),
                serde_json::json!({}),
            )
            .unwrap();
    }
    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    // Request more results than available
    let results = store
        .search_hybrid(&query, "document", 100, None, None, None)
        .unwrap();
    // Should return at most 3 (the number of documents)
    assert!(results.len() <= 3);
}

#[test]
fn test_hybrid_search_without_text_enabled() {
    let store = VectorStore::new(4);
    // Don't enable text search

    store
        .set(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            serde_json::json!({}),
        )
        .unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let result = store.search_hybrid(&query, "test", 10, None, None, None);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires text search")
    );
}

#[test]
fn test_hybrid_search_with_subscores() {
    let mut store = VectorStore::new(4);
    store.enable_text_search().unwrap();

    // doc1: matches both vector and text
    store
        .set_with_text(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning algorithms",
            serde_json::json!({}),
        )
        .unwrap();

    // doc2: matches text only (very different vector)
    store
        .set_with_text(
            "doc2",
            Vector::new(vec![0.0, 0.0, 0.0, 1.0]),
            "machine learning models",
            serde_json::json!({}),
        )
        .unwrap();

    // doc3: matches vector only (no matching text)
    store
        .set_with_text(
            "doc3",
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "cooking recipes",
            serde_json::json!({}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let results = store
        .search_hybrid_with_subscores(&query, "machine learning", 3, None, None, None)
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
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2024}),
        )
        .unwrap();

    store
        .set_with_text(
            "doc2",
            Vector::new(vec![0.9, 0.1, 0.0, 0.0]),
            "machine learning",
            serde_json::json!({"year": 2023}),
        )
        .unwrap();

    store.flush().unwrap();

    let query = Vector::new(vec![1.0, 0.0, 0.0, 0.0]);
    let filter = MetadataFilter::Gte("year".to_string(), 2024.0);

    let results = store
        .search_hybrid_with_subscores(&query, "machine learning", 10, Some(&filter), None, None)
        .unwrap();

    // Only doc1 should match (year >= 2024)
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.id, "doc1");
    assert!(results[0].0.keyword_score.is_some());
    assert!(results[0].0.semantic_score.is_some());
}

#[test]
fn test_store_with_text_single_vector() {
    let mut store = VectorStore::new(3);
    store.enable_text_search().unwrap();

    store
        .store_with_text(
            "doc1",
            vec![1.0f32, 0.0, 0.0],
            "machine learning algorithms",
            serde_json::json!({"type": "article"}),
        )
        .unwrap();

    store.flush().unwrap();

    // Text search should find it
    let text_results = store.search_text("machine", 10).unwrap();
    assert_eq!(text_results.len(), 1);
    assert_eq!(text_results[0].0, "doc1");

    // Vector search should also work
    let query = Vector::new(vec![1.0, 0.0, 0.0]);
    let vec_results = store.search(&query, 1, None).unwrap();
    assert_eq!(vec_results.len(), 1);
    assert_eq!(vec_results[0].id, "doc1");
}

#[test]
fn test_store_with_text_multi_vector() {
    let mut store = VectorStore::multi_vector(32).unwrap();
    store.enable_text_search().unwrap();

    let tokens = vec![vec![1.0f32; 32], vec![0.5f32; 32]];
    store
        .store_with_text(
            "doc1",
            tokens,
            "machine learning for search",
            serde_json::json!({}),
        )
        .unwrap();

    store.flush().unwrap();

    // BM25 should find the document
    let text_results = store.search_text("machine", 10).unwrap();
    assert_eq!(text_results.len(), 1);
    assert_eq!(text_results[0].0, "doc1");
}

#[test]
fn test_store_with_text_requires_enabled() {
    let mut store = VectorStore::new(3);

    let result = store.store_with_text(
        "doc1",
        vec![1.0f32, 0.0, 0.0],
        "some text",
        serde_json::json!({}),
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("requires text search")
    );
}

#[test]
fn test_index_text_standalone() {
    let mut store = VectorStore::new(3);
    store.enable_text_search().unwrap();

    // Insert vector first
    store
        .set(
            "doc1",
            Vector::new(vec![1.0, 0.0, 0.0]),
            serde_json::json!({}),
        )
        .unwrap();

    // Index text separately
    store.index_text("doc1", "machine learning").unwrap();
    store.flush().unwrap();

    let results = store.search_text("machine", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_index_text_requires_enabled() {
    let mut store = VectorStore::new(3);

    let result = store.index_text("doc1", "some text");
    assert!(result.is_err());
}
