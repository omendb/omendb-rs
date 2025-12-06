use super::*;

#[test]
fn test_text_index_in_memory() {
    let mut index = TextIndex::open_in_memory().unwrap();

    index.index_document("doc1", "hello world").unwrap();
    index.index_document("doc2", "goodbye world").unwrap();
    index.commit().unwrap();

    let results = index.search("hello", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");

    let results = index.search("world", 10).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_text_index_update() {
    let mut index = TextIndex::open_in_memory().unwrap();

    index.index_document("doc1", "original content").unwrap();
    index.commit().unwrap();

    let results = index.search("original", 10).unwrap();
    assert_eq!(results.len(), 1);

    // Update document
    index.index_document("doc1", "updated content").unwrap();
    index.commit().unwrap();

    let results = index.search("original", 10).unwrap();
    assert_eq!(results.len(), 0);

    let results = index.search("updated", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_text_index_delete() {
    let mut index = TextIndex::open_in_memory().unwrap();

    index.index_document("doc1", "hello world").unwrap();
    index.index_document("doc2", "goodbye world").unwrap();
    index.commit().unwrap();

    assert_eq!(index.num_docs(), 2);

    index.delete_document("doc1").unwrap();
    index.commit().unwrap();

    // Note: tantivy soft-deletes, so num_docs may not immediately reflect deletion
    let results = index.search("hello", 10).unwrap();
    assert_eq!(results.len(), 0);

    let results = index.search("world", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc2");
}

#[test]
fn test_text_index_empty_query() {
    let mut index = TextIndex::open_in_memory().unwrap();

    index.index_document("doc1", "hello world").unwrap();
    index.commit().unwrap();

    let results = index.search("", 10).unwrap();
    assert!(results.is_empty());

    let results = index.search("   ", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_text_index_bm25_scoring() {
    let mut index = TextIndex::open_in_memory().unwrap();

    // doc1 has "rust" once, doc2 has "rust" twice
    index
        .index_document("doc1", "rust programming language")
        .unwrap();
    index
        .index_document("doc2", "rust rust systems programming")
        .unwrap();
    index.commit().unwrap();

    let results = index.search("rust", 10).unwrap();
    assert_eq!(results.len(), 2);

    // doc2 should score higher due to higher term frequency
    assert_eq!(results[0].0, "doc2");
    assert_eq!(results[1].0, "doc1");
    assert!(results[0].1 > results[1].1);
}

#[test]
fn test_text_index_persistence() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("text_index");

    // Create and populate index
    {
        let mut index = TextIndex::open(&path).unwrap();
        index.index_document("doc1", "persistent data").unwrap();
        index.commit().unwrap();
    }

    // Reopen and verify
    {
        let index = TextIndex::open(&path).unwrap();
        let results = index.search("persistent", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc1");
    }
}

#[test]
fn test_rrf_basic() {
    let vector_results = vec![
        ("doc1".to_string(), 0.1), // rank 0 (closest)
        ("doc2".to_string(), 0.2), // rank 1
        ("doc3".to_string(), 0.3), // rank 2
    ];

    let text_results = vec![
        ("doc2".to_string(), 10.0), // rank 0 (highest BM25)
        ("doc1".to_string(), 8.0),  // rank 1
        ("doc4".to_string(), 5.0),  // rank 2
    ];

    let results = reciprocal_rank_fusion(vector_results, text_results, 10, 60);

    // doc1 and doc2 appear in both, should have highest scores
    assert!(results.len() >= 2);

    // doc1: vector rank 0 (1/61) + text rank 1 (1/62)
    // doc2: vector rank 1 (1/62) + text rank 0 (1/61)
    // They should have equal scores since ranks are symmetric
    let doc1_score = results.iter().find(|(id, _)| id == "doc1").unwrap().1;
    let doc2_score = results.iter().find(|(id, _)| id == "doc2").unwrap().1;

    // Both should have same RRF score: 0.5*(1/61) + 0.5*(1/62) (default alpha=0.5)
    let expected = 0.5 * (1.0 / 61.0 + 1.0 / 62.0);
    assert!((doc1_score - expected).abs() < 0.0001);
    assert!((doc2_score - expected).abs() < 0.0001);
}

#[test]
fn test_rrf_disjoint_results() {
    let vector_results = vec![("doc1".to_string(), 0.1), ("doc2".to_string(), 0.2)];

    let text_results = vec![("doc3".to_string(), 10.0), ("doc4".to_string(), 8.0)];

    let results = reciprocal_rank_fusion(vector_results, text_results, 10, 60);

    // All 4 docs should appear
    assert_eq!(results.len(), 4);

    // Top results should be rank 0 from each (doc1 from vector, doc3 from text)
    let top_ids: Vec<_> = results.iter().take(2).map(|(id, _)| id.as_str()).collect();
    assert!(top_ids.contains(&"doc1") || top_ids.contains(&"doc3"));
}

#[test]
fn test_rrf_limit() {
    let vector_results: Vec<_> = (0..100)
        .map(|i| (format!("vec_{}", i), i as f32 * 0.1))
        .collect();

    let text_results: Vec<_> = (0..100)
        .map(|i| (format!("text_{}", i), 100.0 - i as f32))
        .collect();

    let results = reciprocal_rank_fusion(vector_results, text_results, 10, 60);

    // Should only return top 10
    assert_eq!(results.len(), 10);
}

#[test]
fn test_rrf_empty_inputs() {
    let results = reciprocal_rank_fusion(vec![], vec![], 10, 60);
    assert!(results.is_empty());

    let vector_only = vec![("doc1".to_string(), 0.1)];
    let results = reciprocal_rank_fusion(vector_only, vec![], 10, 60);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "doc1");
}

#[test]
fn test_weighted_rrf_alpha_extremes() {
    let vector_results = vec![
        ("vec_doc".to_string(), 0.1), // rank 0 in vector
    ];

    let text_results = vec![
        ("text_doc".to_string(), 10.0), // rank 0 in text
    ];

    // alpha=1.0: vector only
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, 1.0);
    assert_eq!(results[0].0, "vec_doc");
    // text_doc should have score 0
    let text_score = results.iter().find(|(id, _)| id == "text_doc").unwrap().1;
    assert!(text_score < 0.0001);

    // alpha=0.0: text only
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, 0.0);
    assert_eq!(results[0].0, "text_doc");
    // vec_doc should have score 0
    let vec_score = results.iter().find(|(id, _)| id == "vec_doc").unwrap().1;
    assert!(vec_score < 0.0001);
}

#[test]
fn test_weighted_rrf_alpha_balanced() {
    let vector_results = vec![
        ("doc1".to_string(), 0.1), // rank 0
        ("doc2".to_string(), 0.2), // rank 1
    ];

    let text_results = vec![
        ("doc2".to_string(), 10.0), // rank 0
        ("doc1".to_string(), 8.0),  // rank 1
    ];

    // alpha=0.5 (default): balanced
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, 0.5);

    let doc1_score = results.iter().find(|(id, _)| id == "doc1").unwrap().1;
    let doc2_score = results.iter().find(|(id, _)| id == "doc2").unwrap().1;

    // With alpha=0.5, both get equal weight
    // doc1: 0.5 * 1/61 + 0.5 * 1/62
    // doc2: 0.5 * 1/62 + 0.5 * 1/61
    // Should be equal
    assert!((doc1_score - doc2_score).abs() < 0.0001);
}

#[test]
fn test_weighted_rrf_alpha_bias_vector() {
    let vector_results = vec![
        ("vec_winner".to_string(), 0.1), // rank 0
    ];

    let text_results = vec![
        ("text_winner".to_string(), 10.0), // rank 0
    ];

    // alpha=0.8: heavily favor vector
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, 0.8);

    let vec_score = results.iter().find(|(id, _)| id == "vec_winner").unwrap().1;
    let text_score = results
        .iter()
        .find(|(id, _)| id == "text_winner")
        .unwrap()
        .1;

    // vec should score 4x higher (0.8 vs 0.2)
    assert!((vec_score / text_score - 4.0).abs() < 0.01);
}

#[test]
fn test_weighted_rrf_alpha_clamping() {
    let vector_results = vec![("doc1".to_string(), 0.1)];
    let text_results = vec![("doc2".to_string(), 10.0)];

    // alpha > 1.0 should clamp to 1.0
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, 1.5);
    assert_eq!(results[0].0, "doc1"); // vector only

    // alpha < 0.0 should clamp to 0.0
    let results =
        weighted_reciprocal_rank_fusion(vector_results.clone(), text_results.clone(), 10, 60, -0.5);
    assert_eq!(results[0].0, "doc2"); // text only
}

#[test]
fn test_default_rrf_k_constant() {
    assert_eq!(DEFAULT_RRF_K, 60);
}
