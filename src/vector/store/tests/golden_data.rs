use super::super::*;
use serde_json::json;

/// Helper to create a deterministic vector for golden tests
fn golden_vector(id: usize, dim: usize) -> Vector {
    let mut data = vec![0.0f32; dim];
    for i in 0..dim {
        // Deterministic but non-trivial values
        data[i] = (((id + i) as f32) * 0.1337).sin();
    }
    Vector::new(data)
}

#[test]
fn test_golden_hybrid_ranking_stability() {
    let mut store = VectorStore::new(128);
    store.enable_text_search().unwrap();

    // 1. Seed with deterministic documents
    let docs = [
        (
            0,
            "the quick brown fox jumps over the lazy dog [animal]",
            json!({"tag": "animal"}),
        ),
        (
            1,
            "machine learning algorithms for vector databases [tech]",
            json!({"tag": "tech"}),
        ),
        (
            2,
            "rust programming language is fast and safe [tech]",
            json!({"tag": "tech"}),
        ),
        (
            3,
            "cooking recipes for delicious apple pie [food]",
            json!({"tag": "food"}),
        ),
        (
            4,
            "deep learning models and neural networks [tech]",
            json!({"tag": "tech"}),
        ),
    ];

    for (id_num, text, metadata) in &docs {
        let id = format!("doc_{id_num}");
        let vec = golden_vector(*id_num, 128);
        store
            .set_with_text(&id, vec, text, metadata.clone())
            .unwrap();
    }

    store.flush().unwrap();

    // 2. Hybrid Search: Vector (similar to doc 1) + Text ("learning")
    let query_vec = golden_vector(1, 128);
    let results = store
        .search_hybrid(&query_vec, "learning", 5, None, Some(0.5), None)
        .unwrap();

    // Verify deterministic top results
    // Doc 1 and 4 both have "learning", but Doc 1 is exact vector match
    assert!(!results.is_empty(), "Results should not be empty");
    assert_eq!(
        results[0].0, "doc_1",
        "Doc 1 should rank first (exact vector match + text match)"
    );
    assert_eq!(
        results[1].0, "doc_4",
        "Doc 4 should rank second (text match + some vector similarity)"
    );

    // 3. Pure Text Search Stability
    let text_results = store.search_text("tech", 10).unwrap();
    assert_eq!(text_results.len(), 3, "Should find 3 tech documents");

    // Check specific IDs in order (tantivy BM25 should be deterministic)
    let ids: Vec<String> = text_results.into_iter().map(|(id, _)| id).collect();
    assert!(ids.contains(&"doc_1".to_string()));
    assert!(ids.contains(&"doc_2".to_string()));
    assert!(ids.contains(&"doc_4".to_string()));
}

#[test]
fn test_golden_filtered_search_stability() {
    let store = VectorStore::new(64);

    for i in 0..100 {
        let id = format!("id_{i}");
        let vec = golden_vector(i, 64);
        let category = if i % 2 == 0 { "even" } else { "odd" };
        store
            .set(&id, vec, json!({"cat": category, "val": i}))
            .unwrap();
    }

    let query = golden_vector(42, 64);
    let filter = MetadataFilter::Eq("cat".to_string(), json!("even"));

    let results = store.search(&query, 10, Some(&filter)).unwrap();

    assert_eq!(results.len(), 10);
    for res in &results {
        assert_eq!(res.metadata["cat"], "even");
    }
    // id_42 should be the top result (exact vector match)
    assert_eq!(results[0].id, "id_42");
}
