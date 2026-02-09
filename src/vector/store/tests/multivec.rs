use super::super::*;
use crate::vector::muvera::MuveraConfig;

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

mod muvera {
    use super::*;

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

mod unified_api {
    use super::*;

    fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim).map(|i| ((seed + i) as f32) * 0.1).collect()
    }

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

mod persistence_tests {
    use super::*;
    use tempfile::tempdir;

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
