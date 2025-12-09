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
        .map(|i| (format!("vec_{i}"), i as f32 * 0.1))
        .collect();

    let text_results: Vec<_> = (0..100)
        .map(|i| (format!("text_{i}"), 100.0 - i as f32))
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

#[test]
fn test_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let mut index = TextIndex::open_in_memory().unwrap();

    // Index some documents
    for i in 0..100 {
        index
            .index_document(&format!("doc{i}"), &format!("content {i} searchable"))
            .unwrap();
    }
    index.commit().unwrap();

    // Share the reader across threads
    let reader = Arc::new(index.reader().clone());
    let index_ref = Arc::new(index.index().clone());

    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let reader = Arc::clone(&reader);
            let index = Arc::clone(&index_ref);
            thread::spawn(move || {
                let searcher = reader.searcher();
                let text_field = index.schema().get_field("text").unwrap();
                let query_parser = tantivy::query::QueryParser::for_index(&index, vec![text_field]);

                // Each thread performs multiple searches
                for i in 0..25 {
                    let query = query_parser.parse_query("searchable").unwrap();
                    let results = searcher
                        .search(&query, &tantivy::collector::TopDocs::with_limit(10))
                        .unwrap();
                    assert!(!results.is_empty(), "Thread {thread_id} iteration {i}");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }
}

#[test]
fn test_read_while_write() {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("text_index");

    // Create index and add initial documents
    let mut index = TextIndex::open(&path).unwrap();
    for i in 0..50 {
        index
            .index_document(&format!("doc{i}"), &format!("initial content {i}"))
            .unwrap();
    }
    index.commit().unwrap();

    // Share reader for concurrent reads
    let reader = Arc::new(index.reader().clone());
    let index_ref = Arc::new(index.index().clone());

    // Spawn reader threads
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let reader = Arc::clone(&reader);
            let index = Arc::clone(&index_ref);
            thread::spawn(move || {
                let text_field = index.schema().get_field("text").unwrap();
                let query_parser = tantivy::query::QueryParser::for_index(&index, vec![text_field]);

                for _ in 0..50 {
                    let searcher = reader.searcher();
                    let query = query_parser.parse_query("content").unwrap();
                    let results = searcher
                        .search(&query, &tantivy::collector::TopDocs::with_limit(10))
                        .unwrap();
                    // Results may vary as writes happen, but should never error
                    assert!(results.len() <= 10);
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            })
        })
        .collect();

    // Continue writing while reads happen
    for i in 50..100 {
        index
            .index_document(&format!("doc{i}"), &format!("new content {i}"))
            .unwrap();
        if i % 10 == 0 {
            index.commit().unwrap();
        }
    }
    index.commit().unwrap();

    for h in handles {
        h.join().expect("Reader thread panicked");
    }

    // Final verification
    let results = index.search("content", 200).unwrap();
    assert_eq!(results.len(), 100);
}

#[test]
fn test_high_throughput_indexing() {
    let mut index = TextIndex::open_in_memory().unwrap();

    // Rapid indexing without commits
    for i in 0..1000 {
        index
            .index_document(&format!("doc{i}"), &format!("bulk content {i}"))
            .unwrap();
    }
    index.commit().unwrap();

    let results = index.search("bulk", 1000).unwrap();
    assert_eq!(results.len(), 1000);
}

#[test]
fn test_update_heavy_workload() {
    let mut index = TextIndex::open_in_memory().unwrap();

    // Create initial documents
    for i in 0..100 {
        index
            .index_document(&format!("doc{i}"), &format!("version0 content {i}"))
            .unwrap();
    }
    index.commit().unwrap();

    // Repeatedly update the same documents
    for version in 1..10 {
        for i in 0..100 {
            index
                .index_document(&format!("doc{i}"), &format!("version{version} content {i}"))
                .unwrap();
        }
        index.commit().unwrap();

        // Verify old version is gone
        let old_results = index
            .search(&format!("version{}", version - 1), 100)
            .unwrap();
        assert_eq!(old_results.len(), 0, "Old version should be deleted");

        // Verify new version is present
        let new_results = index.search(&format!("version{version}"), 100).unwrap();
        assert_eq!(new_results.len(), 100, "New version should exist");
    }
}

#[test]
fn test_mixed_operations_workload() {
    let mut index = TextIndex::open_in_memory().unwrap();

    // Interleaved inserts, updates, and deletes
    for round in 0..5 {
        // Insert new docs
        for i in 0..20 {
            let id = round * 20 + i;
            index
                .index_document(
                    &format!("doc{id}"),
                    &format!("workload data round{round} item{i}"),
                )
                .unwrap();
        }

        // Update some previous docs (if any exist)
        if round > 0 {
            for i in 0..10 {
                let id = (round - 1) * 20 + i;
                index
                    .index_document(
                        &format!("doc{id}"),
                        &format!("workload updated in round{round}"),
                    )
                    .unwrap();
            }
        }

        // Delete some docs
        if round > 1 {
            for i in 0..5 {
                let id = (round - 2) * 20 + i;
                index.delete_document(&format!("doc{id}")).unwrap();
            }
        }

        index.commit().unwrap();
    }

    // Verify final state: search for term present in all documents
    let all_results = index.search("workload", 200).unwrap();
    // 5 rounds * 20 docs = 100, minus 3 rounds * 5 deletions = 85
    assert_eq!(all_results.len(), 85);
}
