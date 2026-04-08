use crate::vector::VectorStore;
use crate::vector::store::SearchOptions;
use serde_json::json;

#[test]
fn test_maxsim_explain() {
    let mut store = VectorStore::multi_vector(16).unwrap();

    // Document with 2 distinct tokens
    let mut doc_token0 = vec![0.0; 16];
    doc_token0[0] = 1.0;
    let mut doc_token1 = vec![0.0; 16];
    doc_token1[1] = 1.0;

    let doc_tokens = vec![doc_token0, doc_token1];
    store.store("doc1", doc_tokens, json!({})).unwrap();

    // Query with 2 distinct tokens matching doc tokens
    let mut query_token0 = vec![0.0; 16];
    query_token0[0] = 0.9;
    let mut query_token1 = vec![0.0; 16];
    query_token1[1] = 0.8;

    let query_tokens = vec![query_token0, query_token1];

    let options = SearchOptions::new().explain();
    let results = store
        .query_with_options(&query_tokens, 10, &options)
        .unwrap();

    assert_eq!(results.len(), 1);
    let res = &results[0];
    assert_eq!(res.id, "doc1");

    let explanation = res.explanation.as_ref().expect("Should have explanation");
    let matches = explanation["matches"]
        .as_array()
        .expect("Should have matches array");
    let sub_scores = explanation["sub_scores"]
        .as_array()
        .expect("Should have sub_scores array");

    assert_eq!(matches.len(), 2);
    assert_eq!(sub_scores.len(), 2);

    // Query Token 0 (approx [1,0,0]) should match Doc Token 0
    assert_eq!(matches[0].as_u64().unwrap(), 0);
    // Query Token 1 (approx [0,1,0]) should match Doc Token 1
    assert_eq!(matches[1].as_u64().unwrap(), 1);

    // Verify scores are positive (dot product)
    assert!(sub_scores[0].as_f64().unwrap() > 0.8);
    assert!(sub_scores[1].as_f64().unwrap() > 0.7);
}
