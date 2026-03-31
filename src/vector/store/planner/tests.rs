use super::*;
use crate::vector::store::VectorStore;
use crate::vector::store::tests::random_vector;
use crate::vector::store::input::HybridParams;

#[test]
fn test_planner_dense_search() {
    let mut store = VectorStore::new(128);
    for i in 0..10 {
        store.insert(random_vector(128, i)).unwrap();
    }

    let engine = store.segments.read();
    let engine = engine.as_ref().unwrap();
    
    let planner = QueryPlanner::new(&store.records, engine, None, None);
    let query = random_vector(128, 5);
    let results = planner.search_dense(&query.data, 3, 50).unwrap();

    assert_eq!(results.len(), 3);
}

#[test]
fn test_planner_hybrid_search() {
    let mut store = VectorStore::new(128);
    store.enable_text_search().unwrap();
    
    for i in 0..10 {
        let id = format!("doc{}", i);
        let text = if i % 2 == 0 { "apple pie" } else { "orange juice" };
        store.set_with_text(&id, random_vector(128, i), text, serde_json::json!({})).unwrap();
    }
    store.flush().unwrap(); // Flush commits text index

    let engine = store.segments.read();
    let engine = engine.as_ref().unwrap();
    let text_index = store.text_index.read();
    let text_engine = text_index.as_ref().map(|t| t as &dyn TextEngine);
    
    let planner = QueryPlanner::new(&store.records, engine, text_engine, None);
    let query = random_vector(128, 0);
    let params = HybridParams::default().alpha(0.5);
    
    let results = planner.search_hybrid(&query.data, "apple", 3, &params).unwrap();

    assert_eq!(results.len(), 3);
    // Even results (0, 2, 4, 6, 8) should rank higher due to text match "apple"
}
