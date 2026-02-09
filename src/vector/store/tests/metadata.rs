use super::super::*;

#[test]
fn test_metadata_filter_eq() {
    let filter = MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice"));

    let metadata1 = serde_json::json!({"author": "Alice"});
    let metadata2 = serde_json::json!({"author": "Bob"});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
}

#[test]
fn test_metadata_filter_gte() {
    let filter = MetadataFilter::Gte("year".to_string(), 2020.0);

    let metadata1 = serde_json::json!({"year": 2024});
    let metadata2 = serde_json::json!({"year": 2019});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
}

#[test]
fn test_metadata_filter_and() {
    let filter = MetadataFilter::And(vec![
        MetadataFilter::Eq("author".to_string(), serde_json::json!("Alice")),
        MetadataFilter::Gte("year".to_string(), 2020.0),
    ]);

    let metadata1 = serde_json::json!({"author": "Alice", "year": 2024});
    let metadata2 = serde_json::json!({"author": "Alice", "year": 2019});
    let metadata3 = serde_json::json!({"author": "Bob", "year": 2024});

    assert!(filter.matches(&metadata1));
    assert!(!filter.matches(&metadata2));
    assert!(!filter.matches(&metadata3));
}
