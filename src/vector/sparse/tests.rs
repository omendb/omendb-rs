use super::*;
use roaring::RoaringBitmap;
use std::collections::HashMap;

#[test]
fn sparse_vector_new_valid() {
    let sv = SparseVector::new(vec![1, 5, 10], vec![0.1, 0.5, 1.0]).unwrap();
    assert_eq!(sv.nnz(), 3);
    assert!(!sv.is_empty());
}

#[test]
fn sparse_vector_new_empty() {
    let sv = SparseVector::new(vec![], vec![]).unwrap();
    assert_eq!(sv.nnz(), 0);
    assert!(sv.is_empty());
}

#[test]
fn sparse_vector_new_length_mismatch() {
    let result = SparseVector::new(vec![1, 2], vec![0.1]);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_new_unsorted() {
    let result = SparseVector::new(vec![5, 1, 10], vec![0.1, 0.5, 1.0]);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_new_duplicate_indices() {
    let result = SparseVector::new(vec![1, 1, 5], vec![0.1, 0.5, 1.0]);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_new_rejects_nan() {
    let result = SparseVector::new(vec![1, 5], vec![1.0, f32::NAN]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("finite"));
}

#[test]
fn sparse_vector_new_rejects_infinity() {
    let result = SparseVector::new(vec![1, 5], vec![f32::INFINITY, 1.0]);
    assert!(result.is_err());

    let result = SparseVector::new(vec![1, 5], vec![1.0, f32::NEG_INFINITY]);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_from_pairs() {
    let sv = SparseVector::from_pairs(vec![(10, 1.0), (1, 0.1), (5, 0.5)]).unwrap();
    assert_eq!(sv.indices(), &[1, 5, 10]);
    assert_eq!(sv.values(), &[0.1, 0.5, 1.0]);
}

#[test]
fn sparse_vector_from_pairs_dedup() {
    // First occurrence wins when duplicates exist
    let sv = SparseVector::from_pairs(vec![(1, 0.1), (1, 0.9), (5, 0.5)]).unwrap();
    assert_eq!(sv.indices().len(), 2);
    assert_eq!(sv.values()[0], 0.1); // first occurrence kept
}

#[test]
fn sparse_vector_from_pairs_rejects_nan() {
    let result = SparseVector::from_pairs(vec![(1, f32::NAN), (5, 0.5)]);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_from_map() {
    let mut map = HashMap::new();
    map.insert(10u32, 1.0f32);
    map.insert(1, 0.1);
    map.insert(5, 0.5);
    let sv = SparseVector::from_map(map).unwrap();
    assert_eq!(sv.indices(), &[1, 5, 10]);
    assert_eq!(sv.values(), &[0.1, 0.5, 1.0]);
}

#[test]
fn sparse_vector_from_map_rejects_nan() {
    let mut map = HashMap::new();
    map.insert(1u32, f32::NAN);
    let result = SparseVector::from_map(map);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_dot_product() {
    let a = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
    let b = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
    assert!((a.dot(&b) - 14.0).abs() < 1e-6); // 1+4+9
}

#[test]
fn sparse_vector_dot_product_partial_overlap() {
    let a = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
    let b = SparseVector::new(vec![5, 20], vec![3.0, 1.0]).unwrap();
    assert!((a.dot(&b) - 6.0).abs() < 1e-6); // only dim 5: 2*3=6
}

#[test]
fn sparse_vector_dot_product_no_overlap() {
    let a = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    let b = SparseVector::new(vec![10, 20], vec![3.0, 4.0]).unwrap();
    assert!((a.dot(&b)).abs() < 1e-6);
}

#[test]
fn sparse_vector_dot_empty() {
    let a = SparseVector::new(vec![], vec![]).unwrap();
    let b = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    assert!((a.dot(&b)).abs() < 1e-6);
}

#[test]
fn sparse_vector_zero_weight() {
    // Zero weights are valid (finite) but contribute nothing to dot product
    let sv = SparseVector::new(vec![1, 5], vec![0.0, 1.0]).unwrap();
    let query = SparseVector::new(vec![1, 5], vec![1.0, 1.0]).unwrap();
    assert!((sv.dot(&query) - 1.0).abs() < 1e-6); // 0*1 + 1*1 = 1
}

#[test]
fn sparse_vector_serialization_roundtrip() {
    let sv = SparseVector::new(vec![1, 100, 30000], vec![0.5, 1.5, 2.5]).unwrap();
    let bytes = sv.to_bytes().unwrap();
    let sv2 = SparseVector::from_bytes(&bytes).unwrap();
    assert_eq!(sv, sv2);
}

#[test]
fn sparse_vector_from_bytes_rejects_unsorted() {
    // Craft a valid postcard serialization with unsorted indices
    let bad = SparseVector {
        indices: vec![5, 1],
        values: vec![1.0, 2.0],
    };
    let bytes = postcard::to_allocvec(&bad).unwrap();
    let result = SparseVector::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn sparse_vector_from_bytes_rejects_nan() {
    let bad = SparseVector {
        indices: vec![1, 5],
        values: vec![1.0, f32::NAN],
    };
    let bytes = postcard::to_allocvec(&bad).unwrap();
    let result = SparseVector::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn sparse_index_insert_and_search() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
    let v2 = SparseVector::new(vec![1, 5, 20], vec![0.5, 1.0, 4.0]).unwrap();
    let v3 = SparseVector::new(vec![10, 20, 30], vec![1.0, 1.0, 1.0]).unwrap();

    index.insert(0, &v1);
    index.insert(1, &v2);
    index.insert(2, &v3);

    assert_eq!(index.len(), 3);

    // Query overlapping with v1 most
    let query = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]).unwrap();
    let results = index.search(&query, 2);
    assert_eq!(results.len(), 2);
    // v1 should be first (dot = 1+4+9 = 14)
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 14.0).abs() < 1e-6);
}

#[test]
fn sparse_index_remove() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    let v2 = SparseVector::new(vec![1, 5], vec![0.5, 1.0]).unwrap();

    index.insert(0, &v1);
    index.insert(1, &v2);
    assert_eq!(index.len(), 2);

    index.remove(0);
    assert_eq!(index.len(), 1);

    let query = SparseVector::new(vec![1, 5], vec![1.0, 1.0]).unwrap();
    let results = index.search(&query, 10);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
}

#[test]
fn sparse_index_remove_nonexistent() {
    let mut index = SparseIndex::new();
    index.remove(999);
}

#[test]
fn sparse_index_remove_cleans_empty_posting_lists() {
    let mut index = SparseIndex::new();

    // Only one vector uses dim 42
    let v1 = SparseVector::new(vec![42], vec![1.0]).unwrap();
    index.insert(0, &v1);
    index.remove(0);

    // Search on dim 42 should return nothing (posting list removed)
    let query = SparseVector::new(vec![42], vec![1.0]).unwrap();
    let results = index.search(&query, 10);
    assert!(results.is_empty());
}

#[test]
fn sparse_index_upsert() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    index.insert(0, &v1);

    // Update same slot with different vector
    let v1_updated = SparseVector::new(vec![10, 20], vec![3.0, 4.0]).unwrap();
    index.insert(0, &v1_updated);
    assert_eq!(index.len(), 1);

    // Search should find new vector, not old
    let query = SparseVector::new(vec![10, 20], vec![1.0, 1.0]).unwrap();
    let results = index.search(&query, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 7.0).abs() < 1e-6); // 3+4
}

#[test]
fn sparse_index_search_with_bitmap() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    let v2 = SparseVector::new(vec![1, 5], vec![3.0, 4.0]).unwrap();
    let v3 = SparseVector::new(vec![1, 5], vec![0.1, 0.2]).unwrap();

    index.insert(0, &v1);
    index.insert(1, &v2);
    index.insert(2, &v3);

    // Only allow slots 0 and 2
    let mut allowed = RoaringBitmap::new();
    allowed.insert(0);
    allowed.insert(2);

    let query = SparseVector::new(vec![1, 5], vec![1.0, 1.0]).unwrap();
    let results = index.search_with_bitmap(&query, 10, &allowed);

    // Should not include slot 1 (highest score) since it's filtered out
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(id, _)| *id != 1));
    // Slot 0 should be first (score 3.0 vs 0.3)
    assert_eq!(results[0].0, 0);
}

#[test]
fn sparse_index_compact() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    let v2 = SparseVector::new(vec![1, 5], vec![3.0, 4.0]).unwrap();

    index.insert(10, &v1);
    index.insert(20, &v2);

    // Remap: 10 -> 0, 20 -> 1
    let mut mapping = vec![u32::MAX; 21];
    mapping[10] = 0;
    mapping[20] = 1;
    index.compact(&mapping);

    assert_eq!(index.len(), 2);

    let query = SparseVector::new(vec![1, 5], vec![1.0, 1.0]).unwrap();
    let results = index.search(&query, 2);
    // Results should use new slot IDs
    assert!(results.iter().any(|(id, _)| *id == 0));
    assert!(results.iter().any(|(id, _)| *id == 1));
}

#[test]
fn sparse_index_compact_removes_deleted_slots() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1], vec![1.0]).unwrap();
    let v2 = SparseVector::new(vec![1], vec![2.0]).unwrap();
    let v3 = SparseVector::new(vec![1], vec![3.0]).unwrap();

    index.insert(0, &v1);
    index.insert(1, &v2);
    index.insert(2, &v3);

    // Compact but only remap slots 0 and 2 (slot 1 was deleted)
    let mut mapping = vec![u32::MAX; 3];
    mapping[0] = 0;
    mapping[2] = 1;
    index.compact(&mapping);

    assert_eq!(index.len(), 2);

    let query = SparseVector::new(vec![1], vec![1.0]).unwrap();
    let results = index.search(&query, 10);
    assert_eq!(results.len(), 2);
    // No result should reference old slot 1 or 2
    assert!(results.iter().all(|(id, _)| *id == 0 || *id == 1));
}

#[test]
fn sparse_index_remap_slot() {
    let mut index = SparseIndex::new();

    let v = SparseVector::new(vec![1, 5], vec![1.0, 2.0]).unwrap();
    index.insert(10, &v);

    index.remap_slot(10, 20);
    assert_eq!(index.len(), 1);

    let query = SparseVector::new(vec![1, 5], vec![1.0, 1.0]).unwrap();
    let results = index.search(&query, 1);
    assert_eq!(results[0].0, 20); // new slot ID
}

#[test]
fn sparse_index_remap_nonexistent() {
    let mut index = SparseIndex::new();
    index.remap_slot(999, 1000);
}

#[test]
fn sparse_index_serialization_roundtrip() {
    let mut index = SparseIndex::new();

    let v1 = SparseVector::new(vec![1, 5, 100], vec![1.0, 2.0, 3.0]).unwrap();
    let v2 = SparseVector::new(vec![5, 100, 200], vec![0.5, 1.5, 2.5]).unwrap();

    index.insert(0, &v1);
    index.insert(1, &v2);

    let bytes = index.to_bytes().unwrap();
    let index2 = SparseIndex::from_bytes(&bytes).unwrap();

    assert_eq!(index2.len(), 2);

    // Verify search produces same results
    let query = SparseVector::new(vec![5, 100], vec![1.0, 1.0]).unwrap();
    let r1 = index.search(&query, 2);
    let r2 = index2.search(&query, 2);
    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6);
    }
}

#[test]
fn sparse_index_search_empty() {
    let index = SparseIndex::new();
    let query = SparseVector::new(vec![1], vec![1.0]).unwrap();
    let results = index.search(&query, 10);
    assert!(results.is_empty());
}

#[test]
fn sparse_index_search_empty_query() {
    let mut index = SparseIndex::new();
    let v = SparseVector::new(vec![1], vec![1.0]).unwrap();
    index.insert(0, &v);

    // Empty query should return no results (no overlapping dims)
    let query = SparseVector::new(vec![], vec![]).unwrap();
    let results = index.search(&query, 10);
    assert!(results.is_empty());
}

#[test]
fn sparse_index_search_k_zero() {
    let mut index = SparseIndex::new();
    let v = SparseVector::new(vec![1], vec![1.0]).unwrap();
    index.insert(0, &v);
    let results = index.search(&v, 0);
    assert!(results.is_empty());
}

#[test]
fn sparse_index_search_k_greater_than_results() {
    let mut index = SparseIndex::new();
    let v = SparseVector::new(vec![1], vec![1.0]).unwrap();
    index.insert(0, &v);
    let results = index.search(&v, 100);
    assert_eq!(results.len(), 1);
}

#[test]
fn sparse_index_large_dim_ids() {
    // Boundary test with large dimension indices
    let sv = SparseVector::new(vec![0, u32::MAX - 1, u32::MAX], vec![1.0, 2.0, 3.0]).unwrap();
    let mut index = SparseIndex::new();
    index.insert(0, &sv);

    let query = SparseVector::new(vec![u32::MAX], vec![1.0]).unwrap();
    let results = index.search(&query, 1);
    assert_eq!(results.len(), 1);
    assert!((results[0].1 - 3.0).abs() < 1e-6);
}
