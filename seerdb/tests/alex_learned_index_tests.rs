//! Comprehensive tests for ALEX learned index implementation
//!
//! Tests cover:
//! - Node split logic (when node exceeds MAX_DENSITY)
//! - Model prediction accuracy and error bounds
//! - Batch insert optimizations
//! - Lower bound search (O(log error) guarantee)
//! - Edge cases (empty, single key, duplicates)
//! - Concurrent modifications (thread safety)

use seerdb::alex::AlexTree;
use std::sync::{Arc, Barrier};
use std::thread;

// Node split and capacity tests

#[test]
fn test_node_split_on_capacity() {
    // Use 0.0 expansion factor to force splits quickly
    let mut tree = AlexTree::with_expansion(0.0);

    // Insert enough keys to trigger multiple splits
    // Each node has capacity ~100, so 1000 keys should cause 10+ splits
    for i in 0..1000 {
        tree.insert(i, vec![i as u8]).unwrap();
    }

    // Should have created multiple leaf nodes
    assert!(
        tree.num_leaves() >= 10,
        "Expected multiple splits, got {} leaves",
        tree.num_leaves()
    );
    assert_eq!(tree.len(), 1000, "All keys should be present after splits");

    // Verify all keys are still accessible after splits
    for i in (0..1000).step_by(50) {
        assert!(
            tree.get(i).unwrap().is_some(),
            "Key {} missing after split",
            i
        );
    }
}

#[test]
fn test_split_maintains_sort_order() {
    let mut tree = AlexTree::with_expansion(0.0);

    // Insert keys that will trigger splits
    for i in 0..500 {
        tree.insert(i, vec![i as u8]).unwrap();
    }

    // Range query spanning multiple leaves should return sorted keys
    let results = tree.range(0, 499).unwrap();
    assert_eq!(results.len(), 500);

    // Verify strict ascending order
    for i in 0..results.len() - 1 {
        assert!(
            results[i].0 < results[i + 1].0,
            "Keys not in order: {} >= {}",
            results[i].0,
            results[i + 1].0
        );
    }
}

#[test]
fn test_split_with_skewed_keys() {
    // Test that split works correctly with non-uniform key distribution
    let mut tree = AlexTree::with_expansion(0.0);

    // Insert keys in clusters (simulates real-world workload with hot keys)
    // Cluster 1: 0-99
    for i in 0..100 {
        tree.insert(i, vec![1]).unwrap();
    }

    // Cluster 2: 1000-1099 (large gap)
    for i in 1000..1100 {
        tree.insert(i, vec![2]).unwrap();
    }

    // Cluster 3: 10000-10099 (even larger gap)
    for i in 10000..10100 {
        tree.insert(i, vec![3]).unwrap();
    }

    assert_eq!(tree.len(), 300);

    // All keys should be retrievable
    assert_eq!(tree.get(50).unwrap(), Some(vec![1]));
    assert_eq!(tree.get(1050).unwrap(), Some(vec![2]));
    assert_eq!(tree.get(10050).unwrap(), Some(vec![3]));

    // Keys in gaps should return None
    assert_eq!(tree.get(500).unwrap(), None);
    assert_eq!(tree.get(5000).unwrap(), None);
}

// Batch insert tests

#[test]
fn test_batch_insert_performance() {
    let mut tree = AlexTree::new();

    // Create batch of 1000 random keys
    let mut entries = Vec::new();
    for i in 0..1000 {
        // Use non-sequential keys to test routing logic
        let key = (i * 7919) % 50000; // Prime number for pseudo-randomness
        entries.push((key, vec![(i % 256) as u8]));
    }

    // Batch insert should succeed
    tree.insert_batch(entries).unwrap();

    assert_eq!(tree.len(), 1000);

    // Sample verification
    let key_sample = (100 * 7919) % 50000;
    assert!(
        tree.get(key_sample).unwrap().is_some(),
        "Batch-inserted key missing"
    );
}

#[test]
fn test_batch_insert_with_duplicates() {
    let mut tree = AlexTree::new();

    // Insert some keys
    tree.insert(10, vec![1]).unwrap();
    tree.insert(20, vec![2]).unwrap();
    assert_eq!(tree.len(), 2);

    // Batch with duplicates (ALEX allows them)
    let entries = vec![
        (10, vec![100]), // Duplicate of existing key
        (30, vec![3]),
        (20, vec![200]), // Another duplicate
        (40, vec![4]),
    ];

    tree.insert_batch(entries).unwrap();

    // Length should be 2 (original) + 4 (batch) = 6
    assert_eq!(tree.len(), 6);

    // All keys should be findable via get (returns at least one value)
    assert!(tree.get(10).unwrap().is_some());
    assert!(tree.get(20).unwrap().is_some());
    assert!(tree.get(30).unwrap().is_some());
    assert!(tree.get(40).unwrap().is_some());
}

#[test]
fn test_batch_insert_triggers_split() {
    let mut tree = AlexTree::with_expansion(0.0);

    // Insert batch that exceeds single leaf capacity
    let mut entries = Vec::new();
    for i in 0..200 {
        entries.push((i, vec![i as u8]));
    }

    tree.insert_batch(entries).unwrap();

    // Should have triggered splits
    assert!(tree.num_leaves() > 1, "Batch should trigger split");
    assert_eq!(tree.len(), 200);
}

// Lower bound search tests

#[test]
fn test_lower_bound_exact_match() {
    let mut tree = AlexTree::new();

    // Insert keys: 10, 20, 30, 40, 50
    for i in 1..=5 {
        tree.insert(i * 10, vec![i as u8]).unwrap();
    }

    // Lower bound for exact keys
    assert_eq!(tree.lower_bound(10).unwrap(), Some((10, vec![1])));
    assert_eq!(tree.lower_bound(30).unwrap(), Some((30, vec![3])));
    assert_eq!(tree.lower_bound(50).unwrap(), Some((50, vec![5])));
}

#[test]
fn test_lower_bound_between_keys() {
    let mut tree = AlexTree::new();

    // Insert keys: 10, 20, 30, 40, 50
    for i in 1..=5 {
        tree.insert(i * 10, vec![i as u8]).unwrap();
    }

    // Lower bound for keys between existing keys (should return next highest)
    assert_eq!(tree.lower_bound(15).unwrap(), Some((20, vec![2])));
    assert_eq!(tree.lower_bound(25).unwrap(), Some((30, vec![3])));
    assert_eq!(tree.lower_bound(45).unwrap(), Some((50, vec![5])));

    // Lower bound beyond all keys
    assert_eq!(tree.lower_bound(100).unwrap(), None);

    // Lower bound before all keys
    assert_eq!(tree.lower_bound(0).unwrap(), Some((10, vec![1])));
}

#[test]
fn test_lower_bound_across_splits() {
    let mut tree = AlexTree::with_expansion(0.0);

    // Insert keys that will cause splits
    for i in 0..500 {
        tree.insert(i * 2, vec![(i % 256) as u8]).unwrap(); // Even keys only
    }

    // Lower bound for odd keys (should return next even key)
    assert_eq!(tree.lower_bound(101).unwrap().map(|(k, _)| k), Some(102));
    assert_eq!(tree.lower_bound(501).unwrap().map(|(k, _)| k), Some(502));

    // Lower bound at boundaries
    assert_eq!(tree.lower_bound(0).unwrap().map(|(k, _)| k), Some(0));
    assert_eq!(tree.lower_bound(998).unwrap().map(|(k, _)| k), Some(998));
}

#[test]
fn test_lower_bound_empty_tree() {
    let tree = AlexTree::new();
    assert_eq!(tree.lower_bound(0).unwrap(), None);
    assert_eq!(tree.lower_bound(100).unwrap(), None);
}

// Edge case tests

#[test]
fn test_empty_tree_operations() {
    let tree = AlexTree::new();

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
    assert_eq!(tree.num_leaves(), 1); // Always has at least one leaf

    assert_eq!(tree.get(0).unwrap(), None);
    assert_eq!(tree.lower_bound(0).unwrap(), None);
    assert_eq!(tree.range(0, 100).unwrap().len(), 0);
}

#[test]
fn test_single_key_operations() {
    let mut tree = AlexTree::new();
    tree.insert(42, vec![1, 2, 3]).unwrap();

    assert!(!tree.is_empty());
    assert_eq!(tree.len(), 1);

    // Get
    assert_eq!(tree.get(42).unwrap(), Some(vec![1, 2, 3]));
    assert_eq!(tree.get(41).unwrap(), None);
    assert_eq!(tree.get(43).unwrap(), None);

    // Lower bound
    assert_eq!(tree.lower_bound(40).unwrap(), Some((42, vec![1, 2, 3])));
    assert_eq!(tree.lower_bound(42).unwrap(), Some((42, vec![1, 2, 3])));
    assert_eq!(tree.lower_bound(43).unwrap(), None);

    // Range
    assert_eq!(tree.range(0, 100).unwrap().len(), 1);
    assert_eq!(tree.range(42, 42).unwrap().len(), 1);
    assert_eq!(tree.range(0, 41).unwrap().len(), 0);
    assert_eq!(tree.range(43, 100).unwrap().len(), 0);
}

#[test]
fn test_duplicate_keys_allowed() {
    // ALEX allows duplicate keys (multi-map behavior)
    // This is useful for indexing where multiple values can map to same key
    let mut tree = AlexTree::new();

    // Insert key
    tree.insert(100, vec![1]).unwrap();
    assert_eq!(tree.len(), 1);

    // Insert same key with different value (ALEX allows duplicates)
    tree.insert(100, vec![2]).unwrap();
    assert_eq!(tree.len(), 2); // Length INCREASES for duplicates

    // Insert same key again
    tree.insert(100, vec![3]).unwrap();
    assert_eq!(tree.len(), 3);

    // get() returns ONE of the values (implementation-defined which one)
    let value = tree.get(100).unwrap();
    assert!(
        value.is_some(),
        "Should find at least one value for key 100"
    );

    // Range query returns all occurrences
    let results = tree.range(100, 100).unwrap();
    assert_eq!(results.len(), 3, "Range should return all duplicate keys");

    // All results should have key=100
    for (k, _) in results {
        assert_eq!(k, 100);
    }
}

// Range query tests

#[test]
fn test_range_query_edge_cases() {
    let mut tree = AlexTree::new();

    // Insert keys: 10, 20, 30
    tree.insert(10, vec![1]).unwrap();
    tree.insert(20, vec![2]).unwrap();
    tree.insert(30, vec![3]).unwrap();

    // Range with exact boundaries
    let results = tree.range(10, 30).unwrap();
    assert_eq!(results.len(), 3);

    // Range with one key
    let results = tree.range(20, 20).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 20);

    // Range before all keys
    let results = tree.range(0, 5).unwrap();
    assert_eq!(results.len(), 0);

    // Range after all keys
    let results = tree.range(40, 50).unwrap();
    assert_eq!(results.len(), 0);

    // Range partially overlapping
    let results = tree.range(15, 25).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 20);
}

#[test]
fn test_range_query_large_result_set() {
    let mut tree = AlexTree::new();

    // Insert 10K keys
    for i in 0..10000 {
        tree.insert(i, vec![(i % 256) as u8]).unwrap();
    }

    // Large range query
    let results = tree.range(1000, 5000).unwrap();
    assert_eq!(results.len(), 4001); // Inclusive range

    // Verify keys are sorted
    for i in 0..results.len() - 1 {
        assert!(results[i].0 < results[i + 1].0);
    }

    // Verify first and last keys
    assert_eq!(results[0].0, 1000);
    assert_eq!(results[results.len() - 1].0, 5000);
}

// Concurrent access tests

#[test]
fn test_concurrent_reads() {
    let mut tree = AlexTree::new();

    // Populate tree
    for i in 0..1000 {
        tree.insert(i, vec![(i % 256) as u8]).unwrap();
    }

    let tree_arc = Arc::new(tree);
    let barrier = Arc::new(Barrier::new(4));

    // Spawn 4 reader threads
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let tree_clone = Arc::clone(&tree_arc);
            let barrier_clone = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier_clone.wait();

                // Each thread reads different keys
                let start = thread_id * 250;
                let end = start + 250;

                for i in start..end {
                    let value = tree_clone.get(i).unwrap();
                    assert!(value.is_some(), "Thread {} missing key {}", thread_id, i);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_range_queries() {
    let mut tree = AlexTree::new();

    // Populate tree
    for i in 0..1000 {
        tree.insert(i, vec![(i % 256) as u8]).unwrap();
    }

    let tree_arc = Arc::new(tree);
    let barrier = Arc::new(Barrier::new(4));

    // Spawn 4 threads doing range queries
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let tree_clone = Arc::clone(&tree_arc);
            let barrier_clone = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier_clone.wait();

                // Each thread does range query over different range
                let start = thread_id * 200;
                let end = start + 200;

                let results = tree_clone.range(start, end).unwrap();
                assert_eq!(
                    results.len(),
                    201,
                    "Thread {} got wrong range size",
                    thread_id
                );

                // Verify sorted
                for i in 0..results.len() - 1 {
                    assert!(results[i].0 < results[i + 1].0);
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

// Model accuracy and error bound tests

#[test]
fn test_sequential_keys_low_error() {
    let mut tree = AlexTree::new();

    // Sequential keys should have very low model error
    for i in 0..1000 {
        tree.insert(i, vec![(i % 256) as u8]).unwrap();
    }

    // With good model, shouldn't need many splits
    // Sequential keys = perfect linear model = low error
    // Should use expansion factor efficiently
    assert!(
        tree.num_leaves() < 20,
        "Sequential keys should need few splits, got {} leaves",
        tree.num_leaves()
    );
}

#[test]
fn test_random_keys_higher_error() {
    let mut tree = AlexTree::with_expansion(0.0);

    // Random keys will have higher model error
    for i in 0..1000 {
        let key = (i * 7919 + 1234) % 100000; // Pseudo-random
        tree.insert(key, vec![(i % 256) as u8]).unwrap();
    }

    // Random keys = worse model = more splits
    // Still should be reasonable (not excessive splitting)
    assert!(
        tree.num_leaves() >= 10,
        "Random keys should cause more splits than sequential"
    );
    assert!(
        tree.num_leaves() < 100,
        "Too many splits - model may be over-fitting"
    );
}

#[test]
fn test_reverse_sorted_keys() {
    let mut tree = AlexTree::new();

    // Insert keys in reverse order (worst case for some data structures)
    for i in (0..1000).rev() {
        tree.insert(i, vec![(i % 256) as u8]).unwrap();
    }

    assert_eq!(tree.len(), 1000);

    // Should still handle reverse insertion efficiently
    // Model should adapt

    // Verify all keys present
    for i in (0..1000).step_by(50) {
        assert!(tree.get(i).unwrap().is_some());
    }

    // Range query should work (returns sorted output)
    let results = tree.range(0, 999).unwrap();
    assert_eq!(results.len(), 1000);

    // Verify ascending order despite reverse insertion
    for i in 0..results.len() - 1 {
        assert!(results[i].0 < results[i + 1].0);
    }
}
