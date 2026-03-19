//! Tests for EdgeStore and VectorStore edge operations.

use serde_json::json;
use tempfile::tempdir;

use crate::VectorStore;
use crate::vector::store::edge_store::{Edge, EdgeDirection, EdgeStore};

// --- EdgeStore unit tests ---

#[test]
fn test_edge_store_add_and_get() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "related".into(),
        weight: 1.0,
        metadata: None,
    });
    assert_eq!(store.edge_count(), 1);

    let out = store.get_edges("a", EdgeDirection::Outgoing, None);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].to_id, "b");
    assert_eq!(out[0].edge_type, "related");

    let inc = store.get_edges("b", EdgeDirection::Incoming, None);
    assert_eq!(inc.len(), 1);
    assert_eq!(inc[0].from_id, "a");
}

#[test]
fn test_edge_store_replace_same_type() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 0.5,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 0.9,
        metadata: Some(json!({"key": "value"})),
    });
    assert_eq!(store.edge_count(), 1);
    let out = store.get_edges("a", EdgeDirection::Outgoing, None);
    assert_eq!(out[0].weight, 0.9);
    assert_eq!(out[0].metadata, Some(json!({"key": "value"})));
}

#[test]
fn test_edge_store_multiple_types() {
    let mut store = EdgeStore::new();
    for etype in ["parent", "sibling", "child"] {
        store.add_edge(Edge {
            from_id: "x".into(),
            to_id: "y".into(),
            edge_type: etype.into(),
            weight: 1.0,
            metadata: None,
        });
    }
    assert_eq!(store.edge_count(), 3);
    assert_eq!(store.get_edges("x", EdgeDirection::Outgoing, None).len(), 3);
}

#[test]
fn test_edge_store_remove_edge() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    let removed = store.remove_edge("a", "b", "link");
    assert!(removed);
    assert_eq!(store.edge_count(), 0);
    assert!(
        store
            .get_edges("a", EdgeDirection::Outgoing, None)
            .is_empty()
    );
    assert!(
        store
            .get_edges("b", EdgeDirection::Incoming, None)
            .is_empty()
    );

    let not_removed = store.remove_edge("a", "b", "link");
    assert!(!not_removed);
}

#[test]
fn test_edge_store_remove_all_for() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "c".into(),
        to_id: "a".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });
    assert_eq!(store.edge_count(), 2);

    store.remove_all_for("a");
    assert_eq!(store.edge_count(), 0);
    assert!(
        store
            .get_edges("b", EdgeDirection::Incoming, None)
            .is_empty()
    );
    assert!(
        store
            .get_edges("c", EdgeDirection::Outgoing, None)
            .is_empty()
    );
}

#[test]
fn test_edge_store_traverse_depth1() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "root".into(),
        to_id: "child".into(),
        edge_type: "has".into(),
        weight: 1.0,
        metadata: None,
    });
    let reachable = store.traverse("root", EdgeDirection::Outgoing, 1, None);
    assert_eq!(reachable, vec!["child"]);
}

#[test]
fn test_edge_store_traverse_depth2() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "next".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "b".into(),
        to_id: "c".into(),
        edge_type: "next".into(),
        weight: 1.0,
        metadata: None,
    });
    let reachable = store.traverse("a", EdgeDirection::Outgoing, 2, None);
    assert_eq!(reachable.len(), 2);
    assert!(reachable.contains(&"b".to_string()));
    assert!(reachable.contains(&"c".to_string()));

    // depth 1 should only reach "b"
    let depth1 = store.traverse("a", EdgeDirection::Outgoing, 1, None);
    assert_eq!(depth1, vec!["b"]);
}

#[test]
fn test_edge_store_traverse_cycle_safety() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "e".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "b".into(),
        to_id: "a".into(),
        edge_type: "e".into(),
        weight: 1.0,
        metadata: None,
    });
    let reachable = store.traverse("a", EdgeDirection::Outgoing, 10, None);
    assert_eq!(reachable, vec!["b"]);
}

#[test]
fn test_edge_store_traverse_edge_type_filter() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "c".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });
    let links = store.traverse("a", EdgeDirection::Outgoing, 1, Some("link"));
    assert_eq!(links, vec!["b"]);

    let refs = store.traverse("a", EdgeDirection::Outgoing, 1, Some("ref"));
    assert_eq!(refs, vec!["c"]);
}

#[test]
fn test_edge_store_serialize_roundtrip() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "x".into(),
        to_id: "y".into(),
        edge_type: "typed".into(),
        weight: 0.75,
        metadata: Some(json!({"score": 42})),
    });
    store.add_edge(Edge {
        from_id: "y".into(),
        to_id: "z".into(),
        edge_type: "typed".into(),
        weight: 0.5,
        metadata: None,
    });

    let bytes = store.to_bytes().unwrap();
    let restored = EdgeStore::from_bytes(&bytes).unwrap();

    assert_eq!(restored.edge_count(), 2);
    let xy = restored.get_edges("x", EdgeDirection::Outgoing, None);
    assert_eq!(xy.len(), 1);
    assert_eq!(xy[0].weight, 0.75);
    assert_eq!(xy[0].metadata, Some(json!({"score": 42})));
}

#[test]
fn test_edge_store_self_edge_no_duplicate() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "a".into(),
        edge_type: "self_ref".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    assert_eq!(store.edge_count(), 2);

    // get_edges(Both) must not duplicate the self-edge.
    // Outgoing: a→a, a→b. Incoming self-loop filtered as duplicate.
    let both = store.get_edges("a", EdgeDirection::Both, None);
    assert_eq!(both.len(), 2);

    // edges_involving must not duplicate the self-edge for WAL deletes.
    let involving = store.edges_involving("a");
    assert_eq!(involving.len(), 2);
    assert!(involving.contains(&("a".into(), "a".into(), "self_ref".into())));
    assert!(involving.contains(&("a".into(), "b".into(), "link".into())));
}

#[test]
fn test_edge_store_gc_orphaned() {
    use rustc_hash::FxHashSet;
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "live".into(),
        to_id: "dead".into(),
        edge_type: "e".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "live".into(),
        to_id: "also_live".into(),
        edge_type: "e".into(),
        weight: 1.0,
        metadata: None,
    });

    let live: FxHashSet<String> = ["live", "also_live"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let removed = store.gc_orphaned(&live);
    assert_eq!(removed, 1);
    assert_eq!(store.edge_count(), 1);
}

// --- VectorStore integration tests ---

#[test]
fn test_vector_store_add_and_get_edges() {
    let mut store = VectorStore::new(4);
    store
        .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
        .unwrap();
    store
        .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
        .unwrap();

    store.add_edge("a", "b", "link", 0.9, None).unwrap();
    assert_eq!(store.edge_count(), 1);

    let edges = store.get_edges("a", EdgeDirection::Outgoing, None);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].to_id, "b");
}

#[test]
fn test_vector_store_delete_cascades_edges() {
    let mut store = VectorStore::new(4);
    store
        .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
        .unwrap();
    store
        .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
        .unwrap();
    store.add_edge("a", "b", "link", 1.0, None).unwrap();

    store.delete("a").unwrap();
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_vector_store_compact_gcs_orphaned_edges() {
    let mut store = VectorStore::new(4);
    store
        .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
        .unwrap();
    store
        .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
        .unwrap();
    store.add_edge("a", "b", "link", 1.0, None).unwrap();

    // Delete b without going through the cascade (simulate orphan)
    store.records.delete("b");
    // Edge still exists (orphaned), compact should GC it
    store.compact().unwrap();
    assert_eq!(store.edge_count(), 0);
}

#[test]
fn test_vector_store_traverse() {
    let mut store = VectorStore::new(4);
    for id in ["a", "b", "c"] {
        store
            .set(id, crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
    }
    store.add_edge("a", "b", "next", 1.0, None).unwrap();
    store.add_edge("b", "c", "next", 1.0, None).unwrap();

    let reachable = store.traverse("a", EdgeDirection::Outgoing, 2, None);
    assert_eq!(reachable.len(), 2);
    assert!(reachable.contains(&"b".to_string()));
    assert!(reachable.contains(&"c".to_string()));
}

#[test]
fn test_vector_store_flush_and_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store
            .add_edge("a", "b", "link", 0.8, Some(json!({"w": 1})))
            .unwrap();
        store.flush().unwrap();
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.edge_count(), 1);
        let edges = store.get_edges("a", EdgeDirection::Outgoing, None);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_id, "b");
        assert_eq!(edges[0].weight, 0.8);
        assert_eq!(edges[0].metadata, Some(json!({"w": 1})));
    }
}

#[test]
fn test_vector_store_wal_recovery_without_flush() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store.add_edge("a", "b", "link", 0.5, None).unwrap();
        // No flush — edge is in WAL only
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.edge_count(), 1);
        let edges = store.get_edges("a", EdgeDirection::Outgoing, None);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].to_id, "b");
    }
}

#[test]
fn test_vector_store_remove_edge() {
    let mut store = VectorStore::new(4);
    store
        .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
        .unwrap();
    store
        .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
        .unwrap();
    store.add_edge("a", "b", "link", 1.0, None).unwrap();

    let removed = store.remove_edge("a", "b", "link").unwrap();
    assert!(removed);
    assert_eq!(store.edge_count(), 0);

    let not_removed = store.remove_edge("a", "b", "link").unwrap();
    assert!(!not_removed);
}

#[test]
fn test_vector_store_expand() {
    let mut store = VectorStore::new(4);
    for id in ["a", "b", "c", "d"] {
        store
            .set(id, crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
    }
    store.add_edge("a", "c", "rel", 1.0, None).unwrap();
    store.add_edge("b", "d", "rel", 1.0, None).unwrap();

    let seed = vec!["a".to_string(), "b".to_string()];
    let expanded = store.expand(&seed, EdgeDirection::Outgoing, None);
    assert_eq!(expanded.len(), 4);
}

#[test]
fn test_vector_store_cascade_delete_wal_recovery() {
    // Verifies that cascade-deleted edges are WAL-logged and not resurrected on recovery.
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store.add_edge("a", "b", "link", 1.0, None).unwrap();
        store.flush().unwrap();
        // Edge is now in the manifest snapshot.
        // Delete "a" — should write WAL DeleteEdge entries for the cascade.
        store.delete("a").unwrap();
        // No second flush — edge removal is WAL-only.
    }

    {
        // Recovery: manifest has edge "a→b", WAL has DeleteNode("a") + DeleteEdge("a","b","link").
        // After replay the edge must be gone.
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(
            store.edge_count(),
            0,
            "cascade-deleted edge survived WAL recovery"
        );
        assert!(
            store
                .get_edges("b", EdgeDirection::Incoming, None)
                .is_empty(),
            "stale incoming edge for deleted node 'a'"
        );
    }
}

// --- Batch 1: Critical WAL edge case tests ---

#[test]
fn test_mixed_wal_add_delete_readd_edge() {
    // Add → flush (manifest) → delete edge → add same edge back → crash → reopen.
    // Verifies WAL replay handles add-after-delete for the same edge key.
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store.add_edge("a", "b", "link", 0.5, None).unwrap();
        store.flush().unwrap();

        // Edge is now in manifest. Delete it (WAL only).
        store.remove_edge("a", "b", "link").unwrap();
        assert_eq!(store.edge_count(), 0);

        // Re-add the same edge with different weight (WAL only).
        store
            .add_edge("a", "b", "link", 0.9, Some(json!({"readded": true})))
            .unwrap();
        assert_eq!(store.edge_count(), 1);
        // Crash — no second flush.
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(
            store.edge_count(),
            1,
            "re-added edge should survive WAL replay"
        );
        let edges = store.get_edges("a", EdgeDirection::Outgoing, None);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 0.9);
        assert_eq!(edges[0].metadata, Some(json!({"readded": true})));
    }
}

#[test]
fn test_delete_batch_cascade_wal_recovery() {
    // Same pattern as test_vector_store_cascade_delete_wal_recovery but using
    // delete_batch(). The batch path writes all edge deletes then all node deletes
    // in a single WAL sync.
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store
            .set("c", crate::Vector::new(vec![3.0; 4]), json!({}))
            .unwrap();

        // a→b, b→c, c→a (cycle)
        store.add_edge("a", "b", "link", 1.0, None).unwrap();
        store.add_edge("b", "c", "link", 1.0, None).unwrap();
        store.add_edge("c", "a", "link", 1.0, None).unwrap();
        store.flush().unwrap();
        assert_eq!(store.edge_count(), 3);

        // Batch-delete a and b — cascade should remove all 3 edges.
        let deleted = store.delete_batch(&["a", "b"]).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(store.edge_count(), 0);
        // No flush — edge removal is WAL-only.
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.len(), 1, "only 'c' should remain");
        assert!(store.contains("c"));
        assert_eq!(
            store.edge_count(),
            0,
            "batch cascade-deleted edges survived WAL recovery"
        );
        assert!(
            store
                .get_edges("c", EdgeDirection::Outgoing, None)
                .is_empty()
        );
        assert!(
            store
                .get_edges("c", EdgeDirection::Incoming, None)
                .is_empty()
        );
    }
}

#[test]
fn test_interleaved_edge_vector_wal_operations() {
    // Insert vectors + edges, flush, then interleave: add vectors, add edges,
    // delete vectors (cascade), delete edges explicitly. Crash. Verify all replayed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();

        // Initial state: 4 vectors, 3 edges
        for id in ["v1", "v2", "v3", "v4"] {
            store
                .set(id, crate::Vector::new(vec![1.0; 4]), json!({}))
                .unwrap();
        }
        store.add_edge("v1", "v2", "link", 1.0, None).unwrap();
        store.add_edge("v2", "v3", "link", 1.0, None).unwrap();
        store.add_edge("v3", "v4", "link", 1.0, None).unwrap();
        store.flush().unwrap();
        assert_eq!(store.len(), 4);
        assert_eq!(store.edge_count(), 3);

        // WAL-only operations:
        // 1. Add new vectors
        store
            .set("v5", crate::Vector::new(vec![5.0; 4]), json!({}))
            .unwrap();
        store
            .set("v6", crate::Vector::new(vec![6.0; 4]), json!({}))
            .unwrap();

        // 2. Add new edges
        store.add_edge("v4", "v5", "link", 1.0, None).unwrap();
        store.add_edge("v5", "v6", "next", 0.5, None).unwrap();

        // 3. Delete v1 (cascade removes v1→v2 edge)
        store.delete("v1").unwrap();

        // 4. Explicitly remove v2→v3 edge
        store.remove_edge("v2", "v3", "link").unwrap();

        // Expected: 5 vectors (v2-v6), 3 edges (v3→v4, v4→v5, v5→v6)
        assert_eq!(store.len(), 5);
        assert_eq!(store.edge_count(), 3);
        // Crash
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.len(), 5, "5 vectors should survive");
        assert!(!store.contains("v1"), "v1 was deleted");
        assert!(store.contains("v5"), "v5 was added");
        assert!(store.contains("v6"), "v6 was added");

        assert_eq!(store.edge_count(), 3, "3 edges should survive WAL replay");
        // v1→v2 was cascade-deleted
        assert!(
            store
                .get_edges("v1", EdgeDirection::Outgoing, None)
                .is_empty()
        );
        // v2→v3 was explicitly removed
        let v2_out = store.get_edges("v2", EdgeDirection::Outgoing, None);
        assert!(v2_out.is_empty(), "v2→v3 edge was explicitly removed");
        // v3→v4 should still exist
        let v3_out = store.get_edges("v3", EdgeDirection::Outgoing, None);
        assert_eq!(v3_out.len(), 1);
        assert_eq!(v3_out[0].to_id, "v4");
        // v4→v5 should exist
        let v4_out = store.get_edges("v4", EdgeDirection::Outgoing, None);
        assert_eq!(v4_out.len(), 1);
        assert_eq!(v4_out[0].to_id, "v5");
        // v5→v6 should exist
        let v5_out = store.get_edges("v5", EdgeDirection::Outgoing, None);
        assert_eq!(v5_out.len(), 1);
        assert_eq!(v5_out[0].to_id, "v6");
        assert_eq!(v5_out[0].edge_type, "next");
    }
}

#[test]
fn test_edge_wal_across_checkpoint_boundary() {
    // Insert >10K vectors via set() to trigger auto-checkpoint, add edges before
    // and after the checkpoint. Crash. Verify edges survive.
    // Auto-checkpoint fires in set() at WAL_AUTO_CHECKPOINT_ENTRIES = 10,000.
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open_with_dimensions(&path, 4).unwrap();

        // Insert first two vectors and an edge before checkpoint
        store
            .set("pre_a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("pre_b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store
            .add_edge(
                "pre_a",
                "pre_b",
                "before_ckpt",
                0.7,
                Some(json!({"phase": "pre"})),
            )
            .unwrap();

        // Insert 10K+ vectors via set() to trigger auto-checkpoint.
        // WAL entries: 2 (pre vectors) + 1 (edge) + 10_000 = 10_003 total.
        for i in 0..10_000 {
            store
                .set(
                    &format!("pad_{i}"),
                    crate::Vector::new(vec![i as f32; 4]),
                    json!({}),
                )
                .unwrap();
        }

        // After auto-checkpoint, add more vectors and edges (WAL-only)
        store
            .set("post_a", crate::Vector::new(vec![3.0; 4]), json!({}))
            .unwrap();
        store
            .set("post_b", crate::Vector::new(vec![4.0; 4]), json!({}))
            .unwrap();
        store
            .add_edge(
                "post_a",
                "post_b",
                "after_ckpt",
                0.3,
                Some(json!({"phase": "post"})),
            )
            .unwrap();

        assert_eq!(store.edge_count(), 2);
        // Crash — no flush
    }

    {
        let store = VectorStore::open(&path).unwrap();
        // All vectors should survive (10K pad + 2 pre + 2 post)
        assert_eq!(store.len(), 10_004);

        assert_eq!(
            store.edge_count(),
            2,
            "both edges should survive checkpoint boundary"
        );

        let pre_edges = store.get_edges("pre_a", EdgeDirection::Outgoing, None);
        assert_eq!(pre_edges.len(), 1);
        assert_eq!(pre_edges[0].to_id, "pre_b");
        assert_eq!(pre_edges[0].edge_type, "before_ckpt");
        assert_eq!(pre_edges[0].weight, 0.7);
        assert_eq!(pre_edges[0].metadata, Some(json!({"phase": "pre"})));

        let post_edges = store.get_edges("post_a", EdgeDirection::Outgoing, None);
        assert_eq!(post_edges.len(), 1);
        assert_eq!(post_edges[0].to_id, "post_b");
        assert_eq!(post_edges[0].edge_type, "after_ckpt");
        assert_eq!(post_edges[0].weight, 0.3);
        assert_eq!(post_edges[0].metadata, Some(json!({"phase": "post"})));
    }
}

// --- New graph primitives tests ---

fn make_chain_store() -> EdgeStore {
    let mut store = EdgeStore::new();
    // a -> b -> c -> d
    for (from, to) in [("a", "b"), ("b", "c"), ("c", "d")] {
        store.add_edge(Edge {
            from_id: from.into(),
            to_id: to.into(),
            edge_type: "next".into(),
            weight: 1.0,
            metadata: None,
        });
    }
    store
}

#[test]
fn test_get_edge() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 0.5,
        metadata: Some(json!({"k": 1})),
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "ref".into(),
        weight: 0.9,
        metadata: None,
    });

    let found = store.get_edge("a", "b", "link").unwrap();
    assert_eq!(found.weight, 0.5);
    assert_eq!(found.metadata, Some(json!({"k": 1})));

    let found_ref = store.get_edge("a", "b", "ref").unwrap();
    assert_eq!(found_ref.weight, 0.9);

    assert!(store.get_edge("a", "b", "nonexistent").is_none());
    assert!(store.get_edge("b", "a", "link").is_none());
    assert!(store.get_edge("x", "y", "link").is_none());
}

#[test]
fn test_neighbors() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "c".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "d".into(),
        to_id: "a".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });

    let out = store.neighbors("a", EdgeDirection::Outgoing, None);
    assert_eq!(out.len(), 2);
    assert!(out.contains(&"b".to_string()));
    assert!(out.contains(&"c".to_string()));

    let inc = store.neighbors("a", EdgeDirection::Incoming, None);
    assert_eq!(inc, vec!["d"]);

    let both = store.neighbors("a", EdgeDirection::Both, None);
    assert_eq!(both.len(), 3);

    let filtered = store.neighbors("a", EdgeDirection::Outgoing, Some("link"));
    assert_eq!(filtered, vec!["b"]);
}

#[test]
fn test_node_degree() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "c".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "d".into(),
        to_id: "a".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });

    assert_eq!(store.node_degree("a", EdgeDirection::Outgoing, None), 2);
    assert_eq!(store.node_degree("a", EdgeDirection::Incoming, None), 1);
    assert_eq!(store.node_degree("a", EdgeDirection::Both, None), 3);
    assert_eq!(
        store.node_degree("a", EdgeDirection::Outgoing, Some("link")),
        1
    );
    assert_eq!(store.node_degree("x", EdgeDirection::Both, None), 0);
}

#[test]
fn test_node_degree_self_loop() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "a".into(),
        edge_type: "self".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    // Both should not double-count the self-loop.
    // get_edges(Both) returns: outgoing(a→a, a→b) + incoming filtered(b→a excluded as self-loop already counted).
    // Actually incoming for "a": self(a→a), link from somewhere — no. incoming["a"] has peer_id="a" (self-loop).
    // So degree(Both) = outgoing(2) + incoming(1 self-loop) - self_loop_overlap(1) = 2
    assert_eq!(store.node_degree("a", EdgeDirection::Both, None), 2);
}

#[test]
fn test_has_path_direct() {
    let store = make_chain_store();
    assert!(store.has_path("a", "b", EdgeDirection::Outgoing, 1, None));
}

#[test]
fn test_has_path_indirect() {
    let store = make_chain_store();
    assert!(store.has_path("a", "d", EdgeDirection::Outgoing, 10, None));
}

#[test]
fn test_has_path_no_path() {
    let store = make_chain_store();
    assert!(!store.has_path("d", "a", EdgeDirection::Outgoing, 10, None));
}

#[test]
fn test_has_path_max_depth() {
    let store = make_chain_store();
    // a->b->c->d requires depth 3, depth 2 should fail
    assert!(!store.has_path("a", "d", EdgeDirection::Outgoing, 2, None));
    assert!(store.has_path("a", "d", EdgeDirection::Outgoing, 3, None));
}

#[test]
fn test_has_path_same_node() {
    let store = make_chain_store();
    assert!(store.has_path("a", "a", EdgeDirection::Outgoing, 0, None));
}

#[test]
fn test_shortest_path_direct() {
    let store = make_chain_store();
    let path = store
        .shortest_path("a", "b", EdgeDirection::Outgoing, 10, None)
        .unwrap();
    assert_eq!(path, vec!["a", "b"]);
}

#[test]
fn test_shortest_path_multi_hop() {
    let store = make_chain_store();
    let path = store
        .shortest_path("a", "d", EdgeDirection::Outgoing, 10, None)
        .unwrap();
    assert_eq!(path, vec!["a", "b", "c", "d"]);
}

#[test]
fn test_shortest_path_no_path() {
    let store = make_chain_store();
    assert!(
        store
            .shortest_path("d", "a", EdgeDirection::Outgoing, 10, None)
            .is_none()
    );
}

#[test]
fn test_shortest_path_same_node() {
    let store = make_chain_store();
    let path = store
        .shortest_path("a", "a", EdgeDirection::Outgoing, 10, None)
        .unwrap();
    assert_eq!(path, vec!["a"]);
}

#[test]
fn test_shortest_path_picks_shortest() {
    let mut store = EdgeStore::new();
    // Direct: a -> c
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "c".into(),
        edge_type: "direct".into(),
        weight: 1.0,
        metadata: None,
    });
    // Long: a -> b -> c
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "b".into(),
        to_id: "c".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    let path = store
        .shortest_path("a", "c", EdgeDirection::Outgoing, 10, None)
        .unwrap();
    assert_eq!(path.len(), 2); // a -> c (direct)
}

#[test]
fn test_traverse_edges() {
    let store = make_chain_store();
    let hits = store.traverse_edges("a", EdgeDirection::Outgoing, 2, None);
    assert_eq!(hits.len(), 2);

    let hit_b = hits.iter().find(|h| h.id == "b").unwrap();
    assert_eq!(hit_b.depth, 1);
    assert_eq!(hit_b.edge.from_id, "a");
    assert_eq!(hit_b.edge.to_id, "b");

    let hit_c = hits.iter().find(|h| h.id == "c").unwrap();
    assert_eq!(hit_c.depth, 2);
    assert_eq!(hit_c.edge.from_id, "b");
    assert_eq!(hit_c.edge.to_id, "c");
}

#[test]
fn test_subgraph() {
    let mut store = EdgeStore::new();
    // Triangle: a->b, b->c, c->a, plus d->a (external)
    for (f, t) in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")] {
        store.add_edge(Edge {
            from_id: f.into(),
            to_id: t.into(),
            edge_type: "link".into(),
            weight: 1.0,
            metadata: None,
        });
    }

    let sg = store.subgraph("a", 2, EdgeDirection::Outgoing, None);
    assert!(sg.node_ids.contains(&"a".to_string()));
    assert!(sg.node_ids.contains(&"b".to_string()));
    assert!(sg.node_ids.contains(&"c".to_string()));
    // d is not reachable via outgoing from a
    assert!(!sg.node_ids.contains(&"d".to_string()));
    // All 3 internal edges (a->b, b->c, c->a) should be included
    assert_eq!(sg.edges.len(), 3);
    // d->a should NOT be included (d not in subgraph)
    assert!(!sg.edges.iter().any(|e| e.from_id == "d"));
}

#[test]
fn test_add_edges_batch() {
    let mut store = EdgeStore::new();
    let edges = vec![
        Edge {
            from_id: "a".into(),
            to_id: "b".into(),
            edge_type: "link".into(),
            weight: 1.0,
            metadata: None,
        },
        Edge {
            from_id: "b".into(),
            to_id: "c".into(),
            edge_type: "link".into(),
            weight: 1.0,
            metadata: None,
        },
    ];
    let added = store.add_edges(edges);
    assert_eq!(added, 2);
    assert_eq!(store.edge_count(), 2);

    // Adding an existing edge should not count as new
    let added2 = store.add_edges(vec![Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 0.5,
        metadata: None,
    }]);
    assert_eq!(added2, 0);
    assert_eq!(store.edge_count(), 2);
}

#[test]
fn test_add_edges_wal_recovery() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.omen");

    {
        let mut store = VectorStore::open(&path).unwrap();
        store
            .set("a", crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
        store
            .set("b", crate::Vector::new(vec![2.0; 4]), json!({}))
            .unwrap();
        store
            .set("c", crate::Vector::new(vec![3.0; 4]), json!({}))
            .unwrap();

        let edges = vec![
            crate::Edge {
                from_id: "a".into(),
                to_id: "b".into(),
                edge_type: "link".into(),
                weight: 0.5,
                metadata: Some(json!({"batch": true})),
            },
            crate::Edge {
                from_id: "b".into(),
                to_id: "c".into(),
                edge_type: "link".into(),
                weight: 0.7,
                metadata: None,
            },
        ];
        let added = store.add_edges(edges).unwrap();
        assert_eq!(added, 2);
        // No flush — WAL only
    }

    {
        let store = VectorStore::open(&path).unwrap();
        assert_eq!(store.edge_count(), 2);
        let ab = store.get_edge("a", "b", "link").unwrap();
        assert_eq!(ab.weight, 0.5);
        assert_eq!(ab.metadata, Some(json!({"batch": true})));
        let bc = store.get_edge("b", "c", "link").unwrap();
        assert_eq!(bc.weight, 0.7);
    }
}

#[test]
fn test_edge_types() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "c".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "b".into(),
        to_id: "c".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });

    let mut types = store.edge_types();
    types.sort();
    assert_eq!(types, vec!["link", "ref"]);
}

#[test]
fn test_node_ids() {
    let mut store = EdgeStore::new();
    store.add_edge(Edge {
        from_id: "a".into(),
        to_id: "b".into(),
        edge_type: "link".into(),
        weight: 1.0,
        metadata: None,
    });
    store.add_edge(Edge {
        from_id: "c".into(),
        to_id: "a".into(),
        edge_type: "ref".into(),
        weight: 1.0,
        metadata: None,
    });

    let mut ids = store.node_ids();
    ids.sort();
    assert_eq!(ids, vec!["a", "b", "c"]);
}
