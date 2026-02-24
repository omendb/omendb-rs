//! Tests for EdgeStore and VectorStore edge operations.

use serde_json::json;
use tempfile::tempdir;

use crate::vector::store::edge_store::{Edge, EdgeDirection, EdgeStore};
use crate::VectorStore;

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

    let out = store.get_edges("a", EdgeDirection::Outgoing);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].to_id, "b");
    assert_eq!(out[0].edge_type, "related");

    let inc = store.get_edges("b", EdgeDirection::Incoming);
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
    let out = store.get_edges("a", EdgeDirection::Outgoing);
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
    assert_eq!(store.get_edges("x", EdgeDirection::Outgoing).len(), 3);
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
    assert!(store.get_edges("a", EdgeDirection::Outgoing).is_empty());
    assert!(store.get_edges("b", EdgeDirection::Incoming).is_empty());

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
    assert!(store.get_edges("b", EdgeDirection::Incoming).is_empty());
    assert!(store.get_edges("c", EdgeDirection::Outgoing).is_empty());
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
    let xy = restored.get_edges("x", EdgeDirection::Outgoing);
    assert_eq!(xy.len(), 1);
    assert_eq!(xy[0].weight, 0.75);
    assert_eq!(xy[0].metadata, Some(json!({"score": 42})));
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

    let edges = store.get_edges("a", EdgeDirection::Outgoing);
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
        let edges = store.get_edges("a", EdgeDirection::Outgoing);
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
        let edges = store.get_edges("a", EdgeDirection::Outgoing);
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
fn test_vector_store_expand_via_edges() {
    let mut store = VectorStore::new(4);
    for id in ["a", "b", "c", "d"] {
        store
            .set(id, crate::Vector::new(vec![1.0; 4]), json!({}))
            .unwrap();
    }
    store.add_edge("a", "c", "rel", 1.0, None).unwrap();
    store.add_edge("b", "d", "rel", 1.0, None).unwrap();

    let seed = vec!["a".to_string(), "b".to_string()];
    let expanded = store.expand_via_edges(&seed, EdgeDirection::Outgoing, None);
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
            store.get_edges("b", EdgeDirection::Incoming).is_empty(),
            "stale incoming edge for deleted node 'a'"
        );
    }
}
