"""Tests for EdgeStore Python bindings."""

import os
import tempfile

import pytest

import omendb
from tests.helpers import create_dense_db


def make_db(tmpdir, dims=4):
    path = os.path.join(tmpdir, "edges_db")
    db = create_dense_db(path, dims)
    for doc_id in ["a", "b", "c", "d"]:
        db.set([{"id": doc_id, "vector": [1.0] * dims}])
    return db


def test_add_edge_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        assert db.edge_count() == 1

        out = db.get_edges("a", "outgoing")
        assert len(out) == 1
        assert out[0]["from_id"] == "a"
        assert out[0]["to_id"] == "b"
        assert out[0]["edge_type"] == "link"
        assert out[0]["weight"] == 1.0

        inc = db.get_edges("b", "incoming")
        assert len(inc) == 1
        assert inc[0]["from_id"] == "a"


def test_add_edge_with_weight_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "related", weight=0.75, metadata={"score": 42})
        edges = db.get_edges("a", "outgoing")
        assert edges[0]["weight"] == 0.75
        assert edges[0]["metadata"] == {"score": 42}


def test_add_edge_replaces_same_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link", weight=0.5)
        db.add_edge("a", "b", "link", weight=0.9, metadata={"key": "val"})
        assert db.edge_count() == 1
        edges = db.get_edges("a", "outgoing")
        assert edges[0]["weight"] == pytest.approx(0.9, rel=1e-5)
        assert edges[0]["metadata"] == {"key": "val"}


def test_remove_edge():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        removed = db.remove_edge("a", "b", "link")
        assert removed is True
        assert db.edge_count() == 0
        not_removed = db.remove_edge("a", "b", "link")
        assert not_removed is False


def test_get_edges_both_directions():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("c", "a", "ref")
        both = db.get_edges("a", "both")
        assert len(both) == 2


def test_traverse():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "next")
        db.add_edge("b", "c", "next")
        reachable = db.traverse("a", "outgoing", max_depth=2)
        assert set(reachable) == {"b", "c"}

        depth1 = db.traverse("a", "outgoing", max_depth=1)
        assert depth1 == ["b"]


def test_traverse_with_edge_type_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("a", "c", "ref")
        links = db.traverse("a", "outgoing", max_depth=1, edge_type="link")
        assert links == ["b"]
        refs = db.traverse("a", "outgoing", max_depth=1, edge_type="ref")
        assert refs == ["c"]


def test_expand():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "c", "rel")
        db.add_edge("b", "d", "rel")
        expanded = db.expand(["a", "b"], "outgoing")
        assert set(expanded) == {"a", "b", "c", "d"}


def test_delete_cascades_edges():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.delete("a")
        assert db.edge_count() == 0


def test_persistence_flush_and_reopen():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edges_db")

        db = create_dense_db(path, 4)
        for doc_id in ["a", "b"]:
            db.set([{"id": doc_id, "vector": [1.0] * 4}])
        db.add_edge("a", "b", "link", weight=0.8, metadata={"w": 1})
        db.flush()
        del db

        db2 = omendb.open(path)
        assert db2.edge_count() == 1
        edges = db2.get_edges("a", "outgoing")
        assert edges[0]["to_id"] == "b"
        assert edges[0]["weight"] == pytest.approx(0.8, rel=1e-5)
        assert edges[0]["metadata"] == {"w": 1}


def test_invalid_direction_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        try:
            db.get_edges("a", "sideways")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_get_edge():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link", weight=0.5, metadata={"k": 1})
        edge = db.get_edge("a", "b", "link")
        assert edge is not None
        assert edge["weight"] == pytest.approx(0.5, rel=1e-5)
        assert edge["metadata"] == {"k": 1}
        assert db.get_edge("a", "b", "nonexistent") is None
        assert db.get_edge("b", "a", "link") is None


def test_neighbors():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("a", "c", "ref")
        db.add_edge("d", "a", "link")
        out = db.neighbors("a", "outgoing")
        assert set(out) == {"b", "c"}
        inc = db.neighbors("a", "incoming")
        assert inc == ["d"]
        filtered = db.neighbors("a", "outgoing", edge_type="link")
        assert filtered == ["b"]


def test_node_degree():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("a", "c", "ref")
        db.add_edge("d", "a", "link")
        assert db.node_degree("a", "outgoing") == 2
        assert db.node_degree("a", "incoming") == 1
        assert db.node_degree("a", "both") == 3
        assert db.node_degree("a", "outgoing", edge_type="link") == 1


def test_has_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "next")
        db.add_edge("b", "c", "next")
        assert db.has_path("a", "c") is True
        assert db.has_path("c", "a") is False
        assert db.has_path("a", "c", max_depth=1) is False
        assert db.has_path("a", "a") is True


def test_shortest_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "next")
        db.add_edge("b", "c", "next")
        db.add_edge("c", "d", "next")
        path = db.shortest_path("a", "d")
        assert path == ["a", "b", "c", "d"]
        assert db.shortest_path("d", "a") is None
        assert db.shortest_path("a", "a") == ["a"]


def test_traverse_edges():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "next")
        db.add_edge("b", "c", "next")
        hits = db.traverse_edges("a", "outgoing", max_depth=2)
        assert len(hits) == 2
        hit_b = next(h for h in hits if h["id"] == "b")
        assert hit_b["depth"] == 1
        assert hit_b["edge"]["from_id"] == "a"
        assert hit_b["edge"]["to_id"] == "b"


def test_subgraph():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("b", "c", "link")
        db.add_edge("d", "a", "link")
        sg = db.subgraph("a", max_depth=2, direction="outgoing")
        assert set(sg["node_ids"]) == {"a", "b", "c"}
        assert len(sg["edges"]) == 2  # a->b, b->c (d->a excluded, d not in subgraph)


def test_add_edges():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        added = db.add_edges(
            [
                {"from_id": "a", "to_id": "b", "edge_type": "link"},
                {"from_id": "b", "to_id": "c", "edge_type": "link", "weight": 0.5},
            ]
        )
        assert added == 2
        assert db.edge_count() == 2
        edge = db.get_edge("b", "c", "link")
        assert edge["weight"] == pytest.approx(0.5, rel=1e-5)


def test_edge_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("a", "c", "ref")
        types = sorted(db.edge_types())
        assert types == ["link", "ref"]


def test_node_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = make_db(tmpdir)
        db.add_edge("a", "b", "link")
        db.add_edge("c", "a", "ref")
        ids = sorted(db.node_ids())
        assert ids == ["a", "b", "c"]
