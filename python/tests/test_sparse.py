"""Tests for sparse vector operations (SPLADE-style)."""

import os
import tempfile

import omendb


def test_enable_sparse():
    """Test explicit enable and has_sparse check."""
    db = omendb.open(":memory:", dimensions=3)
    assert not db.has_sparse()
    db.enable_sparse()
    assert db.has_sparse()


def test_set_sparse_with_indices_values():
    """Test sparse insert with (indices, values) format."""
    db = omendb.open(":memory:", dimensions=0)
    db.set_sparse("doc1", [10, 42, 100], [0.5, 1.2, 0.8], {"title": "Hello"})
    assert db.has_sparse()


def test_set_sparse_with_dict():
    """Test sparse insert with {dim: weight} format."""
    db = omendb.open(":memory:", dimensions=0)
    db.set_sparse("doc1", {10: 0.5, 42: 1.2, 100: 0.8}, metadata={"title": "Hello"})
    assert db.has_sparse()


def test_sparse_search():
    """Test sparse search returns results in correct order."""
    db = omendb.open(":memory:", dimensions=0)

    # Insert docs with different sparse profiles
    db.set_sparse("doc1", {10: 1.0, 20: 0.5})
    db.set_sparse("doc2", {10: 0.5, 30: 1.0})
    db.set_sparse("doc3", {10: 0.1, 20: 0.1})

    # Query that strongly matches doc1
    results = db.sparse_search({10: 1.0, 20: 1.0}, k=3)

    assert len(results) == 3
    # doc1 should rank first (highest dot product with query)
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > results[1]["score"]


def test_set_hybrid_sparse():
    """Test combined dense + sparse insert."""
    db = omendb.open(":memory:", dimensions=3)
    db.set_hybrid_sparse(
        "doc1",
        [1.0, 0.0, 0.0],
        {10: 0.5, 42: 1.2},
        metadata={"title": "Hello"},
    )
    assert db.has_sparse()
    # Dense vector should be searchable
    results = db.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_hybrid_sparse_search():
    """Test hybrid dense+sparse search with RRF fusion."""
    db = omendb.open(":memory:", dimensions=3)

    db.set_hybrid_sparse("doc1", [1.0, 0.0, 0.0], {10: 1.0, 20: 0.5})
    db.set_hybrid_sparse("doc2", [0.0, 1.0, 0.0], {10: 0.5, 30: 1.0})
    db.set_hybrid_sparse("doc3", [0.0, 0.0, 1.0], {10: 0.1})

    results = db.hybrid_sparse_search(
        query_vector=[1.0, 0.0, 0.0],
        sparse_indices_or_dict={10: 1.0, 20: 1.0},
        k=3,
        alpha=0.5,
    )

    assert len(results) == 3
    # doc1 should rank first (best in both dense and sparse)
    assert results[0]["id"] == "doc1"


def test_sparse_filtered_search():
    """Test sparse search with metadata filter."""
    db = omendb.open(":memory:", dimensions=0)

    db.set_sparse("doc1", {10: 1.0, 20: 0.5}, metadata={"category": "A"})
    db.set_sparse("doc2", {10: 0.8, 20: 0.3}, metadata={"category": "B"})
    db.set_sparse("doc3", {10: 0.2}, metadata={"category": "A"})

    results = db.sparse_search(
        {10: 1.0, 20: 1.0},
        k=10,
        filter={"category": "A"},
    )

    # Only category A docs
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {"doc1", "doc3"}


def test_sparse_delete():
    """Test that deleted docs are excluded from sparse results."""
    db = omendb.open(":memory:", dimensions=0)

    db.set_sparse("doc1", {10: 1.0})
    db.set_sparse("doc2", {10: 0.5})

    results = db.sparse_search({10: 1.0}, k=10)
    assert len(results) == 2

    db.delete(["doc1"])

    results = db.sparse_search({10: 1.0}, k=10)
    assert len(results) == 1
    assert results[0]["id"] == "doc2"


def test_sparse_persistence():
    """Test that sparse data survives flush/reopen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sparse_test")

        # Create and populate
        db = omendb.open(path, dimensions=3)
        db.set_hybrid_sparse("doc1", [1.0, 0.0, 0.0], {10: 1.0, 20: 0.5})
        db.set_hybrid_sparse("doc2", [0.0, 1.0, 0.0], {10: 0.3})
        db.flush()
        db.close()

        # Reopen and verify sparse data
        db2 = omendb.open(path, dimensions=3)
        assert db2.has_sparse()

        results = db2.sparse_search({10: 1.0}, k=10)
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        db2.close()
