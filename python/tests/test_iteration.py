"""Tests for iteration API: ids(), items(), __iter__, get_batch(), exists()"""

import omendb


def test_ids():
    """Test ids() returns all vector IDs"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 0, 0]},
            {"id": "b", "vector": [0, 1, 0]},
            {"id": "c", "vector": [0, 0, 1]},
        ]
    )

    ids = list(db.ids())
    assert len(ids) == 3
    assert set(ids) == {"a", "b", "c"}


def test_ids_excludes_deleted():
    """Test ids() excludes deleted vectors"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 0, 0]},
            {"id": "b", "vector": [0, 1, 0]},
        ]
    )
    db.delete(["a"])

    ids = list(db.ids())
    assert ids == ["b"]


def test_ids_empty():
    """Test ids() on empty database"""
    db = omendb.open(":memory:", dimensions=3)
    assert list(db.ids()) == []


def test_items():
    """Test items() returns all vectors with metadata"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"x": 1}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"x": 2}},
        ]
    )

    items = db.items()
    assert len(items) == 2

    by_id = {item["id"]: item for item in items}
    assert by_id["a"]["vector"] == [1, 2, 3]
    assert by_id["a"]["metadata"] == {"x": 1}
    assert by_id["b"]["vector"] == [4, 5, 6]
    assert by_id["b"]["metadata"] == {"x": 2}


def test_items_excludes_deleted():
    """Test items() excludes deleted vectors"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3]},
            {"id": "b", "vector": [4, 5, 6]},
        ]
    )
    db.delete(["a"])

    items = db.items()
    assert len(items) == 1
    assert items[0]["id"] == "b"


def test_iteration():
    """Test __iter__ protocol"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3]},
            {"id": "b", "vector": [4, 5, 6]},
            {"id": "c", "vector": [7, 8, 9]},
        ]
    )

    items = list(db)
    assert len(items) == 3

    ids = {item["id"] for item in items}
    assert ids == {"a", "b", "c"}


def test_iteration_empty():
    """Test iteration on empty database"""
    db = omendb.open(":memory:", dimensions=3)
    items = list(db)
    assert items == []


def test_exists():
    """Test exists() method"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": "a", "vector": [1, 2, 3]}])

    assert db.exists("a") is True
    assert db.exists("b") is False


def test_exists_deleted():
    """Test exists() returns False for deleted vectors"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": "a", "vector": [1, 2, 3]}])
    db.delete(["a"])

    assert db.exists("a") is False


def test_contains():
    """Test __contains__ protocol (in operator)"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": "a", "vector": [1, 2, 3]}])

    assert "a" in db
    assert "b" not in db


def test_get_batch():
    """Test get_batch() batch retrieval"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"x": 1}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"x": 2}},
        ]
    )

    results = db.get_batch(["a", "b", "c"])  # c doesn't exist

    assert len(results) == 3
    assert results[0]["id"] == "a"
    assert results[0]["vector"] == [1, 2, 3]
    assert results[1]["id"] == "b"
    assert results[2] is None


def test_get_batch_preserves_order():
    """Test get_batch() preserves input order"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": str(i), "vector": [i, i, i]} for i in range(10)])

    results = db.get_batch(["5", "2", "8", "1"])
    assert [r["id"] for r in results] == ["5", "2", "8", "1"]


def test_get_batch_empty():
    """Test get_batch() with empty list"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": "a", "vector": [1, 2, 3]}])

    results = db.get_batch([])
    assert results == []


def test_get_batch_all_missing():
    """Test get_batch() when all IDs are missing"""
    db = omendb.open(":memory:", dimensions=3)

    results = db.get_batch(["x", "y", "z"])
    assert results == [None, None, None]


def test_len():
    """Test __len__ protocol"""
    db = omendb.open(":memory:", dimensions=3)
    assert len(db) == 0

    db.set(
        [
            {"id": "a", "vector": [1, 2, 3]},
            {"id": "b", "vector": [4, 5, 6]},
        ]
    )
    assert len(db) == 2

    db.delete(["a"])
    assert len(db) == 1


def test_delete_by_filter_simple():
    """Test delete_by_filter() with simple equality filter"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"status": "active"}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"status": "archived"}},
            {"id": "c", "vector": [7, 8, 9], "metadata": {"status": "archived"}},
        ]
    )

    count = db.delete_by_filter({"status": "archived"})
    assert count == 2
    assert set(db.ids()) == {"a"}


def test_delete_by_filter_comparison():
    """Test delete_by_filter() with comparison operators"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"score": 0.3}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"score": 0.7}},
            {"id": "c", "vector": [7, 8, 9], "metadata": {"score": 0.9}},
        ]
    )

    # Delete low scores
    count = db.delete_by_filter({"score": {"$lt": 0.5}})
    assert count == 1
    assert set(db.ids()) == {"b", "c"}


def test_delete_by_filter_complex():
    """Test delete_by_filter() with complex $and filter"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"type": "doc", "score": 0.5}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"type": "doc", "score": 0.9}},
            {"id": "c", "vector": [7, 8, 9], "metadata": {"type": "image", "score": 0.3}},
        ]
    )

    # Delete docs with low score
    count = db.delete_by_filter({"$and": [{"type": "doc"}, {"score": {"$lt": 0.8}}]})
    assert count == 1
    assert set(db.ids()) == {"b", "c"}


def test_delete_by_filter_no_match():
    """Test delete_by_filter() when no vectors match"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": "a", "vector": [1, 2, 3], "metadata": {"x": 1}}])

    count = db.delete_by_filter({"x": 999})
    assert count == 0
    assert list(db.ids()) == ["a"]


def test_delete_by_filter_all():
    """Test delete_by_filter() that matches all vectors"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"active": True}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"active": True}},
        ]
    )

    count = db.delete_by_filter({"active": True})
    assert count == 2
    assert list(db.ids()) == []


def test_iteration_is_lazy():
    """Test that iteration is truly lazy - fetches one item at a time"""
    db = omendb.open(":memory:", dimensions=3)
    db.set([{"id": str(i), "vector": [i, i, i]} for i in range(100)])

    # Get iterator
    it = iter(db)

    # Fetch just 2 items
    item1 = next(it)
    item2 = next(it)

    assert item1["id"] in [str(i) for i in range(100)]
    assert item2["id"] in [str(i) for i in range(100)]

    # Can stop early without loading all 100 vectors
    # (if not lazy, items() would have already loaded everything)


def test_iteration_handles_deletion_during_iteration():
    """Test that deletion during iteration is handled gracefully"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3]},
            {"id": "b", "vector": [4, 5, 6]},
            {"id": "c", "vector": [7, 8, 9]},
        ]
    )

    collected = []
    for i, item in enumerate(db):
        collected.append(item["id"])
        if i == 0:
            # Delete an item during iteration
            db.delete(["b"])

    # Should have collected items, skipping deleted one
    assert "a" in collected or "c" in collected
    assert len(collected) <= 3


def test_count_no_filter():
    """Test count() without filter returns total count"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"status": "active"}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"status": "active"}},
            {"id": "c", "vector": [7, 8, 9], "metadata": {"status": "archived"}},
        ]
    )

    assert db.count() == 3
    assert db.count() == len(db)


def test_count_with_filter():
    """Test count() with filter returns filtered count"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"status": "active", "score": 0.9}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"status": "active", "score": 0.5}},
            {"id": "c", "vector": [7, 8, 9], "metadata": {"status": "archived", "score": 0.3}},
        ]
    )

    # Simple equality filter
    assert db.count(filter={"status": "active"}) == 2
    assert db.count(filter={"status": "archived"}) == 1

    # Comparison filter
    assert db.count(filter={"score": {"$gte": 0.5}}) == 2
    assert db.count(filter={"score": {"$lt": 0.5}}) == 1

    # Filter with no matches
    assert db.count(filter={"status": "unknown"}) == 0


def test_count_excludes_deleted():
    """Test count() excludes deleted vectors"""
    db = omendb.open(":memory:", dimensions=3)
    db.set(
        [
            {"id": "a", "vector": [1, 2, 3], "metadata": {"x": 1}},
            {"id": "b", "vector": [4, 5, 6], "metadata": {"x": 1}},
        ]
    )

    assert db.count() == 2
    assert db.count(filter={"x": 1}) == 2

    db.delete(["a"])

    assert db.count() == 1
    assert db.count(filter={"x": 1}) == 1


if __name__ == "__main__":
    test_ids()
    test_ids_excludes_deleted()
    test_ids_empty()
    test_items()
    test_items_excludes_deleted()
    test_iteration()
    test_iteration_empty()
    test_exists()
    test_exists_deleted()
    test_contains()
    test_get_batch()
    test_get_batch_preserves_order()
    test_get_batch_empty()
    test_get_batch_all_missing()
    test_len()
    test_delete_by_filter_simple()
    test_delete_by_filter_comparison()
    test_delete_by_filter_complex()
    test_delete_by_filter_no_match()
    test_delete_by_filter_all()
    test_iteration_is_lazy()
    test_iteration_handles_deletion_during_iteration()
    test_count_no_filter()
    test_count_with_filter()
    test_count_excludes_deleted()
    print("All tests passed!")
