"""Tests for search operations"""

import pytest


def test_search_empty_database(db):
    """Test searching empty database returns empty list."""
    results = db.search([0.1] * 128, k=10)
    assert results == []


def test_search_basic(db_with_vectors):
    """Test basic k-NN search"""
    query = [0.15] * 128
    results = db_with_vectors.search(query, k=2)

    assert len(results) == 2
    assert all("id" in r for r in results)
    assert all("distance" in r for r in results)
    assert all("metadata" in r for r in results)


def test_search_k_larger_than_database(db_with_vectors):
    """Test k larger than number of vectors"""
    results = db_with_vectors.search([0.1] * 128, k=100)

    # Should return all vectors (5)
    assert len(results) == 5


def test_search_k_zero(db_with_vectors):
    """Test k=0 raises error"""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        db_with_vectors.search([0.1] * 128, k=0)


def test_search_k_one(db_with_vectors):
    """Test k=1 returns single result"""
    results = db_with_vectors.search([0.1] * 128, k=1)
    assert len(results) == 1


def test_search_invalid_dimensions(db_with_vectors):
    """Test search with wrong query dimensions"""
    with pytest.raises(ValueError, match="dimension"):
        db_with_vectors.search([0.1] * 64, k=5)  # Wrong: 64 instead of 128


def test_search_distance_ordering(db):
    """Test that results are ordered by distance"""
    vectors = [
        {"id": "far", "vector": [1.0] * 128, "metadata": {}},
        {"id": "near", "vector": [0.0] * 128, "metadata": {}},
        {"id": "medium", "vector": [0.5] * 128, "metadata": {}},
    ]
    db.set(vectors)

    # Query closest to "near"
    results = db.search([0.0] * 128, k=3)

    assert results[0]["id"] == "near"
    assert results[1]["id"] == "medium"
    assert results[2]["id"] == "far"

    # Distances should be increasing
    assert results[0]["distance"] < results[1]["distance"]
    assert results[1]["distance"] < results[2]["distance"]


def test_search_exact_match(db):
    """Test searching for exact match"""
    vector = {"id": "test1", "vector": [0.123] * 128, "metadata": {}}
    db.set([vector])

    results = db.search([0.123] * 128, k=1)

    assert len(results) == 1
    assert results[0]["id"] == "test1"
    assert results[0]["distance"] < 0.001  # Should be very close to 0


def test_search_with_filter_equals(db_with_vectors):
    """Test search with equality filter"""
    # Search for vectors with label = "A"
    results = db_with_vectors.search([0.1] * 128, k=10, filter={"label": "A"})

    assert len(results) == 2  # vec1 and vec4 have label="A"
    assert all(r["metadata"]["label"] == "A" for r in results)


def test_search_with_filter_gte(db_with_vectors):
    """Test search with $gte filter"""
    # Search for vectors with value >= 3
    results = db_with_vectors.search([0.3] * 128, k=10, filter={"value": {"$gte": 3}})

    assert len(results) == 3  # vec3, vec4, vec5
    assert all(r["metadata"]["value"] >= 3 for r in results)


def test_search_with_filter_lte(db_with_vectors):
    """Test search with $lte filter"""
    results = db_with_vectors.search([0.2] * 128, k=10, filter={"value": {"$lte": 2}})

    assert len(results) == 2  # vec1, vec2
    assert all(r["metadata"]["value"] <= 2 for r in results)


def test_search_with_filter_gt(db_with_vectors):
    """Test search with $gt filter"""
    results = db_with_vectors.search([0.3] * 128, k=10, filter={"value": {"$gt": 3}})

    assert len(results) == 2  # vec4, vec5
    assert all(r["metadata"]["value"] > 3 for r in results)


def test_search_with_filter_lt(db_with_vectors):
    """Test search with $lt filter"""
    results = db_with_vectors.search([0.1] * 128, k=10, filter={"value": {"$lt": 3}})

    assert len(results) == 2  # vec1, vec2
    assert all(r["metadata"]["value"] < 3 for r in results)


def test_search_with_filter_in(db_with_vectors):
    """Test search with $in filter"""
    results = db_with_vectors.search([0.2] * 128, k=10, filter={"label": {"$in": ["A", "C"]}})

    assert len(results) == 3  # vec1, vec3, vec4
    assert all(r["metadata"]["label"] in ["A", "C"] for r in results)


def test_search_with_filter_and(db_with_vectors):
    """Test search with $and filter"""
    results = db_with_vectors.search(
        [0.1] * 128,
        k=10,
        filter={"$and": [{"value": {"$gte": 2}}, {"value": {"$lte": 4}}]},
    )

    assert len(results) == 3  # vec2, vec3, vec4
    assert all(2 <= r["metadata"]["value"] <= 4 for r in results)


def test_search_with_filter_or(db_with_vectors):
    """Test search with $or filter"""
    results = db_with_vectors.search(
        [0.3] * 128, k=10, filter={"$or": [{"label": "A"}, {"value": 5}]}
    )

    assert len(results) == 3  # vec1, vec4 (label=A), vec5 (value=5)


def test_search_no_filter(db_with_vectors):
    """Test search without filter returns all results"""
    results = db_with_vectors.search([0.3] * 128, k=10)
    assert len(results) == 5  # All vectors


def test_search_filter_no_matches(db_with_vectors):
    """Test filter with no matching vectors"""
    results = db_with_vectors.search([0.3] * 128, k=10, filter={"label": "NONEXISTENT"})

    assert results == []


def test_search_after_delete(db_with_vectors):
    """Test that deleted vectors don't appear in search results"""
    # Delete vec2
    db_with_vectors.delete(["vec2"])

    results = db_with_vectors.search([0.2] * 128, k=10)

    # Should have 4 results (vec2 deleted)
    assert len(results) == 4
    assert all(r["id"] != "vec2" for r in results)


def test_search_metadata_returned(db):
    """Test that metadata is correctly returned in search results"""
    vector = {
        "id": "test1",
        "vector": [0.1] * 128,
        "metadata": {
            "title": "Test Document",
            "tags": ["important", "reviewed"],
            "count": 42,
        },
    }
    db.set([vector])

    results = db.search([0.1] * 128, k=1)

    assert results[0]["metadata"]["title"] == "Test Document"
    assert results[0]["metadata"]["tags"] == ["important", "reviewed"]
    assert results[0]["metadata"]["count"] == 42
