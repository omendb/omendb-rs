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


# --- max_distance tests ---


def test_search_max_distance_basic(db):
    """Test max_distance filters out distant results"""
    vectors = [
        {"id": "near", "vector": [0.0] * 128, "metadata": {}},
        {"id": "medium", "vector": [0.5] * 128, "metadata": {}},
        {"id": "far", "vector": [1.0] * 128, "metadata": {}},
    ]
    db.set(vectors)

    # Query from origin - get all within small distance
    results = db.search([0.0] * 128, k=10, max_distance=1.0)

    # "near" should always be included (distance ~0)
    assert any(r["id"] == "near" for r in results)
    # All results should be within max_distance
    assert all(r["distance"] <= 1.0 for r in results)


def test_search_max_distance_filters_all(db):
    """Test max_distance=0 with no exact matches returns empty"""
    vectors = [
        {"id": "v1", "vector": [0.1] * 128, "metadata": {}},
        {"id": "v2", "vector": [0.2] * 128, "metadata": {}},
    ]
    db.set(vectors)

    # Query with very small max_distance - should filter everything
    results = db.search([0.5] * 128, k=10, max_distance=0.001)

    # All vectors are far from [0.5]*128, so should be empty
    assert len(results) == 0


def test_search_max_distance_exact_match(db):
    """Test max_distance with exact match"""
    vector = {"id": "exact", "vector": [0.123] * 128, "metadata": {}}
    db.set([vector])

    # Query for exact match with small threshold
    results = db.search([0.123] * 128, k=1, max_distance=0.001)

    assert len(results) == 1
    assert results[0]["id"] == "exact"
    assert results[0]["distance"] < 0.001


def test_search_max_distance_none_returns_all(db):
    """Test that max_distance=None (default) returns all k results"""
    vectors = [{"id": f"v{i}", "vector": [i * 0.1] * 128, "metadata": {}} for i in range(10)]
    db.set(vectors)

    # Without max_distance, should return k results
    results = db.search([0.0] * 128, k=5)
    assert len(results) == 5

    # With max_distance=None explicitly, same behavior
    results = db.search([0.0] * 128, k=5, max_distance=None)
    assert len(results) == 5


def test_search_max_distance_with_filter(db):
    """Test max_distance combined with metadata filter"""
    vectors = [
        {"id": "near_a", "vector": [0.0] * 128, "metadata": {"type": "A"}},
        {"id": "near_b", "vector": [0.1] * 128, "metadata": {"type": "B"}},
        {"id": "far_a", "vector": [1.0] * 128, "metadata": {"type": "A"}},
        {"id": "far_b", "vector": [1.0] * 128, "metadata": {"type": "B"}},
    ]
    db.set(vectors)

    # Filter by type AND distance
    results = db.search([0.0] * 128, k=10, filter={"type": "A"}, max_distance=5.0)

    # Should only get type=A vectors within distance
    assert all(r["metadata"]["type"] == "A" for r in results)
    assert all(r["distance"] <= 5.0 for r in results)


def test_search_max_distance_returns_fewer_than_k(db):
    """Test that max_distance can return fewer than k results"""
    vectors = [
        {"id": "v1", "vector": [0.0] * 128, "metadata": {}},
        {"id": "v2", "vector": [0.1] * 128, "metadata": {}},
        {"id": "v3", "vector": [10.0] * 128, "metadata": {}},  # Very far
    ]
    db.set(vectors)

    # Request k=10 but max_distance should filter some out
    results = db.search([0.0] * 128, k=10, max_distance=2.0)

    # Should get fewer than k if some are beyond max_distance
    assert len(results) < 3 or all(r["distance"] <= 2.0 for r in results)


def test_search_max_distance_large_value(db_with_vectors):
    """Test max_distance with large value returns all results"""
    results = db_with_vectors.search([0.1] * 128, k=10, max_distance=1000.0)

    # Large max_distance should not filter anything
    assert len(results) == 5  # All vectors in db_with_vectors


def test_search_max_distance_boundary(db):
    """Test max_distance boundary condition (equal to distance)"""
    # Create vectors with known distances
    vectors = [
        {"id": "origin", "vector": [0.0] * 128, "metadata": {}},
    ]
    db.set(vectors)

    # Search from origin - distance to origin is 0
    results = db.search([0.0] * 128, k=1, max_distance=0.0)

    # Exact match should be included (distance <= max_distance)
    assert len(results) == 1
    assert results[0]["id"] == "origin"


def test_search_max_distance_cosine_metric(temp_db_path):
    """Test max_distance with cosine distance metric"""
    import gc

    import omendb

    db = omendb.open(temp_db_path, dimensions=128, metric="cosine")

    # For cosine, distance = 1 - cosine_similarity
    # Identical vectors have distance 0, orthogonal have distance 1, opposite have distance 2
    vectors = [
        {"id": "same_direction", "vector": [1.0] * 128, "metadata": {}},
        {"id": "different", "vector": [1.0] + [0.0] * 127, "metadata": {}},
    ]
    db.set(vectors)

    # Query in same direction as "same_direction"
    query = [1.0] * 128
    results = db.search(query, k=10, max_distance=0.1)

    # "same_direction" should be included (cosine distance ~0)
    assert any(r["id"] == "same_direction" for r in results)
    assert all(r["distance"] <= 0.1 for r in results)

    del db
    gc.collect()


def test_search_max_distance_dot_metric(temp_db_path):
    """Test max_distance with dot product (inner product) metric"""
    import gc

    import omendb

    db = omendb.open(temp_db_path, dimensions=128, metric="dot")

    # For dot/ip, we use negative dot product so lower is better
    # Normalized vectors: high similarity = low distance
    vectors = [
        {"id": "aligned", "vector": [1.0] * 128, "metadata": {}},
        {"id": "orthogonal", "vector": [1.0] + [0.0] * 127, "metadata": {}},
    ]
    db.set(vectors)

    query = [1.0] * 128
    results = db.search(query, k=10, max_distance=0.0)

    # Results depend on how dot product is converted to distance
    # Just verify the filter is applied
    assert all(r["distance"] <= 0.0 for r in results)

    del db
    gc.collect()


def test_search_max_distance_negative_raises(db_with_vectors):
    """Test that negative max_distance raises ValueError"""
    with pytest.raises(ValueError, match="max_distance must be non-negative"):
        db_with_vectors.search([0.1] * 128, k=5, max_distance=-1.0)


def test_search_sparse_filter_acorn1(temp_db_path):
    """Test ACORN-1 filtered search with sparse filter (entry points may not match).

    Regression test for bug where filtered search returned empty when
    the entry point didn't match the filter. ACORN-1 requires traversing
    THROUGH non-matching nodes to find matching ones.
    """
    import gc

    import numpy as np

    import omendb

    # Create dataset where only 1% matches the filter
    n = 1000
    dim = 64
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float32)

    db = omendb.open(temp_db_path, dimensions=dim, m=16, ef_construction=100)

    items = []
    for i in range(n):
        items.append(
            {
                "id": f"d{i}",
                "vector": vectors[i].tolist(),
                "metadata": {"category": 0 if i < 10 else 1},  # 1% match
            }
        )
    db.set(items)

    # Query - entry point is likely NOT in category 0
    query = np.random.randn(dim).astype(np.float32)
    results = db.search(query.tolist(), k=5, filter={"category": 0}, ef=100)

    # Must find results even with sparse filter
    assert len(results) > 0, "ACORN-1 should find results through non-matching entry points"
    assert len(results) <= 5

    # Verify all results match the filter
    for r in results:
        idx = int(r["id"][1:])
        assert idx < 10, f"Result {r['id']} should match category=0 filter"

    del db
    gc.collect()
