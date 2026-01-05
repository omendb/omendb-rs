"""Tests for delete operations"""


def test_delete_single_vector(db_with_vectors):
    """Test deleting a single vector"""
    initial_count = len(db_with_vectors)

    db_with_vectors.delete(["vec1"])

    # len() excludes deleted vectors (returns active count)
    assert len(db_with_vectors) == initial_count - 1

    # Verify vec1 is not in search results (use high ef for small dataset)
    results = db_with_vectors.search([0.1] * 128, k=10, ef=100)
    assert len(results) == initial_count - 1
    assert all(r["id"] != "vec1" for r in results)


def test_delete_multiple_vectors(db_with_vectors):
    """Test deleting multiple vectors"""
    db_with_vectors.delete(["vec1", "vec2", "vec3"])

    # len() returns 2 (excludes deleted vectors)
    assert len(db_with_vectors) == 2

    # Search results should only have remaining vectors (use high ef)
    results = db_with_vectors.search([0.1] * 128, k=10, ef=100)
    assert len(results) == 2
    deleted_ids = {"vec1", "vec2", "vec3"}
    assert all(r["id"] not in deleted_ids for r in results)


def test_delete_all_vectors(db_with_vectors):
    """Test deleting all vectors"""
    db_with_vectors.delete(["vec1", "vec2", "vec3", "vec4", "vec5"])

    # len() returns 0 (all vectors deleted)
    assert len(db_with_vectors) == 0

    # Search should return empty
    results = db_with_vectors.search([0.1] * 128, k=10)
    assert results == []


def test_delete_empty_list(db_with_vectors):
    """Test deleting with empty list"""
    initial_count = len(db_with_vectors)

    db_with_vectors.delete([])

    assert len(db_with_vectors) == initial_count


def test_delete_nonexistent_id(db_with_vectors):
    """Test deleting non-existent ID is a no-op (idempotent)"""
    initial_results = db_with_vectors.search([0.1] * 128, k=10)
    initial_count = len(initial_results)

    # Delete non-existent ID (should be no-op)
    db_with_vectors.delete(["nonexistent_id"])

    # Results should be unchanged
    results = db_with_vectors.search([0.1] * 128, k=10)
    assert len(results) == initial_count


def test_delete_partial_nonexistent(db_with_vectors):
    """Test deleting mix of existing and non-existent IDs"""
    # Should delete existing IDs, ignore non-existent ones
    db_with_vectors.delete(["vec1", "nonexistent", "vec2"])

    results = db_with_vectors.search([0.1] * 128, k=10)
    # Should have deleted vec1 and vec2 (3 remaining)
    assert len(results) == 3
    assert all(r["id"] not in ["vec1", "vec2"] for r in results)


def test_delete_then_insert_same_id(db_with_vectors):
    """Test deleting then re-inserting same ID"""
    # Delete vec1
    db_with_vectors.delete(["vec1"])

    # Re-insert with same ID
    new_vector = {"id": "vec1", "vector": [0.9] * 128, "metadata": {"status": "new"}}
    db_with_vectors.set([new_vector])

    # Should be able to search and find it
    results = db_with_vectors.search([0.9] * 128, k=1)
    assert len(results) == 1
    assert results[0]["id"] == "vec1"
    assert results[0]["metadata"]["status"] == "new"


def test_delete_preserves_other_vectors(db_with_vectors):
    """Test that deleting one vector doesn't affect others"""
    # Get initial state of vec2
    results_before = db_with_vectors.search([0.2] * 128, k=1)
    vec2_metadata_before = results_before[0]["metadata"]

    # Delete vec1
    db_with_vectors.delete(["vec1"])

    # Verify vec2 is unchanged
    results_after = db_with_vectors.search([0.2] * 128, k=1)
    assert results_after[0]["id"] == "vec2"
    assert results_after[0]["metadata"] == vec2_metadata_before


def test_delete_twice_same_id(db_with_vectors):
    """Test deleting same ID twice is idempotent"""
    db_with_vectors.delete(["vec1"])

    # Second delete should be a no-op (idempotent)
    db_with_vectors.delete(["vec1"])

    # Verify vec1 still doesn't appear
    results = db_with_vectors.search([0.1] * 128, k=10)
    assert all(r["id"] != "vec1" for r in results)


def test_delete_from_empty_database(db):
    """Test deleting from empty database is a no-op"""
    # Should be no-op (idempotent)
    db.delete(["vec1"])

    # Database should still be empty
    assert len(db) == 0


def test_delete_special_characters_id(db):
    """Test deleting IDs with special characters"""
    vectors = [
        {"id": "test-1", "vector": [0.1] * 128, "metadata": {}},
        {"id": "test_2", "vector": [0.2] * 128, "metadata": {}},
        {"id": "test.3", "vector": [0.3] * 128, "metadata": {}},
    ]
    db.set(vectors)

    db.delete(["test-1", "test_2"])

    # len() returns 1 (only test.3 remains)
    assert len(db) == 1

    # Search should only find test.3
    results = db.search([0.1] * 128, k=10)
    assert len(results) == 1
    assert results[0]["id"] == "test.3"
