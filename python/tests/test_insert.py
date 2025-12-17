"""Tests for insert/set operations"""

import pytest


def test_insert_single_vector(db):
    """Test inserting a single vector"""
    vector = {"id": "test1", "vector": [0.1] * 128, "metadata": {}}
    indices = db.set([vector])

    assert indices == 1
    assert len(db) == 1


def test_insert_multiple_vectors(db):
    """Test batch insert"""
    vectors = [
        {"id": f"vec{i}", "vector": [float(i)] * 128, "metadata": {"index": i}} for i in range(100)
    ]
    indices = db.set(vectors)

    assert indices == 100
    assert len(db) == 100


def test_insert_empty_list(db):
    """Test inserting empty list"""
    indices = db.set([])
    assert indices == 0
    assert len(db) == 0


def test_insert_without_metadata(db):
    """Test inserting vector without metadata field"""
    vector = {"id": "test1", "vector": [0.1] * 128}
    indices = db.set([vector])

    assert indices == 1
    results = db.search([0.1] * 128, k=1)
    assert len(results) == 1
    assert results[0]["metadata"] == {}


def test_insert_invalid_dimensions(db):
    """Test inserting vector with wrong dimensions after db has vectors.

    Note: First insert sets dimensions (auto-detect), so we need to insert
    a valid vector first, then try to insert one with wrong dimensions.
    """
    # First insert sets dimensions to 128
    db.set([{"id": "valid", "vector": [0.1] * 128}])

    # Now inserting 64-dim vector should fail
    vector = {
        "id": "test1",
        "vector": [0.1] * 64,
        "metadata": {},
    }  # Wrong: 64 instead of 128

    with pytest.raises(ValueError, match="dimension"):
        db.set([vector])


def test_insert_missing_id(db):
    """Test inserting vector without ID"""
    vector = {"vector": [0.1] * 128, "metadata": {}}

    with pytest.raises(ValueError, match="id"):
        db.set([vector])


def test_insert_missing_vector(db):
    """Test inserting item without vector field"""
    item = {"id": "test1", "metadata": {}}

    with pytest.raises(ValueError, match="vector"):
        db.set([item])


def test_insert_not_list(db):
    """Test passing non-list to set"""
    vector = {"id": "test1", "vector": [0.1] * 128, "metadata": {}}

    with pytest.raises((TypeError, ValueError)):
        db.set(vector)  # Should be [vector]


def test_set_updates_existing(db):
    """Test that set updates existing vectors"""
    # Insert initial vector
    vector1 = {"id": "test1", "vector": [0.1] * 128, "metadata": {"version": 1}}
    db.set([vector1])

    # Search to verify initial value
    results = db.search([0.1] * 128, k=1)
    assert results[0]["metadata"]["version"] == 1

    # Update with new embedding and metadata
    vector2 = {"id": "test1", "vector": [0.2] * 128, "metadata": {"version": 2}}
    db.set([vector2])

    # Should still have 1 vector
    assert len(db) == 1

    # Search should find updated vector
    results = db.search([0.2] * 128, k=1)
    assert len(results) == 1
    assert results[0]["id"] == "test1"
    assert results[0]["metadata"]["version"] == 2


def test_set_mixed_new_and_updates(db):
    """Test set with mix of new and existing vectors"""
    # Insert initial vectors
    initial = [
        {"id": "vec1", "vector": [0.1] * 128, "metadata": {"v": 1}},
        {"id": "vec2", "vector": [0.2] * 128, "metadata": {"v": 1}},
    ]
    db.set(initial)
    assert len(db) == 2

    # Set mix of updates and new
    mixed = [
        {"id": "vec1", "vector": [0.15] * 128, "metadata": {"v": 2}},  # Update
        {"id": "vec3", "vector": [0.3] * 128, "metadata": {"v": 1}},  # New
    ]
    db.set(mixed)

    # Should have 3 vectors total
    assert len(db) == 3

    # Verify update by searching for all and checking metadata
    results = db.search([0.15] * 128, k=3)
    vec1_result = next((r for r in results if r["id"] == "vec1"), None)
    assert vec1_result is not None, "vec1 should exist after set"
    assert vec1_result["metadata"]["v"] == 2, "vec1 metadata should be updated"


def test_insert_special_characters_in_id(db):
    """Test inserting vectors with special characters in ID"""
    vectors = [
        {"id": "test-1", "vector": [0.1] * 128, "metadata": {}},
        {"id": "test_2", "vector": [0.2] * 128, "metadata": {}},
        {"id": "test.3", "vector": [0.3] * 128, "metadata": {}},
        {"id": "test/4", "vector": [0.4] * 128, "metadata": {}},
    ]
    indices = db.set(vectors)

    assert indices == 4
    assert len(db) == 4


def test_insert_unicode_metadata(db):
    """Test inserting vectors with Unicode metadata"""
    vector = {
        "id": "test1",
        "vector": [0.1] * 128,
        "metadata": {
            "text": "Hello 世界 🌍",
            "language": "中文",
        },
    }
    db.set([vector])

    results = db.search([0.1] * 128, k=1)
    assert results[0]["metadata"]["text"] == "Hello 世界 🌍"
    assert results[0]["metadata"]["language"] == "中文"


def test_insert_nested_metadata(db):
    """Test inserting vectors with nested metadata"""
    vector = {
        "id": "test1",
        "vector": [0.1] * 128,
        "metadata": {
            "user": {"name": "Alice", "age": 30},
            "tags": ["important", "verified"],
        },
    }
    db.set([vector])

    results = db.search([0.1] * 128, k=1)
    assert results[0]["metadata"]["user"]["name"] == "Alice"
    assert results[0]["metadata"]["tags"] == ["important", "verified"]


def test_insert_with_text(db):
    """Test inserting vectors with text field auto-stores in metadata"""
    vector = {
        "id": "doc1",
        "vector": [0.1] * 128,
        "text": "This is searchable text content.",
    }
    db.set([vector])

    results = db.search([0.1] * 128, k=1)
    assert results[0]["id"] == "doc1"
    # Text is auto-stored in metadata["text"]
    assert results[0]["metadata"]["text"] == "This is searchable text content."


def test_insert_text_with_metadata(db):
    """Test text field is merged with existing metadata"""
    vector = {
        "id": "doc1",
        "vector": [0.1] * 128,
        "text": "Searchable text",
        "metadata": {"title": "Test Document", "author": "Alice"},
    }
    db.set([vector])

    results = db.search([0.1] * 128, k=1)
    assert results[0]["metadata"]["text"] == "Searchable text"
    assert results[0]["metadata"]["title"] == "Test Document"
    assert results[0]["metadata"]["author"] == "Alice"


def test_insert_text_conflicts_with_metadata_text(db):
    """Test that text field and metadata['text'] conflict raises error"""
    import pytest

    vector = {
        "id": "doc1",
        "vector": [0.1] * 128,
        "text": "Text field value",
        "metadata": {"text": "Metadata text value"},
    }
    with pytest.raises(ValueError, match="cannot have both"):
        db.set([vector])
