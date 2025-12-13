"""Tests for persistence (save/load) operations"""

import os

import pytest

import omendb


def test_save_and_load(temp_db_path):
    """Test basic save and load"""
    # Create and populate database
    db = omendb.open(temp_db_path, dimensions=128)
    vectors = [
        {"id": "v1", "vector": [0.1] * 128, "metadata": {"label": "A"}},
        {"id": "v2", "vector": [0.2] * 128, "metadata": {"label": "B"}},
    ]
    db.set(vectors)

    # Save
    db.flush()

    # Close and reopen
    del db
    db2 = omendb.open(temp_db_path, dimensions=128)

    # Verify data persisted
    assert len(db2) == 2

    results = db2.search([0.1] * 128, k=2)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {"v1", "v2"}


def test_save_creates_files(temp_db_path):
    """Test that save creates expected files (.omen format)"""
    db = omendb.open(temp_db_path, dimensions=128)
    vectors = [{"id": "v1", "vector": [0.1] * 128, "metadata": {}}]
    db.set(vectors)

    db.flush()

    # Check that .omen file was created
    directory = os.path.dirname(temp_db_path)
    filename = os.path.basename(temp_db_path)
    omen_file = f"{filename}.omen"

    assert os.path.exists(os.path.join(directory, omen_file)), f"Missing {omen_file}"
    # .omen file should contain all data (vectors, HNSW, metadata, mappings)
    file_size = os.path.getsize(os.path.join(directory, omen_file))
    assert file_size > 0, ".omen file should not be empty"


def test_load_preserves_metadata(temp_db_path):
    """Test that metadata is preserved across save/load"""
    db = omendb.open(temp_db_path, dimensions=128)
    vectors = [
        {
            "id": "v1",
            "vector": [0.1] * 128,
            "metadata": {
                "title": "Test Doc",
                "tags": ["a", "b"],
                "count": 42,
                "nested": {"key": "value"},
            },
        }
    ]
    db.set(vectors)
    db.flush()

    del db
    db2 = omendb.open(temp_db_path, dimensions=128)

    results = db2.search([0.1] * 128, k=1)
    metadata = results[0]["metadata"]

    assert metadata["title"] == "Test Doc"
    assert metadata["tags"] == ["a", "b"]
    assert metadata["count"] == 42
    assert metadata["nested"]["key"] == "value"


def test_load_preserves_distances(temp_db_path):
    """Test that distances are preserved (vectors loaded correctly)"""
    db = omendb.open(temp_db_path, dimensions=4)
    vectors = [
        {"id": "v1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "v2", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
    ]
    db.set(vectors)

    # Search before save
    results_before = db.search([1.0, 0.0, 0.0, 0.0], k=2)
    distances_before = [r["distance"] for r in results_before]

    db.flush()
    del db

    # Search after load
    db2 = omendb.open(temp_db_path, dimensions=4)
    results_after = db2.search([1.0, 0.0, 0.0, 0.0], k=2)
    distances_after = [r["distance"] for r in results_after]

    # Distances should be identical
    assert distances_before == pytest.approx(distances_after, abs=0.0001)


def test_save_after_delete(temp_db_path):
    """Test save/load with deleted vectors"""
    db = omendb.open(temp_db_path, dimensions=128)
    vectors = [
        {"id": "v1", "vector": [0.1] * 128, "metadata": {}},
        {"id": "v2", "vector": [0.2] * 128, "metadata": {}},
        {"id": "v3", "vector": [0.3] * 128, "metadata": {}},
    ]
    db.set(vectors)

    # Delete v2
    db.delete(["v2"])
    db.flush()

    del db
    db2 = omendb.open(temp_db_path, dimensions=128)

    # Should have 2 active vectors (v2 deleted)
    assert len(db2) == 2

    # Search should only return 2 (v2 is deleted)
    results = db2.search([0.2] * 128, k=10)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {"v1", "v3"}


def test_save_multiple_times(temp_db_path):
    """Test saving multiple times (updates existing files)"""
    db = omendb.open(temp_db_path, dimensions=128)

    # First save
    vectors1 = [{"id": "v1", "vector": [0.1] * 128, "metadata": {}}]
    db.set(vectors1)
    db.flush()

    # Second save with more data
    vectors2 = [{"id": "v2", "vector": [0.2] * 128, "metadata": {}}]
    db.set(vectors2)
    db.flush()

    del db

    # Load should have both vectors
    db2 = omendb.open(temp_db_path, dimensions=128)
    assert len(db2) == 2


def test_load_without_save_fails(temp_db_path):
    """Test that loading non-existent database creates new one"""
    # Try to open non-existent database
    db = omendb.open(temp_db_path, dimensions=128)

    # Should create empty database
    assert len(db) == 0


def test_save_empty_database(temp_db_path):
    """Test saving empty database"""
    db = omendb.open(temp_db_path, dimensions=128)
    db.flush()

    del db
    db2 = omendb.open(temp_db_path, dimensions=128)
    assert len(db2) == 0


def test_load_with_wrong_dimensions(temp_db_path):
    """Test loading database with wrong dimensions"""
    # Create with 128 dims
    db = omendb.open(temp_db_path, dimensions=128)
    vectors = [{"id": "v1", "vector": [0.1] * 128, "metadata": {}}]
    db.set(vectors)
    db.flush()
    del db

    # Try to load with 64 dims - should succeed (dimensions is just a parameter)
    db2 = omendb.open(temp_db_path, dimensions=64)

    # But search with wrong dims should fail
    with pytest.raises(ValueError, match="dimension"):
        db2.search([0.1] * 64, k=1)


def test_incremental_updates(temp_db_path):
    """Test incremental updates (insert, save, insert, save)"""
    db = omendb.open(temp_db_path, dimensions=128)

    # First batch
    batch1 = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(10)]
    db.set(batch1)
    db.flush()

    # Second batch
    batch2 = [{"id": f"v{i}", "vector": [float(i)] * 128, "metadata": {}} for i in range(10, 20)]
    db.set(batch2)
    db.flush()

    del db

    # Load should have all vectors
    db2 = omendb.open(temp_db_path, dimensions=128)
    assert len(db2) == 20


def test_set_after_reload(temp_db_path):
    """Test that set works after reload"""
    # Initial save
    db = omendb.open(temp_db_path, dimensions=128)
    db.set([{"id": "v1", "vector": [0.1] * 128, "metadata": {}}])
    db.flush()
    del db

    # Reload and add more
    db2 = omendb.open(temp_db_path, dimensions=128)
    db2.set([{"id": "v2", "vector": [0.2] * 128, "metadata": {}}])

    results = db2.search([0.1] * 128, k=10)
    assert len(results) == 2


def test_delete_after_reload(temp_db_path):
    """Test that delete works after reload"""
    # Initial save
    db = omendb.open(temp_db_path, dimensions=128)
    db.set(
        [
            {"id": "v1", "vector": [0.1] * 128, "metadata": {}},
            {"id": "v2", "vector": [0.2] * 128, "metadata": {}},
        ]
    )
    db.flush()
    del db

    # Reload and delete
    db2 = omendb.open(temp_db_path, dimensions=128)
    db2.delete(["v1"])

    results = db2.search([0.1] * 128, k=10)
    assert len(results) == 1
    assert results[0]["id"] == "v2"
