"""Basic tests for OmenDB Python bindings"""

import os
import tempfile

import omendb


def test_open_database():
    """Test opening a database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=128)
        assert db is not None


def test_set_and_search():
    """Test basic set and search"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=3)

        # Set some vectors
        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0],
                    "metadata": {"title": "Document 1", "year": 2024},
                },
                {
                    "id": "doc2",
                    "vector": [0.0, 1.0, 0.0],
                    "metadata": {"title": "Document 2", "year": 2023},
                },
                {
                    "id": "doc3",
                    "vector": [0.0, 0.0, 1.0],
                    "metadata": {"title": "Document 3", "year": 2024},
                },
            ]
        )

        # Search for nearest neighbors
        results = db.search(query=[1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert "distance" in results[0]
        assert "metadata" in results[0]
        assert results[0]["metadata"]["title"] == "Document 1"


def test_search_with_filter():
    """Test search with metadata filter"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=3)

        # Set vectors
        db.set(
            [
                {
                    "id": "doc1",
                    "vector": [1.0, 0.0, 0.0],
                    "metadata": {"title": "Document 1", "year": 2024},
                },
                {
                    "id": "doc2",
                    "vector": [0.9, 0.1, 0.0],
                    "metadata": {"title": "Document 2", "year": 2023},
                },
                {
                    "id": "doc3",
                    "vector": [0.8, 0.2, 0.0],
                    "metadata": {"title": "Document 3", "year": 2024},
                },
            ]
        )

        # Search with filter for year >= 2024
        results = db.search(query=[1.0, 0.0, 0.0], k=10, filter={"year": {"$gte": 2024}})

        # Should only return doc1 and doc3
        assert len(results) == 2
        for result in results:
            assert result["metadata"]["year"] >= 2024


def test_set_update():
    """Test that set updates existing documents"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=3)

        # Insert initial document
        db.set([{"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"title": "Original"}}])

        # Set with same ID (should update)
        db.set([{"id": "doc1", "vector": [0.0, 1.0, 0.0], "metadata": {"title": "Updated"}}])

        # Get the document
        doc = db.get("doc1")
        assert doc is not None
        assert doc["metadata"]["title"] == "Updated"
        assert doc["vector"] == [0.0, 1.0, 0.0]


def test_delete():
    """Test deleting documents"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        db = omendb.open(db_path, dimensions=3)

        # Insert documents
        db.set(
            [
                {"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"title": "Document 1"}},
                {"id": "doc2", "vector": [0.0, 1.0, 0.0], "metadata": {"title": "Document 2"}},
            ]
        )

        # Delete doc1
        deleted_count = db.delete(["doc1"])
        assert deleted_count == 1

        # Verify doc1 is gone
        assert db.get("doc1") is None
        assert db.get("doc2") is not None


def test_save_and_load():
    """Test saving and loading database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")

        # Create and populate database
        db = omendb.open(db_path, dimensions=3)
        db.set([{"id": "doc1", "vector": [1.0, 0.0, 0.0], "metadata": {"title": "Document 1"}}])

        # Save and close database
        db.flush()
        del db

        # Load database
        db2 = omendb.open(db_path, dimensions=3)

        # Verify data was loaded
        doc = db2.get("doc1")
        assert doc is not None
        assert doc["metadata"]["title"] == "Document 1"


if __name__ == "__main__":
    test_open_database()
    test_set_and_search()
    test_search_with_filter()
    test_set_update()
    test_delete()
    test_save_and_load()
    print("✅ All tests passed!")
