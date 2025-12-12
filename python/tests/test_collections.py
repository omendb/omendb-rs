"""Tests for Collections API."""

import shutil
import tempfile

import pytest

import omendb


class TestCollections:
    """Test collection functionality."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        path = tempfile.mkdtemp(suffix="")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    def test_create_collection(self, db_path):
        """Test creating a named collection."""
        db = omendb.open(db_path, dimensions=128)
        users = db.collection("users")

        # Collection should be usable
        users.set([{"id": "user1", "vector": [0.1] * 128, "metadata": {"name": "Alice"}}])

        assert len(users) == 1
        result = users.get("user1")
        assert result is not None
        assert result["metadata"]["name"] == "Alice"

    def test_multiple_collections(self, db_path):
        """Test multiple isolated collections."""
        db = omendb.open(db_path, dimensions=128)

        users = db.collection("users")
        products = db.collection("products")

        # Insert same ID into both collections - should not conflict
        users.set([{"id": "doc1", "vector": [0.1] * 128, "metadata": {"type": "user"}}])

        products.set([{"id": "doc1", "vector": [0.2] * 128, "metadata": {"type": "product"}}])

        # Both should have 1 item each
        assert len(users) == 1
        assert len(products) == 1

        # Should have different metadata
        user_doc = users.get("doc1")
        product_doc = products.get("doc1")

        assert user_doc["metadata"]["type"] == "user"
        assert product_doc["metadata"]["type"] == "product"

    def test_collections(self, db_path):
        """Test listing collections."""
        db = omendb.open(db_path, dimensions=128)

        # Initially no collections
        assert db.collections() == []

        # Create some collections
        db.collection("alpha")
        db.collection("beta")
        db.collection("gamma")

        # Should list them (sorted)
        collections = db.collections()
        assert collections == ["alpha", "beta", "gamma"]

    def test_delete_collection(self, db_path):
        """Test deleting a collection."""
        db = omendb.open(db_path, dimensions=128)

        # Create and populate a collection
        users = db.collection("users")
        users.set([{"id": "user1", "vector": [0.1] * 128, "metadata": {"name": "Alice"}}])

        # Verify it exists
        assert "users" in db.collections()

        # Delete it
        db.delete_collection("users")

        # Should no longer be listed
        assert "users" not in db.collections()

    def test_delete_nonexistent_collection(self, db_path):
        """Test deleting a collection that doesn't exist."""
        db = omendb.open(db_path, dimensions=128)

        with pytest.raises(ValueError, match="not found"):
            db.delete_collection("nonexistent")

    def test_collection_name_validation(self, db_path):
        """Test collection name validation."""
        db = omendb.open(db_path, dimensions=128)

        # Empty name
        with pytest.raises(ValueError, match="cannot be empty"):
            db.collection("")

        # Invalid characters
        with pytest.raises(ValueError, match="alphanumeric"):
            db.collection("my-collection")

        with pytest.raises(ValueError, match="alphanumeric"):
            db.collection("my.collection")

        with pytest.raises(ValueError, match="alphanumeric"):
            db.collection("my collection")

        # Valid names
        db.collection("my_collection")
        db.collection("MyCollection123")
        db.collection("collection1")

    def test_collection_persistence(self, db_path):
        """Test that collections persist after reopening."""
        # Create and populate collections
        db1 = omendb.open(db_path, dimensions=128)
        users1 = db1.collection("users")
        users1.set([{"id": "user1", "vector": [0.1] * 128, "metadata": {"name": "Alice"}}])

        products1 = db1.collection("products")
        products1.set([{"id": "prod1", "vector": [0.2] * 128, "metadata": {"price": 99.99}}])

        # Explicitly flush before closing (required for persistence)
        users1.flush()
        products1.flush()
        del db1, users1, products1

        # Reopen
        db2 = omendb.open(db_path, dimensions=128)
        collections = db2.collections()
        assert "users" in collections
        assert "products" in collections

        # Verify data persisted
        users2 = db2.collection("users")
        result = users2.get("user1")
        assert result is not None
        assert result["metadata"]["name"] == "Alice"

        products2 = db2.collection("products")
        result = products2.get("prod1")
        assert result is not None
        assert result["metadata"]["price"] == 99.99

    def test_collection_search(self, db_path):
        """Test searching within a collection."""
        db = omendb.open(db_path, dimensions=128)

        users = db.collection("users")
        for i in range(100):
            users.set(
                [
                    {
                        "id": f"user{i}",
                        "vector": [i * 0.01] * 128,
                        "metadata": {"index": i},
                    }
                ]
            )

        # Search should work
        results = users.search(query=[0.5] * 128, k=10)
        assert len(results) == 10

        # Results should be sorted by distance
        for i in range(1, len(results)):
            assert results[i]["distance"] >= results[i - 1]["distance"]

    def test_collection_filtered_search(self, db_path):
        """Test filtered search within a collection."""
        db = omendb.open(db_path, dimensions=128)

        users = db.collection("users")
        for i in range(50):
            users.set(
                [
                    {
                        "id": f"user{i}",
                        "vector": [i * 0.02] * 128,
                        "metadata": {"age": 20 + (i % 30)},
                    }
                ]
            )

        # Filter by age >= 40
        results = users.search(query=[0.5] * 128, k=20, filter={"age": {"$gte": 40}})

        # All results should have age >= 40
        for result in results:
            assert result["metadata"]["age"] >= 40

    def test_collection_support_all_dbs(self):
        """Test that all databases support collections (seerdb is always persistent)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/mydb"
            db = omendb.open(path, dimensions=128)

            # All databases now use seerdb persistent storage, so collections work
            coll = db.collection("test")
            assert coll is not None

    def test_collection_delete_and_recreate(self, db_path):
        """Test deleting and recreating a collection."""
        db = omendb.open(db_path, dimensions=128)

        # Create and populate
        users = db.collection("users")
        users.set([{"id": "user1", "vector": [0.1] * 128, "metadata": {"name": "Alice"}}])

        # Delete
        db.delete_collection("users")
        assert "users" not in db.collections()

        # Recreate
        users2 = db.collection("users")
        assert len(users2) == 0  # Should be empty

        # Add new data
        users2.set([{"id": "user2", "vector": [0.2] * 128, "metadata": {"name": "Bob"}}])

        assert len(users2) == 1
        assert users2.get("user2") is not None
        assert users2.get("user1") is None  # Old data gone
