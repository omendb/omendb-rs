"""Tests for ef_search runtime tuning API"""

import math
import os
import random
import tempfile

import omendb

# ef_search API now implemented


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list:
    """Generate random vectors for testing"""
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]
        vectors.append({"id": f"vec_{i}", "vector": embedding, "metadata": {"index": i}})
    return vectors


class TestEfSearchBasic:
    """Basic ef_search API tests"""

    def test_get_ef_search_empty_db(self):
        """Test get_ef_search on empty database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Empty db returns the default value (100)
            ef = db.get_ef_search()
            assert ef == 100

    def test_get_ef_search_after_insert(self):
        """Test get_ef_search after inserting vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Should return default ef_search value
            ef = db.get_ef_search()
            assert ef is not None
            assert ef > 0

    def test_set_ef_search_basic(self):
        """Test setting ef_search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Set and verify
            db.set_ef_search(100)
            assert db.get_ef_search() == 100

            db.set_ef_search(50)
            assert db.get_ef_search() == 50

            db.set_ef_search(200)
            assert db.get_ef_search() == 200

    def test_set_ef_search_before_insert(self):
        """Test setting ef_search before inserting vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Setting ef_search on empty db now works
            db.set_ef_search(150)
            assert db.get_ef_search() == 150  # Value is stored

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # After insert, ef_search is still 150
            ef = db.get_ef_search()
            assert ef == 150

            # Changing ef_search still works
            db.set_ef_search(200)
            assert db.get_ef_search() == 200


class TestEfSearchConstraints:
    """Test ef_search constraints and validation"""

    def test_ef_search_small_values(self):
        """Test that very small ef_search values work"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Small ef_search values are accepted
            db.set_ef_search(1)
            assert db.get_ef_search() == 1

            # Can still search with k <= ef
            results = db.search(vectors[0]["vector"], k=1)
            assert len(results) == 1

    def test_ef_search_auto_clamp_to_k(self):
        """Test that ef is auto-clamped to k when ef < k (no error)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Set low ef_search
            db.set_ef_search(5)

            # Search with k > ef_search should work (ef auto-clamped to k)
            query = vectors[0]["vector"]
            results = db.search(query, k=10)  # k=10 > ef=5, auto-clamps ef to 10
            assert len(results) == 10

    def test_ef_search_equals_k(self):
        """Test that ef = k is allowed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            db.set_ef_search(10)
            results = db.search(vectors[0]["vector"], k=10)
            assert len(results) == 10


class TestEfSearchPersistence:
    """Test ef_search persistence across sessions"""

    def test_ef_search_not_persisted(self):
        """Test that ef_search setting is NOT persisted (runtime only)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")  # Persistent db

            # Create db and set ef_search
            db = omendb.open(db_path, dimensions=64)
            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            original_ef = db.get_ef_search()
            db.set_ef_search(50)
            assert db.get_ef_search() == 50

            db.flush()
            del db

            # Reopen - ef_search should be back to default
            db2 = omendb.open(db_path, dimensions=64)
            ef_after_reopen = db2.get_ef_search()

            # Should return to default, not the custom value
            assert ef_after_reopen == original_ef


class TestEfSearchWithFilters:
    """Test ef_search with filtered search"""

    def test_ef_search_affects_filtered(self):
        """Test that ef_search affects filtered search too"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Create vectors with labels
            random.seed(42)
            vectors = []
            for i in range(500):
                embedding = [random.gauss(0, 1) for _ in range(64)]
                norm = math.sqrt(sum(x * x for x in embedding))
                embedding = [x / norm for x in embedding]
                vectors.append(
                    {
                        "id": f"vec_{i}",
                        "vector": embedding,
                        "metadata": {"group": i % 10},
                    }
                )
            db.set(vectors)

            query = vectors[0]["vector"]

            # Set high ef_search
            db.set_ef_search(100)
            high_ef_results = db.search(query, k=10, filter={"group": 0})

            # Set lower ef_search (but still >= k)
            db.set_ef_search(20)
            low_ef_results = db.search(query, k=10, filter={"group": 0})

            # Both should return results
            assert len(high_ef_results) > 0
            assert len(low_ef_results) > 0

            # All results should match filter
            assert all(r["metadata"]["group"] == 0 for r in high_ef_results)
            assert all(r["metadata"]["group"] == 0 for r in low_ef_results)
