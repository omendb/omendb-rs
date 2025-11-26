"""Tests for ef_search runtime tuning API"""

import pytest
import omendb
import tempfile
import os
import random
import math


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list:
    """Generate random vectors for testing"""
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]
        vectors.append({
            "id": f"vec_{i}",
            "embedding": embedding,
            "metadata": {"index": i}
        })
    return vectors


class TestEfSearchBasic:
    """Basic ef_search API tests"""

    def test_get_ef_search_empty_db(self):
        """Test get_ef_search on empty database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Empty db returns default ef_search (100)
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

            # Setting ef_search on empty db - should store it for later
            db.set_ef_search(150)
            # get_ef_search may return None before index exists
            # but should apply once vectors are inserted

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # After insert, ef_search should be applied
            ef = db.get_ef_search()
            assert ef == 150


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
            results = db.search(vectors[0]["embedding"], k=1)
            assert len(results) == 1

    def test_ef_search_vs_k_constraint(self):
        """Test that ef >= k is enforced during search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Set low ef_search
            db.set_ef_search(5)

            # Search with k > ef_search should fail
            query = vectors[0]["embedding"]
            with pytest.raises(RuntimeError, match="ef"):
                db.search(query, k=10)  # k=10 > ef=5

    def test_ef_search_equals_k(self):
        """Test that ef = k is allowed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            db.set_ef_search(10)
            results = db.search(vectors[0]["embedding"], k=10)
            assert len(results) == 10


class TestEfSearchPerformance:
    """Test ef_search impact on performance"""

    def test_lower_ef_faster(self):
        """Test that lower ef_search is faster (less work)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            # Insert enough vectors to see timing difference
            vectors = generate_random_vectors(5000, 128)
            db.set(vectors)

            query = generate_random_vectors(1, 128, seed=9000)[0]["embedding"]

            import time

            # Time with low ef_search
            db.set_ef_search(20)
            start = time.time()
            for _ in range(10):
                db.search(query, k=10)
            low_ef_time = time.time() - start

            # Time with high ef_search
            db.set_ef_search(200)
            start = time.time()
            for _ in range(10):
                db.search(query, k=10)
            high_ef_time = time.time() - start

            # High ef should take more time (exploring more nodes)
            # Allow some variance - just check high isn't faster
            assert high_ef_time >= low_ef_time * 0.5, (
                f"High ef ({high_ef_time:.3f}s) should not be faster than "
                f"low ef ({low_ef_time:.3f}s)"
            )


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

            db.save()

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
                vectors.append({
                    "id": f"vec_{i}",
                    "embedding": embedding,
                    "metadata": {"group": i % 10}
                })
            db.set(vectors)

            query = vectors[0]["embedding"]

            # Set high ef_search
            db.set_ef_search(100)
            high_ef_results = db.search(
                query,
                k=10,
                filter={"group": {"$eq": 0}}
            )

            # Set lower ef_search (but still >= k)
            db.set_ef_search(20)
            low_ef_results = db.search(
                query,
                k=10,
                filter={"group": {"$eq": 0}}
            )

            # Both should return results
            assert len(high_ef_results) > 0
            assert len(low_ef_results) > 0

            # All results should match filter
            assert all(r["metadata"]["group"] == 0 for r in high_ef_results)
            assert all(r["metadata"]["group"] == 0 for r in low_ef_results)
