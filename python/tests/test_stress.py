"""Stress tests for OmenDB - large scale operations"""

import math
import os
import random
import tempfile
import time

import pytest

import omendb


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list:
    """Generate random normalized vectors"""
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]
        vectors.append(
            {"id": f"vec_{i}", "vector": embedding, "metadata": {"index": i, "group": i % 100}}
        )
    return vectors


class TestLargeScaleInsert:
    """Test insertion at scale"""

    @pytest.mark.slow
    def test_insert_10k_vectors(self):
        """Test inserting 10K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(10000, 128)

            start = time.time()
            db.set(vectors)
            elapsed = time.time() - start

            assert len(db) == 10000
            print(f"\n10K insert: {elapsed:.2f}s ({10000 / elapsed:.0f} vec/s)")

    @pytest.mark.slow
    def test_insert_100k_vectors(self):
        """Test inserting 100K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(100000, 128)

            start = time.time()
            db.set(vectors)
            elapsed = time.time() - start

            assert len(db) == 100000
            print(f"\n100K insert: {elapsed:.2f}s ({100000 / elapsed:.0f} vec/s)")


class TestLargeScaleSearch:
    """Test search at scale"""

    @pytest.mark.slow
    def test_search_10k_vectors(self):
        """Test search performance at 10K scale"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(10000, 128)
            db.set(vectors)

            # Run multiple queries
            num_queries = 100
            queries = [
                generate_random_vectors(1, 128, seed=1000 + i)[0]["vector"]
                for i in range(num_queries)
            ]

            start = time.time()
            for query in queries:
                results = db.search(query, k=10)
                assert len(results) == 10
            elapsed = time.time() - start

            qps = num_queries / elapsed
            print(f"\n10K search: {qps:.0f} QPS")
            assert qps > 100  # Minimum expected performance

    @pytest.mark.slow
    def test_search_100k_vectors(self):
        """Test search performance at 100K scale"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(100000, 128)
            db.set(vectors)

            num_queries = 100
            queries = [
                generate_random_vectors(1, 128, seed=2000 + i)[0]["vector"]
                for i in range(num_queries)
            ]

            start = time.time()
            for query in queries:
                results = db.search(query, k=10)
                assert len(results) == 10
            elapsed = time.time() - start

            qps = num_queries / elapsed
            print(f"\n100K search: {qps:.0f} QPS")
            assert qps > 50  # Minimum expected at 100K


class TestBatchOperations:
    """Test batch operations"""

    @pytest.mark.slow
    def test_incremental_inserts(self):
        """Test many small batch inserts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Insert in small batches
            batch_size = 100
            num_batches = 100  # 10K total

            start = time.time()
            for batch_num in range(num_batches):
                vectors = generate_random_vectors(batch_size, 64, seed=batch_num * batch_size)
                # Update IDs to be unique across batches
                for i, v in enumerate(vectors):
                    v["id"] = f"vec_{batch_num * batch_size + i}"
                db.set(vectors)
            elapsed = time.time() - start

            assert len(db) == batch_size * num_batches
            print(f"\nIncremental insert ({num_batches} batches): {elapsed:.2f}s")

    @pytest.mark.slow
    def test_set_updates(self):
        """Test updating existing vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Initial insert
            vectors = generate_random_vectors(1000, 64, seed=1)
            db.set(vectors)
            assert len(db) == 1000

            # Update all vectors with new embeddings
            updated = generate_random_vectors(1000, 64, seed=2)
            for i, v in enumerate(updated):
                v["id"] = f"vec_{i}"  # Same IDs

            start = time.time()
            db.set(updated)
            elapsed = time.time() - start

            # Count should remain same (updates, not inserts)
            assert len(db) == 1000
            print(f"\n1K set updates: {elapsed:.2f}s")


class TestFilteredSearchScale:
    """Test filtered search at scale"""

    @pytest.mark.slow
    def test_filtered_search_10k(self):
        """Test filtered search at 10K scale"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(10000, 128)
            db.set(vectors)

            num_queries = 50
            queries = [
                generate_random_vectors(1, 128, seed=3000 + i)[0]["vector"]
                for i in range(num_queries)
            ]

            # Test with different selectivities
            for group_filter in [0, 50]:  # 1% and 1% selectivity
                start = time.time()
                for query in queries:
                    results = db.search(query, k=10, filter={"group": group_filter})
                    # Should find some results (100 vectors per group)
                    assert len(results) <= 10
                    assert all(r["metadata"]["group"] == group_filter for r in results)
                elapsed = time.time() - start

                qps = num_queries / elapsed
                print(f"\n10K filtered (group={group_filter}): {qps:.0f} QPS")


class TestPersistenceScale:
    """Test persistence at scale"""

    @pytest.mark.slow
    def test_save_load_10k(self):
        """Test save/load with 10K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(10000, 128)
            db.set(vectors)

            # Save
            start = time.time()
            db.flush()
            save_time = time.time() - start
            print(f"\n10K save: {save_time:.2f}s")
            del db

            # Load in new instance
            start = time.time()
            db2 = omendb.open(db_path, dimensions=128)
            load_time = time.time() - start
            print(f"10K load: {load_time:.2f}s")

            assert len(db2) == 10000

            # Verify search works after reload
            query = vectors[0]["vector"]
            results = db2.search(query, k=5)
            assert len(results) == 5
            assert results[0]["id"] == "vec_0"


class TestDeleteScale:
    """Test deletion at scale"""

    @pytest.mark.slow
    def test_delete_half(self):
        """Test deleting half the vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(2000, 64)
            db.set(vectors)
            assert len(db) == 2000

            # Delete even-indexed vectors
            ids_to_delete = [f"vec_{i}" for i in range(0, 2000, 2)]

            start = time.time()
            db.delete(ids_to_delete)
            elapsed = time.time() - start

            assert len(db) == 1000  # Half remaining
            print(f"\nDelete 1K: {elapsed:.2f}s")

            # Verify deleted vectors not in search results
            query = vectors[0]["vector"]  # Query for deleted vector
            results = db.search(query, k=10)
            deleted_ids = set(ids_to_delete)
            assert all(r["id"] not in deleted_ids for r in results)


class TestEdgeCasesScale:
    """Edge cases at scale"""

    @pytest.mark.slow
    def test_high_dimension(self):
        """Test with high-dimensional vectors (1536 - OpenAI embedding size)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=1536)

            # Smaller count due to memory
            vectors = generate_random_vectors(1000, 1536)
            db.set(vectors)

            query = vectors[0]["vector"]
            results = db.search(query, k=10)
            assert len(results) == 10
            assert results[0]["id"] == "vec_0"

    @pytest.mark.slow
    def test_many_small_batches(self):
        """Test many tiny batches (simulates real-time inserts)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # 1000 single-vector inserts
            for i in range(1000):
                vector = generate_random_vectors(1, 64, seed=i)
                vector[0]["id"] = f"vec_{i}"
                db.set(vector)

            assert len(db) == 1000

            # Verify search still works
            query = generate_random_vectors(1, 64, seed=500)[0]["vector"]
            results = db.search(query, k=5)
            assert len(results) == 5
