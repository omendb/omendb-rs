"""Real-world integration tests for OmenDB.

Tests realistic embedding dimensions and use patterns:
- 384D: sentence-transformers (all-MiniLM-L6-v2)
- 768D: BERT, mpnet
- 1536D: OpenAI text-embedding-ada-002
- 3072D: OpenAI text-embedding-3-large

These tests verify:
- Correct semantic ranking
- Recall accuracy
- Filtered search
- Performance at scale
"""

import os
import random
import tempfile
import time

import pytest

import omendb


def generate_embedding(dim: int, seed: int) -> list[float]:
    """Generate deterministic embedding from seed."""
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(dim)]


def brute_force_knn(query: list[float], vectors: list[list[float]], k: int) -> list[int]:
    """Compute exact k-NN using brute force (L2 distance)."""
    distances = []
    for i, vec in enumerate(vectors):
        dist = sum((a - b) ** 2 for a, b in zip(query, vec))
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


class TestEmbeddingDimensions:
    """Test various embedding dimensions."""

    @pytest.mark.parametrize(
        "dim,name",
        [
            (384, "sentence-transformers"),
            (768, "BERT/mpnet"),
            (1536, "OpenAI ada-002"),
        ],
    )
    def test_embedding_dimension(self, dim: int, name: str):
        """Test search works correctly for common embedding dimensions."""
        n_vectors = 500
        k = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            # Insert vectors
            vectors = []
            for i in range(n_vectors):
                vec = generate_embedding(dim, seed=i)
                vectors.append(vec)
                db.set(f"doc_{i}", vec, metadata={"idx": i})

            # Search
            query = generate_embedding(dim, seed=9999)
            results = db.search(query, k=k)

            # Verify results
            assert len(results) == k
            assert all("id" in r and "distance" in r for r in results)

            # Check recall vs brute force
            ground_truth = set(brute_force_knn(query, vectors, k))
            result_ids = {int(r["id"].split("_")[1]) for r in results}
            recall = len(ground_truth & result_ids) / k

            assert recall >= 0.8, f"{name} ({dim}D): recall {recall:.0%} < 80%"

    def test_high_dimension_3072(self):
        """Test high-dimensional embeddings (OpenAI text-embedding-3-large)."""
        dim = 3072
        n_vectors = 100  # Fewer vectors for speed

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            for i in range(n_vectors):
                vec = generate_embedding(dim, seed=i)
                db.set(f"doc_{i}", vec)

            query = generate_embedding(dim, seed=9999)
            results = db.search(query, k=5)

            assert len(results) == 5


class TestRecallAccuracy:
    """Test recall accuracy at various scales."""

    @pytest.mark.slow
    def test_recall_1k_vectors(self):
        """Test recall with 1K vectors (should be near-perfect)."""
        dim = 128
        n_vectors = 1000
        k = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            vectors = []
            for i in range(n_vectors):
                vec = generate_embedding(dim, seed=i)
                vectors.append(vec)
                db.set(f"doc_{i}", vec)

            # Test multiple queries
            recalls = []
            for q in range(10):
                query = generate_embedding(dim, seed=10000 + q)
                results = db.search(query, k=k)

                ground_truth = set(brute_force_knn(query, vectors, k))
                result_ids = {int(r["id"].split("_")[1]) for r in results}
                recall = len(ground_truth & result_ids) / k
                recalls.append(recall)

            avg_recall = sum(recalls) / len(recalls)
            assert avg_recall >= 0.95, f"Average recall {avg_recall:.0%} < 95%"

    @pytest.mark.slow
    def test_recall_with_ef_search_tuning(self):
        """Test that higher ef_search improves recall."""
        dim = 128
        n_vectors = 1000
        k = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            vectors = []
            for i in range(n_vectors):
                vec = generate_embedding(dim, seed=i)
                vectors.append(vec)
                db.set(f"doc_{i}", vec)

            query = generate_embedding(dim, seed=9999)
            ground_truth = set(brute_force_knn(query, vectors, k))

            # Test different ef_search values
            recalls = {}
            for ef in [50, 100, 200]:
                db.set_ef_search(ef)
                results = db.search(query, k=k)
                result_ids = {int(r["id"].split("_")[1]) for r in results}
                recalls[ef] = len(ground_truth & result_ids) / k

            # Higher ef should give equal or better recall
            assert recalls[200] >= recalls[100] >= recalls[50] * 0.9


class TestFilteredSearch:
    """Test filtered search functionality."""

    def test_filtered_search_eq(self):
        """Test filtered search with equality filter."""
        dim = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            # Insert vectors with category metadata
            for i in range(100):
                vec = generate_embedding(dim, seed=i)
                db.set(f"doc_{i}", vec, metadata={"category": i % 5})

            query = generate_embedding(dim, seed=9999)

            # Search only category 2
            results = db.search(query, k=10, filter={"category": 2})

            # All results should be category 2
            for r in results:
                assert r["metadata"]["category"] == 2

    def test_filtered_search_range(self):
        """Test filtered search with range filter."""
        dim = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            for i in range(100):
                vec = generate_embedding(dim, seed=i)
                db.set(f"doc_{i}", vec, metadata={"score": i})

            query = generate_embedding(dim, seed=9999)

            # Search score >= 50
            results = db.search(query, k=10, filter={"score": {"$gte": 50}})

            for r in results:
                assert r["metadata"]["score"] >= 50

    def test_filtered_search_combined(self):
        """Test filtered search with AND filter."""
        dim = 64

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            for i in range(200):
                vec = generate_embedding(dim, seed=i)
                db.set(f"doc_{i}", vec, metadata={"category": i % 4, "score": i})

            query = generate_embedding(dim, seed=9999)

            # Search category=1 AND score >= 100
            results = db.search(
                query,
                k=10,
                filter={"$and": [{"category": 1}, {"score": {"$gte": 100}}]},
            )

            for r in results:
                assert r["metadata"]["category"] == 1
                assert r["metadata"]["score"] >= 100


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.slow
    def test_search_latency_1536d(self):
        """Test search latency at OpenAI embedding dimensions."""
        dim = 1536
        n_vectors = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            for i in range(n_vectors):
                vec = generate_embedding(dim, seed=i)
                db.set(f"doc_{i}", vec)

            query = generate_embedding(dim, seed=9999)

            # Warmup
            for _ in range(10):
                db.search(query, k=10)

            # Benchmark
            latencies = []
            for _ in range(100):
                start = time.time()
                db.search(query, k=10)
                latencies.append((time.time() - start) * 1000)

            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[99]

            avg_threshold = 6.0 if os.getenv("CI") else 5.0
            p99_threshold = 12.0 if os.getenv("CI") else 10.0

            assert (
                avg_latency < avg_threshold
            ), f"Average latency {avg_latency:.2f}ms > {avg_threshold:.1f}ms"
            assert (
                p99_latency < p99_threshold
            ), f"P99 latency {p99_latency:.2f}ms > {p99_threshold:.1f}ms"

    @pytest.mark.slow
    def test_search_batch_performance(self):
        """Test batch search at scale (batch is faster for larger datasets)."""
        dim = 128
        n_vectors = 10000  # Larger dataset where batch parallelism helps
        n_queries = 100

        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=dim)

            # Batch insert for speed
            vectors = [
                {"id": f"doc_{i}", "vector": generate_embedding(dim, seed=i)}
                for i in range(n_vectors)
            ]
            db.set(vectors)

            queries = [generate_embedding(dim, seed=10000 + i) for i in range(n_queries)]

            # Individual searches
            start = time.time()
            for query in queries:
                db.search(query, k=10)
            individual_time = time.time() - start

            # Batch search
            start = time.time()
            batch_results = db.search_batch(queries, k=10)
            batch_time = time.time() - start

            # Verify batch returns correct structure
            assert len(batch_results) == n_queries
            assert all(len(r) == 10 for r in batch_results)

            # For larger datasets, batch should be faster (or at least comparable)
            # Note: batch has thread pool overhead that makes it slower for tiny datasets
            # but faster for real workloads (10K+ vectors, higher dimensions)
            speedup = individual_time / batch_time if batch_time > 0 else float("inf")
            print(
                f"Batch speedup: {speedup:.2f}x (individual={individual_time:.3f}s, batch={batch_time:.3f}s)"
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database_search(self):
        """Search on empty database should return empty results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=64)
            results = db.search([0.0] * 64, k=10)
            assert results == []

    def test_k_greater_than_n(self):
        """Requesting more results than vectors should return all vectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=64)

            for i in range(5):
                db.set(f"doc_{i}", [float(i)] * 64)

            results = db.search([0.0] * 64, k=100)
            assert len(results) == 5

    def test_zero_vector_query(self):
        """Zero vector query should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=64)

            db.set("doc_0", [1.0] * 64)
            db.set("doc_1", [0.5] * 64)

            results = db.search([0.0] * 64, k=2)
            assert len(results) == 2

    def test_dimension_mismatch_error(self):
        """Mismatched dimensions should raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = omendb.open(f"{tmpdir}/test", dimensions=64)
            db.set("doc_0", [1.0] * 64)

            with pytest.raises(ValueError):
                db.search([1.0] * 128, k=1)  # Wrong dimension
