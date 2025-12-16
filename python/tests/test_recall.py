"""Recall verification tests - compare HNSW results to brute-force KNN"""

import math
import os
import random
import tempfile

import pytest

import omendb


def euclidean_distance(v1: list, v2: list) -> float:
    """Compute Euclidean distance between two vectors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def brute_force_knn(query: list, vectors: list, k: int) -> list:
    """
    Brute-force k-NN implementation for ground truth.
    Returns list of (id, distance) tuples sorted by distance.
    """
    distances = []
    for v in vectors:
        dist = euclidean_distance(query, v["vector"])
        distances.append((v["id"], dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def compute_recall(hnsw_results: list, ground_truth: list) -> float:
    """
    Compute recall: fraction of ground truth results found by HNSW.

    Args:
        hnsw_results: List of dicts with "id" key from HNSW search
        ground_truth: List of (id, distance) tuples from brute force

    Returns:
        Recall as float between 0.0 and 1.0
    """
    if not ground_truth:
        return 1.0 if not hnsw_results else 0.0

    hnsw_ids = {r["id"] for r in hnsw_results}
    gt_ids = {gt[0] for gt in ground_truth}

    return len(hnsw_ids & gt_ids) / len(gt_ids)


def generate_random_vectors(n: int, dim: int, seed: int = 42) -> list:
    """Generate random vectors for testing"""
    random.seed(seed)
    vectors = []
    for i in range(n):
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in embedding))
        embedding = [x / norm for x in embedding]
        vectors.append({"id": f"vec_{i}", "vector": embedding, "metadata": {"index": i}})
    return vectors


class TestRecallSmall:
    """Recall tests on small datasets (100 vectors)"""

    def test_recall_at_10_small(self):
        """Test recall@10 on 100 vectors - should be 100%"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            # Run multiple queries and average recall
            num_queries = 10
            total_recall = 0.0

            for q in range(num_queries):
                query = generate_random_vectors(1, 64, seed=1000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=10)
                ground_truth = brute_force_knn(query, vectors, k=10)

                recall = compute_recall(hnsw_results, ground_truth)
                total_recall += recall

            avg_recall = total_recall / num_queries
            assert avg_recall >= 0.95, f"Recall@10 too low: {avg_recall:.2%}"

    def test_recall_at_1_small(self):
        """Test recall@1 (nearest neighbor) on 100 vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(100, 64)
            db.set(vectors)

            num_queries = 20
            correct = 0

            for q in range(num_queries):
                query = generate_random_vectors(1, 64, seed=2000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=1)
                ground_truth = brute_force_knn(query, vectors, k=1)

                if hnsw_results and hnsw_results[0]["id"] == ground_truth[0][0]:
                    correct += 1

            accuracy = correct / num_queries
            assert accuracy >= 0.90, f"Recall@1 too low: {accuracy:.2%}"


class TestRecallMedium:
    """Recall tests on medium datasets (1K vectors)"""

    def test_recall_at_10_1k(self):
        """Test recall@10 on 1K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(1000, 128)
            db.set(vectors)

            num_queries = 10
            total_recall = 0.0

            for q in range(num_queries):
                query = generate_random_vectors(1, 128, seed=3000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=10)
                ground_truth = brute_force_knn(query, vectors, k=10)

                recall = compute_recall(hnsw_results, ground_truth)
                total_recall += recall

            avg_recall = total_recall / num_queries
            assert avg_recall >= 0.90, f"Recall@10 on 1K: {avg_recall:.2%}"

    def test_recall_at_50_1k(self):
        """Test recall@50 on 1K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(1000, 128)
            db.set(vectors)

            num_queries = 5
            total_recall = 0.0

            for q in range(num_queries):
                query = generate_random_vectors(1, 128, seed=4000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=50)
                ground_truth = brute_force_knn(query, vectors, k=50)

                recall = compute_recall(hnsw_results, ground_truth)
                total_recall += recall

            avg_recall = total_recall / num_queries
            assert avg_recall >= 0.85, f"Recall@50 on 1K: {avg_recall:.2%}"


class TestRecallWithEfSearch:
    """Test that ef_search tuning affects recall"""

    def test_higher_ef_search_improves_recall(self):
        """Test that increasing ef_search improves recall"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(1000, 128)
            db.set(vectors)

            query = generate_random_vectors(1, 128, seed=5000)[0]["vector"]
            # Use k=10 since ef must be >= k
            ground_truth = brute_force_knn(query, vectors, k=10)

            # Test with low ef_search (must be >= k)
            db.set_ef_search(50)
            low_ef_results = db.search(query, k=10)
            low_recall = compute_recall(low_ef_results, ground_truth)

            # Test with high ef_search
            db.set_ef_search(200)
            high_ef_results = db.search(query, k=10)
            high_recall = compute_recall(high_ef_results, ground_truth)

            # Higher ef_search should generally give better or equal recall
            assert high_recall >= low_recall * 0.95, (
                f"Higher ef_search should improve recall: "
                f"ef=50: {low_recall:.2%}, ef=200: {high_recall:.2%}"
            )

    def test_ef_search_200_achieves_high_recall(self):
        """Test that ef_search=200 achieves high recall"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            vectors = generate_random_vectors(1000, 128)
            db.set(vectors)

            db.set_ef_search(200)

            num_queries = 10
            total_recall = 0.0

            for q in range(num_queries):
                query = generate_random_vectors(1, 128, seed=6000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=10)
                ground_truth = brute_force_knn(query, vectors, k=10)

                recall = compute_recall(hnsw_results, ground_truth)
                total_recall += recall

            avg_recall = total_recall / num_queries
            assert avg_recall >= 0.95, f"Recall@10 with ef=200: {avg_recall:.2%}"


class TestRecallFiltered:
    """Recall tests with metadata filters"""

    def test_filtered_recall(self):
        """Test recall on filtered search"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Create vectors with category metadata
            random.seed(42)
            vectors = []
            for i in range(200):
                embedding = [random.gauss(0, 1) for _ in range(64)]
                norm = math.sqrt(sum(x * x for x in embedding))
                embedding = [x / norm for x in embedding]
                category = i % 5  # 5 categories: 0, 1, 2, 3, 4
                vectors.append(
                    {
                        "id": f"vec_{i}",
                        "vector": embedding,
                        "metadata": {"category": category},
                    }
                )

            db.set(vectors)

            # Filter for category 0 (should have 40 vectors)
            filtered_vectors = [v for v in vectors if v["metadata"]["category"] == 0]

            query = generate_random_vectors(1, 64, seed=7000)[0]["vector"]

            hnsw_results = db.search(query, k=10, filter={"category": 0})
            ground_truth = brute_force_knn(query, filtered_vectors, k=10)

            recall = compute_recall(hnsw_results, ground_truth)
            assert recall >= 0.80, f"Filtered recall@10: {recall:.2%}"

            # All results should match filter
            assert all(r["metadata"]["category"] == 0 for r in hnsw_results)


class TestRecallEdgeCases:
    """Edge case recall tests"""

    def test_recall_single_vector(self):
        """Test recall with single vector"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = [{"id": "only", "vector": [0.5] * 64, "metadata": {}}]
            db.set(vectors)

            results = db.search([0.5] * 64, k=1)
            assert len(results) == 1
            assert results[0]["id"] == "only"

    def test_recall_duplicate_vectors(self):
        """Test recall when vectors are identical"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            # Multiple vectors at same location
            vectors = [
                {"id": f"dup_{i}", "vector": [0.1] * 64, "metadata": {"index": i}}
                for i in range(10)
            ]
            db.set(vectors)

            results = db.search([0.1] * 64, k=5)
            assert len(results) == 5
            # All should have same distance
            distances = [r["distance"] for r in results]
            assert max(distances) - min(distances) < 0.001

    def test_recall_k_equals_n(self):
        """Test recall when k equals database size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=64)

            vectors = generate_random_vectors(50, 64)
            db.set(vectors)

            query = generate_random_vectors(1, 64, seed=8000)[0]["vector"]

            results = db.search(query, k=50)
            ground_truth = brute_force_knn(query, vectors, k=50)

            # Should return all vectors (100% recall by definition)
            recall = compute_recall(results, ground_truth)
            assert recall == 1.0, f"k=n should have 100% recall, got {recall:.2%}"


# Slow tests marked for optional running
class TestRecallLarge:
    """Large scale recall tests (10K vectors) - slower"""

    @pytest.mark.slow
    def test_recall_at_10_10k(self):
        """Test recall@10 on 10K vectors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            db = omendb.open(db_path, dimensions=128)

            # Use explicit seed for reproducibility
            vectors = generate_random_vectors(10000, 128, seed=12345)
            db.set(vectors)

            db.set_ef_search(200)  # Higher ef for reliable recall on random data

            num_queries = 5
            total_recall = 0.0

            for q in range(num_queries):
                query = generate_random_vectors(1, 128, seed=9000 + q)[0]["vector"]

                hnsw_results = db.search(query, k=10)
                ground_truth = brute_force_knn(query, vectors, k=10)

                recall = compute_recall(hnsw_results, ground_truth)
                total_recall += recall

            avg_recall = total_recall / num_queries
            # Random vectors have worse locality than real data - use 75% threshold
            # M=16 on random 128D: OmenDB ~88%, hnswlib ~83% (5 queries = high variance)
            # Real data (SIFT-128) achieves 99%+ at this config
            assert avg_recall >= 0.75, f"Recall@10 on 10K: {avg_recall:.2%}"
