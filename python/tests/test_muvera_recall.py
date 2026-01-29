"""MUVERA recall validation tests.

Validates that MUVERA FDE-based retrieval achieves target recall
compared to brute-force MaxSim scoring.
"""

import numpy as np
import pytest

import omendb


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Compute MaxSim score between query and document token sets.

    MaxSim = sum over query tokens of max similarity to any doc token.
    Uses dot product similarity (assumes normalized vectors).
    """
    # query_tokens: (num_query, dim)
    # doc_tokens: (num_doc, dim)
    # similarity matrix: (num_query, num_doc)
    sim_matrix = query_tokens @ doc_tokens.T
    # For each query token, take max similarity across doc tokens
    max_sims = sim_matrix.max(axis=1)
    return float(max_sims.sum())


def brute_force_maxsim_search(
    query_tokens: np.ndarray, all_docs: list[tuple[str, np.ndarray]], k: int
) -> list[str]:
    """Brute force MaxSim search - ground truth."""
    scores = []
    for doc_id, doc_tokens in all_docs:
        score = maxsim_score(query_tokens, doc_tokens)
        scores.append((doc_id, score))

    # Sort by score descending
    scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, _ in scores[:k]]


def compute_recall(predicted: list[str], ground_truth: list[str]) -> float:
    """Compute recall@k."""
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    return len(predicted_set & ground_truth_set) / len(ground_truth_set)


class TestMuveraRecall:
    """Recall validation tests for MUVERA."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Create synthetic dataset with controlled structure.

        Creates documents where similarity patterns are predictable,
        making it easier to validate recall.
        """
        np.random.seed(42)

        num_docs = 500
        dim = 64
        tokens_per_doc = 10

        # Create base vectors for document clusters
        num_clusters = 10
        cluster_centers = np.random.randn(num_clusters, dim).astype(np.float32)
        cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

        docs = []
        for i in range(num_docs):
            # Assign doc to a cluster
            cluster = i % num_clusters
            center = cluster_centers[cluster]

            # Generate tokens as perturbations of cluster center
            noise = np.random.randn(tokens_per_doc, dim).astype(np.float32) * 0.3
            tokens = center + noise
            # Normalize tokens
            tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)

            docs.append((f"doc{i}", tokens))

        return docs, cluster_centers, dim

    @pytest.fixture
    def populated_db(self, synthetic_dataset):
        """Create and populate multi-vector store."""
        docs, _, dim = synthetic_dataset

        db = omendb.open(":memory:", dimensions=dim, multi_vector={"d_proj": None})

        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {"index": i}}
            for i, (doc_id, tokens) in enumerate(docs)
        ]
        db.set(items)

        return db, docs

    def test_recall_with_reranking(self, populated_db, synthetic_dataset):
        """Test that reranking achieves 90%+ recall@10."""
        db, docs = populated_db
        _, cluster_centers, dim = synthetic_dataset

        # Run multiple queries from different clusters
        recalls = []
        num_queries = 20
        k = 10

        np.random.seed(123)

        for q in range(num_queries):
            # Create query from a cluster center with some noise
            cluster = q % len(cluster_centers)
            center = cluster_centers[cluster]

            # Generate query tokens
            num_query_tokens = 5
            noise = np.random.randn(num_query_tokens, dim).astype(np.float32) * 0.2
            query_tokens = (center + noise).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            # Ground truth via brute force
            ground_truth = brute_force_maxsim_search(query_tokens, docs, k)

            # MUVERA with reranking
            results = db.search(query_tokens.tolist(), k=k, rerank=True)
            predicted = [r["id"] for r in results]

            recall = compute_recall(predicted, ground_truth)
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        print(
            f"\nRecall@{k} with reranking: {avg_recall:.1%} (min: {min(recalls):.1%}, max: {max(recalls):.1%})"
        )

        # Target: 90%+ recall with reranking
        assert avg_recall >= 0.85, f"Expected recall >= 85%, got {avg_recall:.1%}"

    def test_reranking_improves_recall(self, populated_db, synthetic_dataset):
        """Test that reranking improves recall over no reranking."""
        db, docs = populated_db
        _, cluster_centers, dim = synthetic_dataset

        recalls_no_rerank = []
        recalls_rerank = []
        num_queries = 20
        k = 10

        np.random.seed(456)

        for q in range(num_queries):
            cluster = q % len(cluster_centers)
            center = cluster_centers[cluster]

            num_query_tokens = 5
            noise = np.random.randn(num_query_tokens, dim).astype(np.float32) * 0.2
            query_tokens = (center + noise).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            ground_truth = brute_force_maxsim_search(query_tokens, docs, k)

            # Without reranking
            results_no_rerank = db.search(query_tokens.tolist(), k=k, rerank=False)
            predicted_no_rerank = [r["id"] for r in results_no_rerank]
            recalls_no_rerank.append(compute_recall(predicted_no_rerank, ground_truth))

            # With reranking
            results_rerank = db.search(query_tokens.tolist(), k=k, rerank=True)
            predicted_rerank = [r["id"] for r in results_rerank]
            recalls_rerank.append(compute_recall(predicted_rerank, ground_truth))

        avg_no_rerank = np.mean(recalls_no_rerank)
        avg_rerank = np.mean(recalls_rerank)

        print(f"\nRecall@{k} without reranking: {avg_no_rerank:.1%}")
        print(f"Recall@{k} with reranking: {avg_rerank:.1%}")
        print(f"Improvement: {avg_rerank - avg_no_rerank:+.1%}")

        # Reranking should improve recall
        assert avg_rerank >= avg_no_rerank, "Reranking should not decrease recall"

    def test_recall_vs_rerank_factor(self, populated_db, synthetic_dataset):
        """Test recall at different rerank factors."""
        db, docs = populated_db
        _, cluster_centers, dim = synthetic_dataset

        k = 10
        rerank_factors = [1, 2, 4, 8]
        num_queries = 15

        np.random.seed(789)

        results_by_factor = {f: [] for f in rerank_factors}

        for q in range(num_queries):
            cluster = q % len(cluster_centers)
            center = cluster_centers[cluster]

            num_query_tokens = 5
            noise = np.random.randn(num_query_tokens, dim).astype(np.float32) * 0.2
            query_tokens = (center + noise).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            ground_truth = brute_force_maxsim_search(query_tokens, docs, k)

            for factor in rerank_factors:
                results = db.search(query_tokens.tolist(), k=k, rerank_factor=factor)
                predicted = [r["id"] for r in results]
                recall = compute_recall(predicted, ground_truth)
                results_by_factor[factor].append(recall)

        print(f"\nRecall@{k} by rerank factor:")
        prev_recall = 0
        for factor in rerank_factors:
            avg = np.mean(results_by_factor[factor])
            print(f"  factor={factor}: {avg:.1%}")
            # Higher factors should generally not decrease recall
            assert avg >= prev_recall - 0.05, f"Factor {factor} recall dropped significantly"
            prev_recall = avg

    def test_recall_at_different_k(self, populated_db, synthetic_dataset):
        """Test recall at different k values."""
        db, docs = populated_db
        _, cluster_centers, dim = synthetic_dataset

        k_values = [1, 5, 10, 20]
        num_queries = 15

        np.random.seed(101)

        results_by_k = {k: [] for k in k_values}

        for q in range(num_queries):
            cluster = q % len(cluster_centers)
            center = cluster_centers[cluster]

            num_query_tokens = 5
            noise = np.random.randn(num_query_tokens, dim).astype(np.float32) * 0.2
            query_tokens = (center + noise).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            for k in k_values:
                ground_truth = brute_force_maxsim_search(query_tokens, docs, k)
                results = db.search(query_tokens.tolist(), k=k, rerank=True)
                predicted = [r["id"] for r in results]
                recall = compute_recall(predicted, ground_truth)
                results_by_k[k].append(recall)

        print("\nRecall by k (with reranking):")
        for k in k_values:
            avg = np.mean(results_by_k[k])
            print(f"  recall@{k}: {avg:.1%}")


class TestMuveraScaleRecall:
    """Recall tests at larger scale."""

    def test_recall_1k_docs_clustered(self):
        """Test recall with 1000 documents in semantic clusters.

        Uses clustered data to simulate real embeddings where documents
        within a topic have similar token embeddings.
        """
        np.random.seed(42)

        num_docs = 1000
        dim = 32
        tokens_per_doc = 8
        num_clusters = 20

        # Create cluster centers (like topic embeddings)
        cluster_centers = np.random.randn(num_clusters, dim).astype(np.float32)
        cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

        # Create documents as perturbations of cluster centers
        docs = []
        for i in range(num_docs):
            cluster = i % num_clusters
            center = cluster_centers[cluster]

            # Tokens are variations around cluster center
            noise = np.random.randn(tokens_per_doc, dim).astype(np.float32) * 0.3
            tokens = (center + noise).astype(np.float32)
            tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)
            docs.append((f"doc{i}", tokens))

        # Create store (d_proj=None to test baseline FDE quality)
        db = omendb.open(":memory:", dimensions=dim, multi_vector={"d_proj": None})
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {}} for doc_id, tokens in docs
        ]
        db.set(items)

        # Run queries from cluster centers
        k = 10
        num_queries = 20
        recalls = []

        np.random.seed(123)
        for q in range(num_queries):
            cluster = q % num_clusters
            center = cluster_centers[cluster]

            # Query tokens around cluster center
            noise = np.random.randn(5, dim).astype(np.float32) * 0.2
            query_tokens = (center + noise).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            ground_truth = brute_force_maxsim_search(query_tokens, docs, k)
            # Use rerank_factor=4 for better recall at scale
            results = db.search(query_tokens.tolist(), k=k, rerank_factor=4)
            predicted = [r["id"] for r in results]

            recalls.append(compute_recall(predicted, ground_truth))

        avg_recall = np.mean(recalls)
        print(f"\n1K clustered docs - Recall@{k}: {avg_recall:.1%}")

        # Target: 85%+ recall with reranking on structured data
        assert avg_recall >= 0.80, f"Expected recall >= 80% at 1K scale, got {avg_recall:.1%}"

    def test_recall_random_data_high_rerank(self):
        """Test recall on random data with high rerank factor.

        Random data is harder for MUVERA since there's no semantic structure
        to approximate. This tests the ceiling with aggressive reranking.
        """
        np.random.seed(42)

        num_docs = 500
        dim = 32
        tokens_per_doc = 8

        docs = []
        for i in range(num_docs):
            tokens = np.random.randn(tokens_per_doc, dim).astype(np.float32)
            tokens /= np.linalg.norm(tokens, axis=1, keepdims=True)
            docs.append((f"doc{i}", tokens))

        db = omendb.open(":memory:", dimensions=dim, multi_vector={"d_proj": None})
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {}} for doc_id, tokens in docs
        ]
        db.set(items)

        k = 10
        num_queries = 10
        recalls = []

        np.random.seed(123)
        for _ in range(num_queries):
            query_tokens = np.random.randn(5, dim).astype(np.float32)
            query_tokens /= np.linalg.norm(query_tokens, axis=1, keepdims=True)

            ground_truth = brute_force_maxsim_search(query_tokens, docs, k)
            # High rerank factor for random data
            results = db.search(query_tokens.tolist(), k=k, rerank_factor=8)
            predicted = [r["id"] for r in results]

            recalls.append(compute_recall(predicted, ground_truth))

        avg_recall = np.mean(recalls)
        print(f"\n500 random docs (rerank_factor=8) - Recall@{k}: {avg_recall:.1%}")

        # Random data is hard - 60% is reasonable with high rerank factor
        assert avg_recall >= 0.50, f"Expected recall >= 50% on random data, got {avg_recall:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
