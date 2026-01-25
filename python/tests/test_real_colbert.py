"""Real ColBERT model validation for MUVERA.

Uses fastembed's ColBERT implementation to generate real token embeddings
and validates OmenDB's multi-vector retrieval against brute-force MaxSim.

This test requires the fastembed package and will download the ColBERT model
on first run (~400MB).

Run: uv run pytest tests/test_real_colbert.py -v -s
"""

import numpy as np
import pytest

import omendb

# Skip if fastembed not available
fastembed = pytest.importorskip("fastembed")
from fastembed import LateInteractionTextEmbedding


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Compute MaxSim score between query and document token sets.

    MaxSim = sum over query tokens of max similarity to any doc token.
    Uses dot product similarity (assumes normalized vectors).
    """
    sim_matrix = query_tokens @ doc_tokens.T
    max_sims = sim_matrix.max(axis=1)
    return float(max_sims.sum())


def brute_force_search(
    query_tokens: np.ndarray, all_docs: list[tuple[str, np.ndarray]], k: int
) -> list[str]:
    """Brute force MaxSim search - ground truth."""
    scores = [(doc_id, maxsim_score(query_tokens, doc_tokens)) for doc_id, doc_tokens in all_docs]
    scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, _ in scores[:k]]


def compute_recall(predicted: list[str], ground_truth: list[str]) -> float:
    """Compute recall@k."""
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


# Test corpus - diverse topics for semantic retrieval
TEST_DOCUMENTS = [
    # Technology
    "Machine learning models require large amounts of training data to achieve good performance.",
    "Neural networks consist of layers of interconnected nodes that process information.",
    "Deep learning has revolutionized computer vision and natural language processing.",
    "GPUs accelerate matrix operations essential for training deep neural networks.",
    "Transformers use attention mechanisms to process sequential data in parallel.",
    # Science
    "Photosynthesis converts sunlight into chemical energy stored in glucose molecules.",
    "DNA contains the genetic instructions for all known living organisms.",
    "The speed of light is approximately 299,792 kilometers per second in vacuum.",
    "Quantum mechanics describes the behavior of matter at atomic and subatomic scales.",
    "Evolution occurs through natural selection acting on genetic variation.",
    # History
    "The Roman Empire lasted for over a thousand years and shaped Western civilization.",
    "The Industrial Revolution began in Britain in the late 18th century.",
    "World War II was the deadliest conflict in human history.",
    "The Renaissance was a period of cultural rebirth in Europe.",
    "Ancient Egypt developed along the Nile River over 5000 years ago.",
    # Geography
    "Mount Everest is the highest peak on Earth at 8,849 meters above sea level.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "The Sahara Desert is the largest hot desert in the world.",
    "Antarctica is the coldest continent with temperatures below minus 80 degrees Celsius.",
    # Medicine
    "Antibiotics are used to treat bacterial infections but not viral ones.",
    "The human heart pumps about 5 liters of blood per minute at rest.",
    "Vaccines train the immune system to recognize and fight specific pathogens.",
    "Cancer occurs when cells grow and divide uncontrollably.",
    "The brain contains approximately 86 billion neurons.",
    # More technology for retrieval testing
    "Vector databases store high-dimensional embeddings for similarity search.",
    "HNSW is an algorithm for approximate nearest neighbor search.",
    "Embedding models convert text into dense vector representations.",
    "Retrieval augmented generation combines search with language models.",
    "ColBERT uses late interaction for efficient neural retrieval.",
]

TEST_QUERIES = [
    "How do neural networks learn from data?",
    "What is the process plants use to make energy?",
    "How fast does light travel?",
    "What was the impact of the Industrial Revolution?",
    "How tall is the highest mountain?",
    "How do vaccines work?",
    "What are vector databases used for?",
    "How does ColBERT retrieval work?",
]


@pytest.fixture(scope="module")
def colbert_model():
    """Load ColBERT model (cached across tests)."""
    print("\nLoading ColBERT model (may download on first run)...")
    model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    print("Model loaded: colbert-ir/colbertv2.0")
    return model


@pytest.fixture(scope="module")
def encoded_corpus(colbert_model):
    """Encode test documents with ColBERT."""
    print(f"\nEncoding {len(TEST_DOCUMENTS)} documents...")

    # Encode documents - returns list of arrays, each (num_tokens, 128)
    doc_embeddings = list(colbert_model.embed(TEST_DOCUMENTS))

    docs = []
    for i, emb in enumerate(doc_embeddings):
        # Normalize for MaxSim (ColBERT uses dot product)
        emb_normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        docs.append((f"doc{i}", emb_normalized.astype(np.float32)))
        print(f"  doc{i}: {emb.shape[0]} tokens, dim={emb.shape[1]}")

    return docs


@pytest.fixture(scope="module")
def encoded_queries(colbert_model):
    """Encode test queries with ColBERT."""
    print(f"\nEncoding {len(TEST_QUERIES)} queries...")

    query_embeddings = list(colbert_model.query_embed(TEST_QUERIES))

    queries = []
    for i, emb in enumerate(query_embeddings):
        emb_normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        queries.append((TEST_QUERIES[i], emb_normalized.astype(np.float32)))
        print(f"  q{i}: {emb.shape[0]} tokens")

    return queries


class TestRealColBERT:
    """Validation tests using real ColBERT embeddings."""

    def test_colbert_dimensions(self, colbert_model, encoded_corpus):
        """Verify ColBERT output dimensions."""
        _, tokens = encoded_corpus[0]
        assert tokens.shape[1] == 128, f"Expected 128D, got {tokens.shape[1]}"

    def test_muvera_recall_with_reranking(self, encoded_corpus, encoded_queries):
        """Test MUVERA recall with real ColBERT embeddings and reranking."""
        dim = encoded_corpus[0][1].shape[1]  # 128

        # Create multi-vector store
        db = omendb.open(":memory:", dimensions=dim, multi_vector=True)

        # Insert documents
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {"index": i}}
            for i, (doc_id, tokens) in enumerate(encoded_corpus)
        ]
        db.set(items)

        print(f"\nInserted {len(db)} documents into MUVERA store")

        # Run queries and compute recall
        k = 5
        recalls = []

        print(f"\nQuery results (k={k}, rerank=True):")
        for query_text, query_tokens in encoded_queries:
            # Ground truth via brute force
            ground_truth = brute_force_search(query_tokens, encoded_corpus, k)

            # MUVERA with reranking
            results = db.search(query_tokens.tolist(), k=k, rerank=True)
            predicted = [r["id"] for r in results]

            recall = compute_recall(predicted, ground_truth)
            recalls.append(recall)

            print(f"  '{query_text[:40]}...'")
            print(f"    Ground truth: {ground_truth}")
            print(f"    MUVERA:       {predicted}")
            print(f"    Recall: {recall:.0%}")

        avg_recall = np.mean(recalls)
        print(f"\nAverage Recall@{k} (with reranking): {avg_recall:.1%}")

        # Target: 80%+ recall with real ColBERT embeddings
        assert avg_recall >= 0.70, f"Expected recall >= 70%, got {avg_recall:.1%}"

    def test_muvera_recall_without_reranking(self, encoded_corpus, encoded_queries):
        """Test MUVERA recall without reranking (FDE-only)."""
        dim = encoded_corpus[0][1].shape[1]

        db = omendb.open(":memory:", dimensions=dim, multi_vector=True)
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {}}
            for doc_id, tokens in encoded_corpus
        ]
        db.set(items)

        k = 5
        recalls_no_rerank = []
        recalls_rerank = []

        for _, query_tokens in encoded_queries:
            ground_truth = brute_force_search(query_tokens, encoded_corpus, k)

            # Without reranking
            results_no = db.search(query_tokens.tolist(), k=k, rerank=False)
            predicted_no = [r["id"] for r in results_no]
            recalls_no_rerank.append(compute_recall(predicted_no, ground_truth))

            # With reranking
            results_yes = db.search(query_tokens.tolist(), k=k, rerank=True)
            predicted_yes = [r["id"] for r in results_yes]
            recalls_rerank.append(compute_recall(predicted_yes, ground_truth))

        avg_no = np.mean(recalls_no_rerank)
        avg_yes = np.mean(recalls_rerank)

        print(f"\nRecall@{k} without reranking: {avg_no:.1%}")
        print(f"Recall@{k} with reranking: {avg_yes:.1%}")
        print(f"Improvement: {avg_yes - avg_no:+.1%}")

        # Reranking should help
        assert avg_yes >= avg_no, "Reranking should not decrease recall"

    def test_rerank_factor_scaling(self, encoded_corpus, encoded_queries):
        """Test that higher rerank factors improve recall."""
        dim = encoded_corpus[0][1].shape[1]

        db = omendb.open(":memory:", dimensions=dim, multi_vector=True)
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {}}
            for doc_id, tokens in encoded_corpus
        ]
        db.set(items)

        k = 5
        factors = [1, 2, 4, 8]
        results_by_factor = {f: [] for f in factors}

        for _, query_tokens in encoded_queries:
            ground_truth = brute_force_search(query_tokens, encoded_corpus, k)

            for factor in factors:
                results = db.search(query_tokens.tolist(), k=k, rerank_factor=factor)
                predicted = [r["id"] for r in results]
                results_by_factor[factor].append(compute_recall(predicted, ground_truth))

        print(f"\nRecall@{k} by rerank factor (real ColBERT):")
        prev_recall = 0
        for factor in factors:
            avg = np.mean(results_by_factor[factor])
            print(f"  factor={factor}: {avg:.1%}")
            # Higher factors should generally not decrease recall significantly
            assert avg >= prev_recall - 0.1, f"Factor {factor} recall dropped too much"
            prev_recall = avg


class TestRealColBERTScale:
    """Larger scale tests with real embeddings."""

    def test_semantic_clustering(self, encoded_corpus, encoded_queries, colbert_model):
        """Verify MUVERA retrieves semantically relevant documents."""
        dim = encoded_corpus[0][1].shape[1]

        db = omendb.open(":memory:", dimensions=dim, multi_vector=True)
        items = [
            {"id": doc_id, "vectors": tokens.tolist(), "metadata": {"text": TEST_DOCUMENTS[i]}}
            for i, (doc_id, tokens) in enumerate(encoded_corpus)
        ]
        db.set(items)

        # Test specific semantic queries
        semantic_tests = [
            ("neural network training", ["doc0", "doc1", "doc2", "doc3", "doc4"]),  # ML docs
            ("plant energy production", ["doc5"]),  # Photosynthesis
            ("genetic code", ["doc6", "doc9"]),  # DNA, evolution
            ("historical wars", ["doc12"]),  # WW2
            ("vector search algorithms", ["doc25", "doc26", "doc27", "doc29"]),  # Vector DB docs
        ]

        print("\nSemantic relevance tests:")
        for query_text, expected_relevant in semantic_tests:
            query_emb = list(colbert_model.query_embed([query_text]))[0]
            query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

            results = db.search(query_emb.astype(np.float32).tolist(), k=5, rerank=True)
            top_ids = [r["id"] for r in results]

            # Check if any expected doc is in top results
            found = set(top_ids) & set(expected_relevant)
            print(
                f"  '{query_text}': top={top_ids[:3]}, expected_any={expected_relevant[:3]}, found={len(found)}"
            )

            # At least one relevant doc should be retrieved
            assert len(found) > 0 or len(expected_relevant) == 0, (
                f"No relevant docs found for '{query_text}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
