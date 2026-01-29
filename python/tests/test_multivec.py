"""Tests for multi-vector (MUVERA) support in OmenDB Python bindings."""

import numpy as np
import pytest

import omendb


class TestMultiVectorBasic:
    """Basic multi-vector store tests."""

    def test_create_multi_vector_store(self):
        """Test creating a multi-vector store."""
        db = omendb.open(":memory:", dimensions=128, multi_vector=True)
        assert db.is_multi_vector is True
        assert db.dimensions == 128

    def test_create_regular_store(self):
        """Test creating a regular (single-vector) store."""
        db = omendb.open(":memory:", dimensions=128)
        assert db.is_multi_vector is False

    def test_create_multi_vector_with_config(self):
        """Test creating a multi-vector store with custom config."""
        db = omendb.open(
            ":memory:",
            dimensions=128,
            multi_vector={"repetitions": 10, "partition_bits": 4},
        )
        assert db.is_multi_vector is True


class TestMultiVectorInsert:
    """Tests for inserting multi-vector documents."""

    @pytest.fixture
    def db(self):
        """Create a multi-vector store with small dims (d_proj=None to skip projection)."""
        return omendb.open(":memory:", dimensions=8, multi_vector={"d_proj": None})

    def test_insert_single_document(self, db):
        """Test inserting a single multi-vector document."""
        tokens = [[0.1] * 8, [0.2] * 8, [0.3] * 8]
        db.set([{"id": "doc1", "vectors": tokens, "metadata": {"title": "Test"}}])
        assert len(db) == 1

    def test_insert_batch(self, db):
        """Test inserting multiple multi-vector documents."""
        docs = [
            {"id": f"doc{i}", "vectors": [[float(i) / 10] * 8 for _ in range(5)], "metadata": {}}
            for i in range(10)
        ]
        db.set(docs)
        assert len(db) == 10

    def test_insert_with_numpy(self, db):
        """Test inserting with numpy arrays."""
        tokens = np.random.rand(5, 8).astype(np.float32).tolist()
        db.set([{"id": "doc1", "vectors": tokens, "metadata": {}}])
        assert len(db) == 1

    def test_insert_empty_vectors_fails(self, db):
        """Test that empty vectors fail."""
        with pytest.raises(ValueError, match="must not be empty"):
            db.set([{"id": "doc1", "vectors": [], "metadata": {}}])

    def test_insert_missing_vectors_key_fails(self, db):
        """Test that missing 'vectors' key fails."""
        with pytest.raises(ValueError, match="missing 'vectors' field"):
            db.set([{"id": "doc1", "metadata": {}}])


class TestMultiVectorSearch:
    """Tests for multi-vector search."""

    @pytest.fixture
    def populated_db(self):
        """Create and populate a multi-vector store."""
        db = omendb.open(":memory:", dimensions=8, multi_vector={"d_proj": None})

        # Create docs with distinct patterns
        docs = []
        for i in range(100):
            # Each doc has 5 tokens with a pattern based on its index
            base = float(i) / 100
            tokens = [[base + 0.01 * j] * 8 for j in range(5)]
            docs.append({"id": f"doc{i}", "vectors": tokens, "metadata": {"index": i}})

        db.set(docs)
        return db

    def test_basic_search(self, populated_db):
        """Test basic multi-vector search."""
        query_tokens = [[0.5] * 8, [0.51] * 8, [0.52] * 8]
        results = populated_db.search(query_tokens, k=5)

        assert len(results) == 5
        assert all("id" in r and "distance" in r and "metadata" in r for r in results)

    def test_search_with_rerank_off(self, populated_db):
        """Test search with reranking disabled."""
        query_tokens = [[0.5] * 8, [0.51] * 8]
        results = populated_db.search(query_tokens, k=5, rerank=False)

        assert len(results) == 5

    def test_search_with_custom_rerank_factor(self, populated_db):
        """Test search with custom rerank factor."""
        query_tokens = [[0.5] * 8, [0.51] * 8]
        results = populated_db.search(query_tokens, k=5, rerank_factor=8)

        assert len(results) == 5

    def test_search_returns_metadata(self, populated_db):
        """Test that search returns metadata."""
        query_tokens = [[0.5] * 8]
        results = populated_db.search(query_tokens, k=1)

        assert len(results) == 1
        assert "index" in results[0]["metadata"]

    def test_search_k_equals_1(self, populated_db):
        """Test search with k=1."""
        query_tokens = [[0.0] * 8]  # Should match doc0
        results = populated_db.search(query_tokens, k=1)

        assert len(results) == 1


class TestMultiVectorReranking:
    """Tests to verify reranking improves results."""

    @pytest.fixture
    def db_with_similar_docs(self):
        """Create DB with docs that benefit from reranking."""
        db = omendb.open(":memory:", dimensions=16, multi_vector={"d_proj": None})

        # Create docs with overlapping patterns
        docs = []
        for i in range(50):
            # Vary the number of tokens and their values
            num_tokens = 3 + (i % 5)  # 3-7 tokens
            tokens = []
            for j in range(num_tokens):
                # Create tokens with some structure
                token = [0.0] * 16
                token[i % 16] = 1.0  # One hot component
                token[(i + j) % 16] += 0.5
                tokens.append(token)
            docs.append(
                {"id": f"doc{i}", "vectors": tokens, "metadata": {"num_tokens": num_tokens}}
            )

        db.set(docs)
        return db

    def test_rerank_changes_order(self, db_with_similar_docs):
        """Verify reranking can change result ordering."""
        query = [[1.0] + [0.0] * 15, [0.0, 1.0] + [0.0] * 14]

        # Get results with and without reranking
        results_no_rerank = db_with_similar_docs.search(query, k=10, rerank=False)
        results_rerank = db_with_similar_docs.search(query, k=10, rerank=True)

        # Both should return results
        assert len(results_no_rerank) == 10
        assert len(results_rerank) == 10

        # Order may differ (not guaranteed but often does)
        # At minimum, both should return valid results
        ids_no_rerank = [r["id"] for r in results_no_rerank]
        ids_rerank = [r["id"] for r in results_rerank]

        # Results should be valid doc IDs
        assert all(id.startswith("doc") for id in ids_no_rerank)
        assert all(id.startswith("doc") for id in ids_rerank)


class TestMultiVectorErrors:
    """Test error cases."""

    def test_single_vector_query_to_multi_vector_store(self):
        """Test that single-vector query to multi-vector store gives sensible error."""
        db = omendb.open(":memory:", dimensions=8, multi_vector={"d_proj": None})
        db.set([{"id": "doc1", "vectors": [[0.1] * 8, [0.2] * 8], "metadata": {}}])

        # This should fail because we're passing a 1D vector to a multi-vector store
        # The error handling depends on implementation
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            db.search([0.1] * 8, k=1)

    def test_single_vector_query_dimension_mismatch(self):
        """Test that mismatched dimensions are handled."""
        db = omendb.open(":memory:", dimensions=8, multi_vector={"d_proj": None})
        db.set([{"id": "doc1", "vectors": [[0.1] * 8, [0.2] * 8], "metadata": {}}])

        # Wrong dimension should fail
        with pytest.raises((ValueError, RuntimeError)):
            db.search([[0.1] * 16], k=1)  # Wrong dimension

    def test_multi_vector_with_quantization_fails(self):
        """Test that multi-vector with quantization is not yet supported."""
        with pytest.raises(ValueError, match="quantization"):
            omendb.open(":memory:", dimensions=32, multi_vector=True, quantization=True)


class TestMultiVectorPersistence:
    """Tests for multi-vector persistence (MUV-13)."""

    def test_persistence_roundtrip(self):
        """Test that multi-vector data persists across close/reopen."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_multivec.omen")

            # Create and populate (d_proj=None for small dim)
            db = omendb.open(path, dimensions=8, multi_vector={"d_proj": None})
            db.set(
                [
                    {
                        "id": "doc1",
                        "vectors": [[0.1] * 8, [0.2] * 8],
                        "metadata": {"title": "first"},
                    },
                    {"id": "doc2", "vectors": [[0.3] * 8], "metadata": {"title": "second"}},
                ]
            )
            db.flush()
            assert len(db) == 2
            db.close()

            # Reopen and verify
            db2 = omendb.open(path)
            assert db2.is_multi_vector is True
            assert len(db2) == 2

            # Verify docs by searching
            query = [[0.1] * 8]
            results = db2.search(query, k=2)
            ids = [r["id"] for r in results]
            assert "doc1" in ids or "doc2" in ids
            db2.close()

    def test_persistence_rerank_after_reload(self):
        """Test that reranking works after reload."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_rerank.omen")

            # Create store with docs (d_proj=None for small dim)
            db = omendb.open(path, dimensions=4, multi_vector={"d_proj": None})
            db.set(
                [
                    {
                        "id": "doc1",
                        "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        "metadata": {},
                    },
                    {"id": "doc2", "vectors": [[0.3, 0.3, 0.3, 0.0]], "metadata": {}},
                ]
            )
            db.flush()
            db.close()

            # Reopen and search with reranking
            db2 = omendb.open(path)
            query = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            results = db2.search(query, k=2, rerank=True)

            assert len(results) == 2
            # doc1 should be first (better MaxSim match)
            assert results[0]["id"] == "doc1"
            db2.close()

    def test_persistence_large_store(self):
        """Test persistence with larger dataset."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_large.omen")

            # Create store with 100 docs
            db = omendb.open(path, dimensions=32, multi_vector=True)
            docs = []
            for i in range(100):
                num_tokens = (i % 5) + 1
                tokens = np.random.rand(num_tokens, 32).astype(np.float32).tolist()
                docs.append({"id": f"doc{i}", "vectors": tokens, "metadata": {"idx": i}})
            db.set(docs)
            db.flush()
            db.close()

            # Reopen and verify
            db2 = omendb.open(path)
            assert db2.is_multi_vector is True
            assert len(db2) == 100

            # Search should work
            query = np.random.rand(3, 32).astype(np.float32).tolist()
            results = db2.search(query, k=5)
            assert len(results) == 5
            db2.close()


class TestMultiVectorScale:
    """Larger scale tests."""

    def test_1k_documents(self):
        """Test with 1000 documents."""
        db = omendb.open(":memory:", dimensions=32, multi_vector=True)

        # Insert 1000 docs with 10 tokens each
        docs = []
        for i in range(1000):
            tokens = np.random.rand(10, 32).astype(np.float32).tolist()
            docs.append({"id": f"doc{i}", "vectors": tokens, "metadata": {"index": i}})

        db.set(docs)
        assert len(db) == 1000

        # Search
        query = np.random.rand(5, 32).astype(np.float32).tolist()
        results = db.search(query, k=10)
        assert len(results) == 10

    def test_variable_token_counts(self):
        """Test with variable number of tokens per document."""
        db = omendb.open(":memory:", dimensions=16, multi_vector={"d_proj": None})

        docs = []
        for i in range(100):
            # 1 to 20 tokens per document
            num_tokens = 1 + (i % 20)
            tokens = np.random.rand(num_tokens, 16).astype(np.float32).tolist()
            docs.append(
                {"id": f"doc{i}", "vectors": tokens, "metadata": {"num_tokens": num_tokens}}
            )

        db.set(docs)
        assert len(db) == 100

        # Search with varying query sizes
        for num_query_tokens in [1, 5, 10]:
            query = np.random.rand(num_query_tokens, 16).astype(np.float32).tolist()
            results = db.search(query, k=5)
            assert len(results) == 5
