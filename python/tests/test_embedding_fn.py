"""Tests for embedding_fn support in omendb."""

import numpy as np
import pytest

import omendb
from tests.helpers import create_dense_db


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Simple deterministic embedding function for testing.

    Maps each text to a vector based on its hash. Returns shape (n, 4).
    """
    vectors = []
    for text in texts:
        h = hash(text) % 1000
        vec = np.array(
            [h / 1000, (h * 3 % 1000) / 1000, (h * 7 % 1000) / 1000, (h * 11 % 1000) / 1000],
            dtype=np.float32,
        )
        # Normalize to unit length
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


class TestEmbeddingFnOpen:
    """Test embedding_fn parameter on open()."""

    def test_open_with_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        assert db.embedding_fn is not None

    def test_open_without_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4)
        assert db.embedding_fn is None

    def test_open_memory_with_embedding_fn(self):
        db = omendb.open(":memory:", dimensions=4, embedding_fn=fake_embedder)
        assert db.embedding_fn is not None


class TestSetWithDocument:
    """Test set() with document parameter."""

    def test_single_document(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set("doc1", document="hello world")
        assert len(db) == 1

    def test_single_document_with_metadata(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set("doc1", document="hello world", metadata={"source": "test"})
        result = db.get("doc1")
        assert result is not None
        assert result["metadata"]["source"] == "test"

    def test_batch_documents_kwargs(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(ids=["d1", "d2", "d3"], documents=["hello", "world", "foo"])
        assert len(db) == 3

    def test_batch_documents_with_metadatas(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            ids=["d1", "d2"],
            documents=["hello", "world"],
            metadatas=[{"a": 1}, {"a": 2}],
        )
        r1 = db.get("d1")
        r2 = db.get("d2")
        assert r1["metadata"]["a"] == 1
        assert r2["metadata"]["a"] == 2

    def test_batch_documents_dict_format(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            [
                {"id": "d1", "document": "hello world"},
                {"id": "d2", "document": "goodbye world", "metadata": {"src": "b"}},
            ]
        )
        assert len(db) == 2
        r2 = db.get("d2")
        assert r2["metadata"]["src"] == "b"

    def test_explicit_vectors_still_work(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set("v1", [1.0, 0.0, 0.0, 0.0])
        assert len(db) == 1

    def test_mixed_vector_and_document_in_batch(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            [
                {"id": "v1", "vector": [1.0, 0.0, 0.0, 0.0]},
                {"id": "d1", "document": "hello world"},
            ]
        )
        assert len(db) == 2


class TestSetValidation:
    """Test error handling for document-related validation."""

    def test_document_without_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4)
        with pytest.raises(ValueError, match="No embedding function configured"):
            db.set("doc1", document="hello")

    def test_both_vector_and_document_single(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        with pytest.raises(ValueError, match="Cannot provide both vector and document"):
            db.set("doc1", vector=[1.0, 0.0, 0.0, 0.0], document="hello")

    def test_both_vector_and_document_batch(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        with pytest.raises(ValueError, match="cannot have both"):
            db.set([{"id": "d1", "vector": [1.0, 0, 0, 0], "document": "hello"}])

    def test_documents_without_embedding_fn_kwargs(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4)
        with pytest.raises(ValueError, match="No embedding function configured"):
            db.set(ids=["d1"], documents=["hello"])

    def test_both_vectors_and_documents_kwargs(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        with pytest.raises(ValueError, match="Cannot provide both vectors and documents"):
            db.set(ids=["d1"], vectors=[[1.0, 0, 0, 0]], documents=["hello"])

    def test_ids_documents_length_mismatch(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        with pytest.raises(ValueError, match="same length"):
            db.set(ids=["d1", "d2"], documents=["hello"])


class TestSearchWithString:
    """Test search() with string queries."""

    def test_search_string_query(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(ids=["d1", "d2", "d3"], documents=["hello", "world", "foo"])
        results = db.search("hello", k=2)
        assert len(results) == 2
        # First result should be "hello" since query is identical
        assert results[0]["id"] == "d1"

    def test_search_string_without_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4)
        db.set("v1", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="embedding function"):
            db.search("hello", k=1)

    def test_search_vector_still_works(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set("v1", [1.0, 0.0, 0.0, 0.0])
        results = db.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0]["id"] == "v1"

    def test_search_string_with_filter(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            [
                {"id": "d1", "document": "hello", "metadata": {"cat": "a"}},
                {"id": "d2", "document": "world", "metadata": {"cat": "b"}},
                {"id": "d3", "document": "foo", "metadata": {"cat": "a"}},
            ]
        )
        results = db.search("hello", k=5, filter={"cat": "a"})
        assert all(r["metadata"]["cat"] == "a" for r in results)


class TestSearchHybridWithString:
    """Test search_hybrid() with string query auto-embedding."""

    def test_hybrid_string_query(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            [
                {"id": "d1", "document": "machine learning", "text": "machine learning algorithms"},
                {"id": "d2", "document": "web development", "text": "web development frameworks"},
            ]
        )
        db.flush()
        results = db.search_hybrid("machine learning", k=2)
        assert len(results) >= 1

    def test_hybrid_vector_still_works(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        db.set(
            [
                {"id": "d1", "vector": [1.0, 0, 0, 0], "text": "hello world"},
            ]
        )
        db.flush()
        results = db.search_hybrid([1.0, 0, 0, 0], "hello", k=1)
        assert len(results) >= 1


class TestCollectionInheritance:
    """Test that collections inherit embedding_fn from parent."""

    def test_collection_inherits_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        col = db.collection("test_col")
        assert col.embedding_fn is not None
        col.set("d1", document="hello world")
        assert len(col) == 1

    def test_collection_override_embedding_fn(self, tmp_path):
        def other_embedder(texts):
            return np.zeros((len(texts), 4), dtype=np.float32) + 0.5

        db = create_dense_db(str(tmp_path / "db"), 4, embedding_fn=fake_embedder)
        col = db.collection("test_col", embedding_fn=other_embedder)
        assert col.embedding_fn is not None
        # Should use the override, not the parent
        col.set("d1", document="hello")
        result = col.get("d1")
        # With other_embedder, all vectors are [0.5, 0.5, 0.5, 0.5]
        assert all(abs(v - 0.5) < 0.01 for v in result["vector"])

    def test_collection_without_parent_embedding_fn(self, tmp_path):
        db = create_dense_db(str(tmp_path / "db"), 4)
        col = db.collection("test_col")
        assert col.embedding_fn is None
