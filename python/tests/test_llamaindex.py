"""Tests for LlamaIndex VectorStore integration."""

import shutil
import tempfile

import pytest

# Skip all tests if llama-index-core is not installed
pytest.importorskip("llama_index.core")

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from omendb.llamaindex import OmenDBVectorStore


class FakeEmbedding:
    """Fake embedding model for testing (deterministic embeddings)."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def get_query_embedding(self, query: str) -> list[float]:
        """Generate deterministic embedding from query."""
        return self._embed(query)

    def get_text_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text."""
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        # Convert hex to floats
        embedding = []
        for i in range(0, min(len(h), self.dimensions * 2), 2):
            val = int(h[i : i + 2], 16) / 255.0
            embedding.append(val)
        # Pad with zeros if needed
        while len(embedding) < self.dimensions:
            embedding.append(0.0)
        return embedding[: self.dimensions]


class TestOmenDBVectorStore:
    """Test OmenDBVectorStore implementation."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary database path."""
        path = tempfile.mkdtemp(suffix="")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def fake_embed(self):
        """Create fake embedding model."""
        return FakeEmbedding(dimensions=128)

    def test_init(self, db_path):
        """Test vector store initialization."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)
        assert store.path == db_path
        assert store.dimensions == 128

    def test_add_nodes(self, db_path, fake_embed):
        """Test adding nodes to vector store."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="Hello world",
                id_="node1",
                embedding=fake_embed.get_text_embedding("Hello world"),
                metadata={"source": "test"},
            ),
            TextNode(
                text="How are you?",
                id_="node2",
                embedding=fake_embed.get_text_embedding("How are you?"),
                metadata={"source": "test"},
            ),
        ]

        ids = store.add(nodes)

        assert len(ids) == 2
        assert "node1" in ids
        assert "node2" in ids

    def test_query_basic(self, db_path, fake_embed):
        """Test basic query functionality."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="The quick brown fox",
                id_="fox",
                embedding=fake_embed.get_text_embedding("The quick brown fox"),
            ),
            TextNode(
                text="The lazy dog",
                id_="dog",
                embedding=fake_embed.get_text_embedding("The lazy dog"),
            ),
        ]
        store.add(nodes)

        # Query with embedding
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("The quick brown fox"),
            similarity_top_k=2,
        )
        result = store.query(query)

        assert len(result.nodes) == 2
        assert len(result.similarities) == 2
        assert len(result.ids) == 2
        # First result should be "fox" (exact match)
        assert result.ids[0] == "fox"

    def test_query_with_metadata_filter_eq(self, db_path, fake_embed):
        """Test query with equality filter."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="Document A",
                id_="a",
                embedding=fake_embed.get_text_embedding("Document A"),
                metadata={"category": "science"},
            ),
            TextNode(
                text="Document B",
                id_="b",
                embedding=fake_embed.get_text_embedding("Document B"),
                metadata={"category": "history"},
            ),
        ]
        store.add(nodes)

        # Query with filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", operator=FilterOperator.EQ, value="science"),
            ]
        )
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("document"),
            similarity_top_k=10,
            filters=filters,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.ids[0] == "a"

    def test_query_with_metadata_filter_gte(self, db_path, fake_embed):
        """Test query with comparison filter."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="Document 1",
                id_="d1",
                embedding=fake_embed.get_text_embedding("Document 1"),
                metadata={"score": 10},
            ),
            TextNode(
                text="Document 2",
                id_="d2",
                embedding=fake_embed.get_text_embedding("Document 2"),
                metadata={"score": 50},
            ),
            TextNode(
                text="Document 3",
                id_="d3",
                embedding=fake_embed.get_text_embedding("Document 3"),
                metadata={"score": 90},
            ),
        ]
        store.add(nodes)

        # Query with GTE filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="score", operator=FilterOperator.GTE, value=50),
            ]
        )
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("document"),
            similarity_top_k=10,
            filters=filters,
        )
        result = store.query(query)

        assert len(result.nodes) == 2
        assert all(r in ["d2", "d3"] for r in result.ids)

    def test_delete(self, db_path, fake_embed):
        """Test deleting nodes."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="To be deleted",
                id_="delete_me",
                embedding=fake_embed.get_text_embedding("To be deleted"),
            ),
            TextNode(
                text="Keep this",
                id_="keep_me",
                embedding=fake_embed.get_text_embedding("Keep this"),
            ),
        ]
        store.add(nodes)

        # Delete one node
        store.delete("delete_me")

        # Query should only find the kept node
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("test"),
            similarity_top_k=10,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.ids[0] == "keep_me"

    def test_delete_nodes(self, db_path, fake_embed):
        """Test deleting multiple nodes."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text=f"Node {i}", id_=f"n{i}", embedding=fake_embed.get_text_embedding(f"Node {i}")
            )
            for i in range(5)
        ]
        store.add(nodes)

        # Delete some nodes
        store.delete_nodes(["n1", "n3"])

        # Query should find remaining nodes
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("node"),
            similarity_top_k=10,
        )
        result = store.query(query)

        assert len(result.nodes) == 3
        assert all(r in ["n0", "n2", "n4"] for r in result.ids)

    def test_text_retrieval(self, db_path, fake_embed):
        """Test that text content is properly stored and retrieved."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        original_text = "This is the original document content."
        nodes = [
            TextNode(
                text=original_text,
                id_="doc1",
                embedding=fake_embed.get_text_embedding(original_text),
            ),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding(original_text),
            similarity_top_k=1,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].get_content() == original_text

    def test_metadata_preserved(self, db_path, fake_embed):
        """Test that metadata is properly stored and retrieved."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        metadata = {
            "author": "Alice",
            "year": 2024,
            "tags": ["important", "verified"],
        }
        nodes = [
            TextNode(
                text="Test document",
                id_="doc1",
                embedding=fake_embed.get_text_embedding("Test document"),
                metadata=metadata,
            ),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("test"),
            similarity_top_k=1,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.metadata["author"] == "Alice"
        assert node.metadata["year"] == 2024
        assert node.metadata["tags"] == ["important", "verified"]

    def test_persistence(self, db_path, fake_embed):
        """Test that data persists after reopening."""
        # Create and populate store
        store1 = OmenDBVectorStore(path=db_path, dimensions=128)
        nodes = [
            TextNode(
                text="Persistent data",
                id_="persist1",
                embedding=fake_embed.get_text_embedding("Persistent data"),
                metadata={"key": "value"},
            ),
        ]
        store1.add(nodes)
        store1.flush()  # Required for persistence

        # Close by going out of scope
        del store1

        # Reopen store
        store2 = OmenDBVectorStore(path=db_path, dimensions=128)
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("persistent"),
            similarity_top_k=1,
        )
        result = store2.query(query)

        assert len(result.nodes) == 1
        assert result.ids[0] == "persist1"
        assert result.nodes[0].get_content() == "Persistent data"

    def test_and_filter(self, db_path, fake_embed):
        """Test AND filter combination."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        nodes = [
            TextNode(
                text="Doc A",
                id_="a",
                embedding=fake_embed.get_text_embedding("Doc A"),
                metadata={"category": "science", "score": 80},
            ),
            TextNode(
                text="Doc B",
                id_="b",
                embedding=fake_embed.get_text_embedding("Doc B"),
                metadata={"category": "science", "score": 40},
            ),
            TextNode(
                text="Doc C",
                id_="c",
                embedding=fake_embed.get_text_embedding("Doc C"),
                metadata={"category": "history", "score": 90},
            ),
        ]
        store.add(nodes)

        # Query with AND filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", operator=FilterOperator.EQ, value="science"),
                MetadataFilter(key="score", operator=FilterOperator.GTE, value=50),
            ],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=fake_embed.get_query_embedding("doc"),
            similarity_top_k=10,
            filters=filters,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.ids[0] == "a"

    def test_class_name(self):
        """Test class name for serialization."""
        assert OmenDBVectorStore.class_name() == "OmenDBVectorStore"

    def test_stores_text_flag(self, db_path):
        """Test that stores_text flag is True."""
        store = OmenDBVectorStore(path=db_path)
        assert store.stores_text is True

    def test_empty_nodes_list(self, db_path):
        """Test adding empty list of nodes."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)
        ids = store.add([])
        assert ids == []

    def test_empty_query(self, db_path, fake_embed):
        """Test query with no embedding."""
        store = OmenDBVectorStore(path=db_path, dimensions=128)

        # Add some nodes first
        nodes = [
            TextNode(text="Test", id_="test", embedding=fake_embed.get_text_embedding("Test")),
        ]
        store.add(nodes)

        # Query without embedding
        query = VectorStoreQuery(query_embedding=None, similarity_top_k=10)
        result = store.query(query)

        assert len(result.nodes) == 0
