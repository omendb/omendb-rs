"""Tests for LangChain VectorStore integration."""

import shutil
import tempfile

import pytest


class FakeEmbeddings:
    """Fake embeddings model for testing (no API calls)."""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings from text."""
        embeddings = []
        for text in texts:
            # Use hash of text to generate deterministic embedding
            seed = hash(text) % 2**31
            embedding = self._generate_embedding(seed)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate deterministic embedding from query text."""
        seed = hash(text) % 2**31
        return self._generate_embedding(seed)

    def _generate_embedding(self, seed: int) -> list[float]:
        """Generate a deterministic embedding from seed using LCG."""
        s = seed
        embedding = []
        for _ in range(self.dimensions):
            s = (s * 1103515245 + 12345) % 2147483648
            embedding.append((float(s) / 2147483648.0) * 2.0 - 1.0)
        return embedding


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the database."""
    temp_dir = tempfile.mkdtemp()
    yield f"{temp_dir}/vectors"
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embeddings():
    """Create fake embeddings model."""
    return FakeEmbeddings(dimensions=128)


class TestOmenDBVectorStore:
    """Tests for OmenDBVectorStore."""

    def test_from_texts(self, temp_db_path, embeddings):
        """Test creating a vector store from texts."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["Hello world", "How are you?", "Vector databases are cool"]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db_path,
        )

        assert len(store) == 3

    def test_from_texts_with_metadata(self, temp_db_path, embeddings):
        """Test creating a vector store with metadata."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["Doc 1", "Doc 2"]
        metadatas = [{"source": "a.txt"}, {"source": "b.txt"}]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            path=temp_db_path,
        )

        assert len(store) == 2

    def test_from_documents(self, temp_db_path, embeddings):
        """Test creating a vector store from documents."""
        from langchain_core.documents import Document

        from omendb.langchain import OmenDBVectorStore

        docs = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="How are you?", metadata={"source": "test2"}),
        ]

        store = OmenDBVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            path=temp_db_path,
        )

        assert len(store) == 2

    def test_similarity_search(self, temp_db_path, embeddings):
        """Test similarity search returns correct documents."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["cats are great", "dogs are awesome", "birds can fly"]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db_path,
        )

        # Search for something similar to first text
        results = store.similarity_search("cats are great", k=2)

        assert len(results) == 2
        # First result should be exact match
        assert results[0].page_content == "cats are great"

    def test_similarity_search_by_vector(self, temp_db_path, embeddings):
        """Test similarity search by vector."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["hello", "world", "test"]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db_path,
        )

        # Get embedding for "hello" and search
        query_embedding = embeddings.embed_query("hello")
        results = store.similarity_search_by_vector(query_embedding, k=2)

        assert len(results) == 2
        assert results[0].page_content == "hello"

    def test_similarity_search_with_score(self, temp_db_path, embeddings):
        """Test similarity search returns scores."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["hello world", "goodbye world"]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db_path,
        )

        results = store.similarity_search_with_score("hello world", k=2)

        assert len(results) == 2
        doc, score = results[0]
        assert doc.page_content == "hello world"
        assert score == 0.0  # Exact match should have 0 distance

    def test_add_texts(self, temp_db_path, embeddings):
        """Test adding texts incrementally."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore(
            embedding=embeddings,
            path=temp_db_path,
        )

        assert len(store) == 0

        ids = store.add_texts(["text1", "text2"])
        assert len(ids) == 2
        assert len(store) == 2

        ids = store.add_texts(["text3"])
        assert len(ids) == 1
        assert len(store) == 3

    def test_add_texts_with_ids(self, temp_db_path, embeddings):
        """Test adding texts with custom IDs."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore(
            embedding=embeddings,
            path=temp_db_path,
        )

        ids = store.add_texts(
            texts=["hello", "world"],
            ids=["doc1", "doc2"],
        )

        assert ids == ["doc1", "doc2"]

    def test_add_documents(self, temp_db_path, embeddings):
        """Test adding documents."""
        from langchain_core.documents import Document

        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore(
            embedding=embeddings,
            path=temp_db_path,
        )

        docs = [
            Document(page_content="Hello", metadata={"type": "greeting"}),
        ]

        ids = store.add_documents(docs)
        assert len(ids) == 1
        assert len(store) == 1

    def test_delete(self, temp_db_path, embeddings):
        """Test deleting documents."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore.from_texts(
            texts=["keep", "delete"],
            embedding=embeddings,
            ids=["keep_id", "delete_id"],
            path=temp_db_path,
        )

        assert len(store) == 2

        result = store.delete(["delete_id"])
        assert result is True
        assert len(store) == 1

        # Verify correct document remains
        results = store.similarity_search("keep", k=1)
        assert results[0].page_content == "keep"

    def test_get_by_ids(self, temp_db_path, embeddings):
        """Test retrieving documents by IDs."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore.from_texts(
            texts=["doc1 content", "doc2 content"],
            embedding=embeddings,
            ids=["id1", "id2"],
            path=temp_db_path,
        )

        docs = store.get_by_ids(["id1", "id2"])
        assert len(docs) == 2

        # Test with missing ID
        docs = store.get_by_ids(["id1", "nonexistent"])
        assert len(docs) == 1

    def test_metadata_preserved(self, temp_db_path, embeddings):
        """Test that metadata is preserved through search."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["content"]
        metadatas = [{"key": "value", "number": 42}]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            path=temp_db_path,
        )

        results = store.similarity_search("content", k=1)
        assert results[0].metadata["key"] == "value"
        assert results[0].metadata["number"] == 42

    def test_filter_search(self, temp_db_path, embeddings):
        """Test filtered similarity search."""
        from omendb.langchain import OmenDBVectorStore

        texts = ["cat document", "dog document", "bird document"]
        metadatas = [
            {"animal": "cat"},
            {"animal": "dog"},
            {"animal": "bird"},
        ]

        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            path=temp_db_path,
        )

        # Search with filter
        results = store.similarity_search(
            "animal",
            k=3,
            filter={"animal": "dog"},
        )

        assert len(results) == 1
        assert results[0].metadata["animal"] == "dog"

    def test_as_retriever(self, temp_db_path, embeddings):
        """Test creating a retriever from the vector store."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore.from_texts(
            texts=["hello world", "goodbye world"],
            embedding=embeddings,
            path=temp_db_path,
        )

        retriever = store.as_retriever(search_kwargs={"k": 2})

        # Use invoke method - search for exact match to ensure deterministic results
        docs = retriever.invoke("hello world")
        assert len(docs) == 2
        # First result should be exact match
        assert docs[0].page_content == "hello world"

    def test_embeddings_property(self, temp_db_path, embeddings):
        """Test that embeddings property returns the model."""
        from omendb.langchain import OmenDBVectorStore

        store = OmenDBVectorStore(
            embedding=embeddings,
            path=temp_db_path,
        )

        assert store.embeddings is embeddings

    def test_persistence(self, temp_db_path, embeddings):
        """Test that data persists across sessions."""
        from omendb.langchain import OmenDBVectorStore

        # Create and populate store
        store1 = OmenDBVectorStore.from_texts(
            texts=["persistent data"],
            embedding=embeddings,
            ids=["persist_id"],
            path=temp_db_path,
        )
        assert len(store1) == 1
        store1.flush()  # Required for persistence
        del store1

        # Open in new session
        store2 = OmenDBVectorStore(
            embedding=embeddings,
            path=temp_db_path,
        )

        assert len(store2) == 1
        results = store2.similarity_search("persistent data", k=1)
        assert results[0].page_content == "persistent data"


class TestOmenDBVectorStoreScaleTest:
    """Scale tests for OmenDBVectorStore (marked slow)."""

    @pytest.mark.slow
    def test_1k_vectors(self, temp_db_path, embeddings):
        """Test with 1K vectors."""
        import time

        from omendb.langchain import OmenDBVectorStore

        n_vectors = 1000
        texts = [f"Document number {i} with some content" for i in range(n_vectors)]

        # Build
        start = time.time()
        store = OmenDBVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            path=temp_db_path,
        )
        build_time = time.time() - start

        assert len(store) == n_vectors
        print(f"\nBuild time for {n_vectors} vectors: {build_time:.2f}s")

        # Search
        start = time.time()
        n_queries = 100
        for i in range(n_queries):
            results = store.similarity_search(f"Document number {i}", k=10)
            assert len(results) == 10
        search_time = time.time() - start

        qps = n_queries / search_time
        print(f"Search QPS: {qps:.0f}")
        assert qps > 100  # Should be much faster, but be conservative
