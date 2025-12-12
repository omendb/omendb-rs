"""LangChain VectorStore integration for OmenDB.

This module provides a LangChain-compatible VectorStore implementation
that wraps OmenDB for seamless integration with LangChain RAG pipelines.

Example:
    >>> from langchain_openai import OpenAIEmbeddings
    >>> from omendb.langchain import OmenDBVectorStore
    >>>
    >>> # Create from texts
    >>> vectorstore = OmenDBVectorStore.from_texts(
    ...     texts=["Hello world", "How are you?"],
    ...     embedding=OpenAIEmbeddings(),
    ...     path="./my_vectors",
    ... )
    >>>
    >>> # Search
    >>> docs = vectorstore.similarity_search("greeting", k=2)
    >>> print(docs[0].page_content)
    Hello world
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class OmenDBVectorStore(VectorStore):
    """LangChain VectorStore implementation using OmenDB.

    OmenDB is a fast embedded vector database with HNSW + ACORN-1
    that provides ~19,000 QPS @ 10K vectors with 100% recall.

    Features:
        - HNSW index with adaptive parameters
        - Extended RaBitQ quantization (8x compression)
        - ACORN-1 filtered search (37.79x speedup)
        - MongoDB-style metadata filtering
        - Automatic persistence with persistent storage

    Args:
        embedding: LangChain Embeddings model for text-to-vector conversion.
        path: Path to database directory. Uses persistent persistent storage.
        dimensions: Vector dimensionality. Auto-detected when loading
            existing database.
        **kwargs: Additional arguments passed to omendb.open().

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from omendb.langchain import OmenDBVectorStore
        >>>
        >>> vectorstore = OmenDBVectorStore(
        ...     embedding=OpenAIEmbeddings(),
        ...     path="./my_vectors",
        ... )
        >>> vectorstore.add_texts(["Hello world", "How are you?"])
        >>> docs = vectorstore.similarity_search("greeting", k=2)
    """

    def __init__(
        self,
        embedding: Embeddings,
        path: str = "./omendb-vectors",
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OmenDBVectorStore.

        Args:
            embedding: LangChain Embeddings model.
            path: Path to database directory.
            dimensions: Vector dimensionality (auto-detected from embedding if None).
            **kwargs: Additional arguments for omendb.open().
        """
        import omendb

        self._embedding = embedding
        self._path = path

        # Auto-detect dimensions from embedding model if not specified
        if dimensions is None:
            # Embed a test string to get dimensions
            test_embedding = embedding.embed_query("test")
            dimensions = len(test_embedding)

        self._dimensions = dimensions
        self._db = omendb.open(path, dimensions=dimensions, **kwargs)

    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings model."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text. Auto-generated if not provided.
            **kwargs: Additional arguments (unused).

        Returns:
            List of IDs for added texts.
        """
        texts_list = list(texts)

        # Generate embeddings
        embeddings = self._embedding.embed_documents(texts_list)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare metadata with page_content stored
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # Build batch for set
        items = []
        for text, embedding, id_, metadata in zip(texts_list, embeddings, ids, metadatas):
            # Store page_content in metadata for retrieval
            item_metadata = {**metadata, "page_content": text}
            items.append(
                {
                    "id": id_,
                    "vector": embedding,
                    "metadata": item_metadata,
                }
            )

        self._db.set(items)
        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: LangChain Documents to add.
            ids: Optional IDs for each document.
            **kwargs: Additional arguments.

        Returns:
            List of IDs for added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

    def delete(
        self,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Delete documents by ID.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional arguments (unused).

        Returns:
            True if deletion was successful.
        """
        if ids is None:
            return False

        deleted = self._db.delete(ids)
        return deleted > 0

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents by text query.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional MongoDB-style metadata filter.
            **kwargs: Additional arguments (unused).

        Returns:
            List of similar Documents.
        """
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for similar documents by vector.

        Args:
            embedding: Query vector.
            k: Number of results to return.
            filter: Optional MongoDB-style metadata filter.
            **kwargs: Additional arguments (unused).

        Returns:
            List of similar Documents.
        """
        results = self._db.search(query=embedding, k=k, filter=filter)

        documents = []
        for result in results:
            metadata = result.get("metadata", {})
            # Extract page_content from metadata
            page_content = metadata.pop("page_content", "")
            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                    id=result.get("id"),
                )
            )

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional MongoDB-style metadata filter.
            **kwargs: Additional arguments (unused).

        Returns:
            List of (Document, score) tuples. Lower scores = more similar.
        """
        embedding = self._embedding.embed_query(query)
        results = self._db.search(query=embedding, k=k, filter=filter)

        documents_with_scores = []
        for result in results:
            metadata = result.get("metadata", {})
            page_content = metadata.pop("page_content", "")
            doc = Document(
                page_content=page_content,
                metadata=metadata,
                id=result.get("id"),
            )
            # OmenDB returns L2 distance (lower = more similar)
            score = result.get("distance", 0.0)
            documents_with_scores.append((doc, score))

        return documents_with_scores

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: Sequence of document IDs.

        Returns:
            List of Documents. Missing IDs are skipped.
        """
        documents = []
        for id_ in ids:
            result = self._db.get(id_)
            if result is not None:
                metadata = result.get("metadata", {})
                page_content = metadata.pop("page_content", "")
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                        id=result.get("id"),
                    )
                )

        return documents

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        path: str = "./omendb-vectors",
        **kwargs: Any,
    ) -> OmenDBVectorStore:
        """Create a vector store from texts.

        Args:
            texts: Texts to add.
            embedding: LangChain Embeddings model.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            path: Path to database directory.
            **kwargs: Additional arguments for OmenDBVectorStore.

        Returns:
            Initialized OmenDBVectorStore with texts added.

        Example:
            >>> from langchain_openai import OpenAIEmbeddings
            >>> vectorstore = OmenDBVectorStore.from_texts(
            ...     texts=["Hello", "World"],
            ...     embedding=OpenAIEmbeddings(),
            ...     path="./my_vectors",
            ... )
        """
        store = cls(embedding=embedding, path=path, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        ids: list[str] | None = None,
        path: str = "./omendb-vectors",
        **kwargs: Any,
    ) -> OmenDBVectorStore:
        """Create a vector store from documents.

        Args:
            documents: LangChain Documents to add.
            embedding: LangChain Embeddings model.
            ids: Optional IDs for each document.
            path: Path to database directory.
            **kwargs: Additional arguments for OmenDBVectorStore.

        Returns:
            Initialized OmenDBVectorStore with documents added.

        Example:
            >>> from langchain_core.documents import Document
            >>> from langchain_openai import OpenAIEmbeddings
            >>> docs = [Document(page_content="Hello", metadata={"source": "test"})]
            >>> vectorstore = OmenDBVectorStore.from_documents(
            ...     documents=docs,
            ...     embedding=OpenAIEmbeddings(),
            ... )
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            ids=ids,
            path=path,
            **kwargs,
        )

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return len(self._db)

    def flush(self) -> None:
        """Flush data to disk for persistence."""
        self._db.flush()
