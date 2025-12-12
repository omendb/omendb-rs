"""LlamaIndex VectorStore integration for OmenDB.

This module provides a LlamaIndex-compatible VectorStore implementation
that wraps OmenDB for seamless integration with LlamaIndex RAG pipelines.

Example:
    >>> from llama_index.core import VectorStoreIndex, StorageContext
    >>> from llama_index.embeddings.openai import OpenAIEmbedding
    >>> from omendb.llamaindex import OmenDBVectorStore
    >>>
    >>> # Create vector store
    >>> vector_store = OmenDBVectorStore(path="./my_vectors")
    >>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
    >>>
    >>> # Build index from documents
    >>> index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    >>>
    >>> # Query
    >>> query_engine = index.as_query_engine()
    >>> response = query_engine.query("What is the main topic?")
"""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class OmenDBVectorStore(BasePydanticVectorStore):
    """LlamaIndex VectorStore implementation using OmenDB.

    OmenDB is a fast embedded vector database with HNSW + ACORN-1
    that provides ~19,000 QPS @ 10K vectors with 100% recall.

    Features:
        - HNSW index with adaptive parameters
        - Extended RaBitQ quantization (8x compression)
        - ACORN-1 filtered search (37.79x speedup)
        - MongoDB-style metadata filtering
        - Automatic persistence with persistent storage

    Args:
        path: Path to database directory. Uses persistent persistent storage.
        dimensions: Vector dimensionality. If None, auto-detected on first insert.
        **kwargs: Additional arguments passed to omendb.open().

    Example:
        >>> from llama_index.core import VectorStoreIndex, StorageContext
        >>> from omendb.llamaindex import OmenDBVectorStore
        >>>
        >>> vector_store = OmenDBVectorStore(path="./my_vectors")
        >>> storage_context = StorageContext.from_defaults(vector_store=vector_store)
        >>> index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    """

    stores_text: bool = True
    flat_metadata: bool = True

    # Pydantic fields
    path: str = "./omendb-vectors"
    dimensions: int | None = None

    # Private attributes (not serialized)
    _db: Any = None
    _initialized: bool = False

    def __init__(
        self,
        path: str = "./omendb-vectors",
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OmenDBVectorStore.

        Args:
            path: Path to database directory.
            dimensions: Vector dimensionality (auto-detected on first insert if None).
            **kwargs: Additional arguments for omendb.open().
        """
        super().__init__(path=path, dimensions=dimensions)
        self._kwargs = kwargs
        self._db = None
        self._initialized = False

    def _ensure_db(self, dimensions: int | None = None) -> None:
        """Ensure database is initialized."""
        if self._db is not None:
            return

        import omendb

        # Use provided dimensions or fall back to stored
        dims = dimensions or self.dimensions or 1536  # Default to OpenAI dimensions
        self._db = omendb.open(self.path, dimensions=dims, **getattr(self, "_kwargs", {}))
        self._initialized = True

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "OmenDBVectorStore"

    @property
    def client(self) -> Any:
        """Return the underlying OmenDB database client."""
        self._ensure_db()
        return self._db

    def add(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of nodes to add.
            **kwargs: Additional arguments (unused).

        Returns:
            List of node IDs that were added.
        """
        if not nodes:
            return []

        # Get dimensions from first node's embedding
        first_embedding = nodes[0].get_embedding()
        if first_embedding:
            self._ensure_db(dimensions=len(first_embedding))
        else:
            self._ensure_db()

        ids = []
        items = []

        for node in nodes:
            node_id = node.node_id
            embedding = node.get_embedding()

            if embedding is None:
                continue

            # Build metadata from node
            metadata = node.metadata.copy() if node.metadata else {}

            # Store text content in metadata for retrieval
            text = node.get_content()
            if text:
                metadata["_text"] = text

            # Store node type info
            metadata["_node_type"] = node.class_name()

            items.append(
                {
                    "id": node_id,
                    "vector": embedding,
                    "metadata": metadata,
                }
            )
            ids.append(node_id)

        if items:
            self._db.set(items)

        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete nodes by reference document ID.

        Args:
            ref_doc_id: The document ID to delete.
            **kwargs: Additional arguments (unused).
        """
        self._ensure_db()
        # OmenDB delete expects a list of IDs
        self._db.delete([ref_doc_id])

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete specific nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete.
            **kwargs: Additional arguments (unused).
        """
        if not node_ids:
            return

        self._ensure_db()
        self._db.delete(node_ids)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Args:
            query: VectorStoreQuery containing query embedding and parameters.
            **kwargs: Additional arguments (unused).

        Returns:
            VectorStoreQueryResult with matching nodes, similarities, and IDs.
        """
        self._ensure_db()

        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # Build filter dict from LlamaIndex filters
        filter_dict = None
        if query.filters is not None:
            filter_dict = self._convert_filters(query.filters)

        # Query OmenDB
        k = query.similarity_top_k or 10
        results = self._db.search(
            query=query.query_embedding,
            k=k,
            filter=filter_dict,
        )

        # Convert results to LlamaIndex format
        nodes = []
        similarities = []
        ids = []

        for result in results:
            node_id = result.get("id", "")
            distance = result.get("distance", 0.0)
            metadata = result.get("metadata", {})

            # Extract text from metadata
            text = metadata.pop("_text", "")
            metadata.pop("_node_type", None)

            # Create TextNode
            node = TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
                embedding=result.get("vector"),
            )

            nodes.append(node)
            # Convert distance to similarity (assuming L2 distance)
            # For L2: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
            similarities.append(similarity)
            ids.append(node_id)

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def _convert_filters(self, filters: Any) -> dict[str, Any] | None:
        """Convert LlamaIndex MetadataFilters to OmenDB filter format.

        Args:
            filters: LlamaIndex MetadataFilters object.

        Returns:
            OmenDB-compatible filter dictionary.
        """
        if filters is None:
            return None

        from llama_index.core.vector_stores.types import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        if not isinstance(filters, MetadataFilters):
            return None

        filter_list = []

        for f in filters.filters:
            if not isinstance(f, MetadataFilter):
                continue

            key = f.key
            value = f.value
            op = f.operator

            # Map LlamaIndex operators to OmenDB operators
            if op == FilterOperator.EQ:
                filter_list.append({key: value})
            elif op == FilterOperator.NE:
                filter_list.append({key: {"$ne": value}})
            elif op == FilterOperator.GT:
                filter_list.append({key: {"$gt": value}})
            elif op == FilterOperator.GTE:
                filter_list.append({key: {"$gte": value}})
            elif op == FilterOperator.LT:
                filter_list.append({key: {"$lt": value}})
            elif op == FilterOperator.LTE:
                filter_list.append({key: {"$lte": value}})
            elif op == FilterOperator.IN:
                filter_list.append({key: {"$in": value}})
            elif op == FilterOperator.CONTAINS:
                filter_list.append({key: {"$contains": value}})
            else:
                # Default to equality
                filter_list.append({key: value})

        if not filter_list:
            return None

        # Combine filters based on condition
        if len(filter_list) == 1:
            return filter_list[0]

        condition = getattr(filters, "condition", FilterCondition.AND)
        if condition == FilterCondition.OR:
            return {"$or": filter_list}
        else:
            return {"$and": filter_list}

    def flush(self) -> None:
        """Flush data to disk for persistence."""
        self._db.flush()
