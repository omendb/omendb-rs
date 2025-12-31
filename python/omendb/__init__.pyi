"""Type stubs for omendb - Fast embedded vector database."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypedDict, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

# Type aliases for vectors
Vector = Sequence[float] | npt.NDArray[np.floating[Any]]
VectorBatch = Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]]

class SearchResult(TypedDict):
    """Single search result."""

    id: str
    distance: float
    metadata: dict[str, Any]

class TextSearchResult(TypedDict):
    """Single text search result."""

    id: str
    score: float
    metadata: dict[str, Any]

class HybridSearchResult(TypedDict):
    """Single hybrid search result."""

    id: str
    score: float
    metadata: dict[str, Any]

class HybridSearchResultWithSubscores(TypedDict):
    """Hybrid search result with separate keyword and semantic scores."""

    id: str
    score: float
    metadata: dict[str, Any]
    keyword_score: float | None  # BM25 score (None if only matched vector search)
    semantic_score: float | None  # Vector distance (None if only matched text search)

class VectorRecord(TypedDict, total=False):
    """Input record for set()."""

    id: str  # Required
    vector: list[float]  # Required
    metadata: dict[str, Any]
    text: str  # For hybrid search - indexed AND auto-stored in metadata["text"]

class GetResult(TypedDict):
    """Result from get()."""

    id: str
    vector: list[float]
    metadata: dict[str, Any]

class StatsResult(TypedDict):
    """Database statistics."""

    dimensions: int
    count: int
    path: str

# Filter types for MongoDB-style queries
FilterValue = str | int | float | bool | None | list[Any] | dict[str, Any]
FilterOperator = TypedDict(
    "FilterOperator",
    {
        "$eq": FilterValue,
        "$ne": FilterValue,
        "$gt": float,
        "$gte": float,
        "$lt": float,
        "$lte": float,
        "$in": list[FilterValue],
        "$contains": str,
    },
    total=False,
)
MetadataFilter = dict[str, FilterValue | FilterOperator]

class VectorDatabase:
    """High-performance embedded vector database.

    Provides fast similarity search using HNSW indexing with:
    - ~19,000 QPS @ 10K vectors with 100% recall
    - 20,000-28,000 vec/s insert throughput
    - Extended RaBitQ 8x compression
    - ACORN-1 filtered search (37.79x speedup)

    Supports context manager protocol for automatic cleanup.
    """

    @property
    def dimensions(self) -> int:
        """Vector dimensionality of this database."""
        ...

    # Set methods with multiple signatures
    @overload
    def set(
        self,
        id: str,
        vector: Vector,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert single vector."""
        ...

    @overload
    def set(self, items: list[VectorRecord]) -> int:
        """Insert batch of vectors."""
        ...

    @overload
    def set(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]] | VectorBatch,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert batch using kwargs."""
        ...

    def set(
        self,
        id_or_items: str | list[VectorRecord] | None = None,
        vector: Vector | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        ids: list[str] | None = None,
        vectors: list[list[float]] | VectorBatch | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Set (insert or replace) vectors.

        Supports multiple input formats:
        - Single: set("id", [0.1, 0.2], {"key": "value"})
        - Batch list: set([{"id": "a", "vector": [...], "metadata": {...}}])
        - Batch kwargs: set(ids=["a"], vectors=[[...]], metadatas=[{...}])

        Args:
            id_or_items: Vector ID (str) or list of VectorRecord dicts
            vector: Vector data (required when id_or_items is str)
            metadata: Optional metadata dict
            ids: List of IDs (batch kwargs format)
            vectors: List of vectors (batch kwargs format)
            metadatas: List of metadata dicts (batch kwargs format)

        Returns:
            Number of vectors inserted/updated.

        Raises:
            ValueError: If required fields missing or dimensions mismatch.
        """
        ...

    def search(
        self,
        query: Vector,
        k: int,
        ef: int | None = None,
        filter: MetadataFilter | None = None,
        max_distance: float | None = None,
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector (list or 1D numpy array).
            k: Number of nearest neighbors to return.
            ef: Search width override (default: auto-tuned).
            filter: MongoDB-style metadata filter.
            max_distance: Maximum distance threshold (filters out distant results).

        Returns:
            List of results with id, distance, metadata.

        Examples:
            >>> results = db.search([0.1, 0.2, 0.3], k=5)
            >>> results = db.search([...], k=10, filter={"category": "A"})
            >>> results = db.search([...], k=10, max_distance=0.5)
        """
        ...

    def search_batch(
        self,
        queries: VectorBatch,
        k: int,
        ef: int | None = None,
    ) -> list[list[SearchResult]]:
        """Batch search multiple queries with parallel execution.

        Args:
            queries: 2D numpy array or list of query vectors.
            k: Number of nearest neighbors per query.
            ef: Search width override.

        Returns:
            List of results for each query.
        """
        ...

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            Number of vectors deleted.
        """
        ...

    def delete_by_filter(self, filter: MetadataFilter) -> int:
        """Delete vectors matching a metadata filter.

        Evaluates the filter against all vectors and deletes those that match.
        Uses the same MongoDB-style filter syntax as search().

        Args:
            filter: MongoDB-style metadata filter.

        Returns:
            Number of vectors deleted.

        Examples:
            >>> db.delete_by_filter({"status": "archived"})
            5
            >>> db.delete_by_filter({"score": {"$lt": 0.5}})
            3
            >>> db.delete_by_filter({"$and": [{"type": "draft"}, {"age": {"$gt": 30}}]})
            2
        """
        ...

    def count(self, filter: MetadataFilter | None = None) -> int:
        """Count vectors, optionally filtered by metadata.

        Without a filter, returns total count (same as len(db)).
        With a filter, returns count of vectors matching the filter.

        Args:
            filter: Optional MongoDB-style metadata filter.

        Returns:
            Number of vectors (matching filter if provided).

        Examples:
            >>> db.count()
            1000
            >>> db.count(filter={"status": "active"})
            750
            >>> db.count(filter={"score": {"$gte": 0.8}})
            250
        """
        ...

    def update(
        self,
        id: str,
        vector: Vector | None = None,
        metadata: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> None:
        """Update vector, metadata, and/or text for existing ID.

        At least one of vector, metadata, or text must be provided.

        Args:
            id: Vector ID to update.
            vector: New vector data (optional).
            metadata: New metadata (replaces existing, optional).
            text: New text for hybrid search (re-indexed for BM25, optional).

        Raises:
            ValueError: If no update parameters provided.
            RuntimeError: If vector with given ID doesn't exist.
        """
        ...

    def get(self, id: str) -> GetResult | None:
        """Get vector by ID.

        Args:
            id: Vector ID to retrieve.

        Returns:
            Dict with id, vector, metadata or None if not found.
        """
        ...

    def get_ef_search(self) -> int:
        """Get current ef_search value."""
        ...

    def set_ef_search(self, ef_search: int) -> None:
        """Set ef_search value for search quality/speed tradeoff."""
        ...

    def optimize(self) -> int:
        """Optimize index for cache-efficient search.

        Returns:
            Number of nodes reordered.
        """
        ...

    def __len__(self) -> int:
        """Number of vectors in database."""
        ...

    def is_empty(self) -> bool:
        """Check if database is empty."""
        ...

    def ids(self) -> Iterator[str]:
        """Iterate over all vector IDs (without loading vector data).

        Returns a lazy iterator. Use `list(db.ids())` if you need all IDs at once.
        Memory efficient for large datasets.

        Returns:
            Iterator over all vector IDs in the database.

        Examples:
            >>> for id in db.ids():
            ...     print(id)
            >>> all_ids = list(db.ids())  # Get as list if needed
        """
        ...

    def items(self) -> list[GetResult]:
        """Get all items as list of dicts.

        WARNING: Loads all vectors into memory. For 1M vectors at 768D,
        this uses ~3GB RAM. For large datasets, use `for item in db:` which
        is lazy, or use `ids()` + `get_batch()` with batching.

        Returns:
            List of {"id": str, "vector": list[float], "metadata": dict}
        """
        ...

    def exists(self, id: str) -> bool:
        """Check if an ID exists in the database.

        Args:
            id: Vector ID to check.

        Returns:
            True if ID exists and is not deleted.
        """
        ...

    def __contains__(self, id: str) -> bool:
        """Support `in` operator: `"id" in db`"""
        ...

    def __iter__(self) -> Iterator[GetResult]:
        """Iterate over all items lazily.

        Memory efficient: stores only IDs (~20MB for 1M items), fetches
        vectors one at a time. Handles deletions during iteration gracefully.

        For export/migration of small datasets, `items()` is more convenient.

        Examples:
            >>> for item in db:
            ...     print(item["id"])
            >>> # Early termination is efficient
            >>> for i, item in enumerate(db):
            ...     if i >= 10: break
        """
        ...

    def get_batch(self, ids: list[str]) -> list[GetResult | None]:
        """Get multiple vectors by ID.

        Batch version of get(). More efficient than calling get() in a loop.

        Args:
            ids: List of vector IDs to retrieve.

        Returns:
            List of results in same order as input. None for missing IDs.
        """
        ...

    def stats(self) -> StatsResult:
        """Get database statistics."""
        ...

    def flush(self) -> None:
        """Flush pending changes to disk."""
        ...

    def merge_from(self, other: VectorDatabase) -> int:
        """Merge vectors from another database.

        Args:
            other: Source database to merge from.

        Returns:
            Number of vectors merged.
        """
        ...

    # Collections
    def collection(self, name: str) -> VectorDatabase:
        """Create or get a named collection.

        Args:
            name: Collection name (alphanumeric and underscores).

        Returns:
            VectorDatabase instance for this collection.

        Raises:
            ValueError: If name is invalid or db is in-memory.
        """
        ...

    def collections(self) -> list[str]:
        """List all collection names."""
        ...

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name to delete.

        Raises:
            ValueError: If collection doesn't exist.
        """
        ...

    # Hybrid search
    def enable_text_search(self, buffer_mb: int | None = None) -> None:
        """Enable text search for hybrid search.

        Note: Called automatically when using set() with items that have a `text` field.
        Only call manually if you need custom buffer_mb config.

        Args:
            buffer_mb: Writer buffer size in MB (default: 50).
        """
        ...

    def has_text_search(self) -> bool:
        """Check if text search is enabled."""
        ...

    def search_text(self, query: str, k: int) -> list[TextSearchResult]:
        """Search using text only (BM25 scoring).

        Args:
            query: Text query.
            k: Number of results.

        Returns:
            List of results with id, score, and metadata.
        """
        ...

    @overload
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        k: int,
        filter: MetadataFilter | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
        subscores: Literal[False] | None = None,
    ) -> list[HybridSearchResult]:
        """Hybrid search combining vector and text."""
        ...

    @overload
    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        k: int,
        filter: MetadataFilter | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
        subscores: Literal[True] = ...,
    ) -> list[HybridSearchResultWithSubscores]:
        """Hybrid search with separate keyword and semantic scores."""
        ...

    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        k: int,
        filter: MetadataFilter | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
        subscores: bool | None = None,
    ) -> list[HybridSearchResult] | list[HybridSearchResultWithSubscores]:
        """Hybrid search combining vector and text.

        Uses Reciprocal Rank Fusion (RRF) to combine results.

        Args:
            query_vector: Query embedding.
            query_text: Text query for BM25.
            k: Number of results.
            filter: Optional metadata filter.
            alpha: Vector vs text weight (0.0=text, 1.0=vector, default=0.5).
            rrf_k: RRF constant (default: 60).
            subscores: Return separate keyword_score and semantic_score (default: False).

        Returns:
            List of results with id, score, metadata.
            When subscores=True, also includes keyword_score and semantic_score.

        Examples:
            >>> results = db.search_hybrid(vec, "query", k=10)
            >>> results = db.search_hybrid(vec, "query", k=10, subscores=True)
            >>> for r in results:
            ...     print(f"{r['id']}: keyword={r['keyword_score']}, semantic={r['semantic_score']}")
        """
        ...

    # Context manager
    def __enter__(self) -> Self:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context manager, flush changes."""
        ...

def open(
    path: str,
    dimensions: int,
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    quantization: bool | Literal["sq8", "rabitq"] | None = None,
    rescore: bool | None = None,
    oversample: float | None = None,
    metric: Literal["l2", "euclidean", "cosine", "dot", "ip"] | None = None,
    config: dict[str, Any] | None = None,
) -> VectorDatabase:
    """Open or create a vector database.

    Args:
        path: Database path, or ":memory:" for in-memory.
        dimensions: Vector dimensionality.
        m: HNSW neighbors per node (default: 16, range: 4-64).
        ef_construction: Build quality (default: 100).
        ef_search: Search quality (default: 100).
        quantization: Enable quantization:
            - True or "sq8": 4x smaller, ~99% recall (recommended)
            - "rabitq": 8x smaller, ~98% recall
            - None/False: Full precision
        rescore: Rerank with full precision (default: True when quantized).
        oversample: Candidate multiplier for rescoring (default: 3.0).
        metric: Distance metric for similarity search (default: "l2"):
            - "l2" or "euclidean": Euclidean distance (default)
            - "cosine": Cosine distance (1 - cosine similarity)
            - "dot" or "ip": Inner product (for MIPS)
        config: Advanced config dict (deprecated).

    Returns:
        VectorDatabase instance.

    Examples:
        >>> db = omendb.open("./vectors", dimensions=768)
        >>> db = omendb.open("./vectors", dimensions=768, quantization=True)
        >>> db = omendb.open(":memory:", dimensions=128)
    """
    ...

__version__: str
__all__: list[str]
