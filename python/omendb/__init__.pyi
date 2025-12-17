"""Type stubs for omendb - Fast embedded vector database."""

from typing import Any, Iterator, Literal, Sequence, TypedDict, overload

from typing_extensions import Self

import numpy as np
import numpy.typing as npt

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

class HybridSearchResult(TypedDict):
    """Single hybrid search result."""

    id: str
    score: float
    metadata: dict[str, Any]

class VectorRecord(TypedDict, total=False):
    """Input record for set()."""

    id: str  # Required
    vector: list[float]  # Required
    metadata: dict[str, Any]
    text: str  # For hybrid search
    document: str  # Stored in metadata["document"]

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
    ) -> list[int]:
        """Insert single vector."""
        ...

    @overload
    def set(self, items: list[VectorRecord]) -> list[int]:
        """Insert batch of vectors."""
        ...

    @overload
    def set(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]] | VectorBatch,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
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
    ) -> list[int]:
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
            List of internal indices for stored vectors.

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
    ) -> list[SearchResult]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector (list or 1D numpy array).
            k: Number of nearest neighbors to return.
            ef: Search width override (default: auto-tuned).
            filter: MongoDB-style metadata filter.

        Returns:
            List of results with id, distance, metadata.

        Examples:
            >>> results = db.search([0.1, 0.2, 0.3], k=5)
            >>> results = db.search([...], k=10, filter={"category": "A"})
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

    def delete_where(self, filter: MetadataFilter) -> int:
        """Delete vectors matching a metadata filter.

        Evaluates the filter against all vectors and deletes those that match.
        Uses the same MongoDB-style filter syntax as search().

        Args:
            filter: MongoDB-style metadata filter.

        Returns:
            Number of vectors deleted.

        Examples:
            >>> db.delete_where({"status": "archived"})
            5
            >>> db.delete_where({"score": {"$lt": 0.5}})
            3
            >>> db.delete_where({"$and": [{"type": "draft"}, {"age": {"$gt": 30}}]})
            2
        """
        ...

    def update(
        self,
        id: str,
        vector: Vector,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update vector and/or metadata for existing ID.

        Args:
            id: Vector ID to update.
            vector: New vector data.
            metadata: New metadata (replaces existing).

        Raises:
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

    def len(self) -> int:
        """Number of vectors in database."""
        ...

    def __len__(self) -> int:
        """Number of vectors in database."""
        ...

    def is_empty(self) -> bool:
        """Check if database is empty."""
        ...

    def ids(self) -> list[str]:
        """List all vector IDs (without loading vector data).

        Efficient way to get all IDs for iteration, export, or debugging.

        Returns:
            List of all vector IDs in the database.
        """
        ...

    def items(self) -> list[GetResult]:
        """Get all items as list of dicts.

        Returns all vectors with their IDs and metadata. For large datasets,
        consider iterating with `for item in db:` instead.

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
        """Iterate over all items.

        Examples:
            >>> for item in db:
            ...     print(item["id"])
        """
        ...

    def get_many(self, ids: list[str]) -> list[GetResult | None]:
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

        Args:
            buffer_mb: Writer buffer size in MB (default: 50).
        """
        ...

    def has_text_search(self) -> bool:
        """Check if text search is enabled."""
        ...

    def set_with_text(self, items: list[VectorRecord]) -> list[int]:
        """Set vectors with text for hybrid search.

        Each item must have id, vector, and text fields.

        Returns:
            List of internal indices.
        """
        ...

    def search_text(self, query: str, k: int) -> list[TextSearchResult]:
        """Search using text only (BM25 scoring).

        Args:
            query: Text query.
            k: Number of results.

        Returns:
            List of results with id and score.
        """
        ...

    def search_hybrid(
        self,
        query_vector: Vector,
        query_text: str,
        k: int,
        filter: MetadataFilter | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """Hybrid search combining vector and text.

        Uses Reciprocal Rank Fusion (RRF) to combine results.

        Args:
            query_vector: Query embedding.
            query_text: Text query for BM25.
            k: Number of results.
            filter: Optional metadata filter.
            alpha: Vector vs text weight (0.0=text, 1.0=vector, default=0.5).
            rrf_k: RRF constant (default: 60).

        Returns:
            List of results with id, score, metadata.
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
