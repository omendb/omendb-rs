"""OmenDB algorithm wrapper for ann-benchmarks."""

import numpy as np
import omendb

from ..base.module import BaseANN


class OmenDB(BaseANN):
    """OmenDB HNSW implementation for ann-benchmarks."""

    def __init__(self, metric, m=16, ef_construction=100):
        self._metric = metric
        self._m = m
        self._ef_construction = ef_construction
        self._ef = None
        self._db = None

    def fit(self, X):
        """Build index from training vectors."""
        self._db = omendb.open(
            ":memory:",
            dimensions=X.shape[1],
            m=self._m,
            ef_construction=self._ef_construction,
        )

        # Insert vectors with integer IDs matching array indices
        records = [{"id": str(i), "vector": vec.tolist()} for i, vec in enumerate(X)]
        self._db.set(records)

    def set_query_arguments(self, ef):
        """Set ef parameter for search."""
        self._ef = ef

    def query(self, v, n):
        """Return n nearest neighbor indices."""
        results = self._db.search(v.tolist(), k=n, ef=self._ef)
        return np.array([int(r["id"]) for r in results])

    def batch_query(self, X, n):
        """Batch query using native batch support."""
        queries = [v.tolist() for v in X]
        all_results = self._db.search_batch(queries, k=n, ef=self._ef)
        self._batch_res = [
            np.array([int(r["id"]) for r in results]) for results in all_results
        ]

    def get_batch_results(self):
        """Return batch query results."""
        return self._batch_res

    def __str__(self):
        return f"OmenDB(m={self._m}, ef_construction={self._ef_construction}, ef={self._ef})"

    def done(self):
        """Cleanup."""
        if self._db is not None:
            del self._db
            self._db = None


class OmenDBQuantized(BaseANN):
    """OmenDB with SQ8 quantization for ann-benchmarks."""

    def __init__(self, metric, m=16, ef_construction=100):
        self._metric = metric
        self._m = m
        self._ef_construction = ef_construction
        self._ef = None
        self._db = None

    def fit(self, X):
        """Build index with SQ8 quantization."""
        self._db = omendb.open(
            ":memory:",
            dimensions=X.shape[1],
            m=self._m,
            ef_construction=self._ef_construction,
            quantization="sq8",
            rescore=True,
        )

        records = [{"id": str(i), "vector": vec.tolist()} for i, vec in enumerate(X)]
        self._db.set(records)

    def set_query_arguments(self, ef):
        """Set ef parameter for search."""
        self._ef = ef

    def query(self, v, n):
        """Return n nearest neighbor indices."""
        results = self._db.search(v.tolist(), k=n, ef=self._ef)
        return np.array([int(r["id"]) for r in results])

    def batch_query(self, X, n):
        """Batch query using native batch support."""
        queries = [v.tolist() for v in X]
        all_results = self._db.search_batch(queries, k=n, ef=self._ef)
        self._batch_res = [
            np.array([int(r["id"]) for r in results]) for results in all_results
        ]

    def get_batch_results(self):
        """Return batch query results."""
        return self._batch_res

    def __str__(self):
        return f"OmenDB-SQ8(m={self._m}, ef_construction={self._ef_construction}, ef={self._ef})"

    def done(self):
        """Cleanup."""
        if self._db is not None:
            del self._db
            self._db = None
