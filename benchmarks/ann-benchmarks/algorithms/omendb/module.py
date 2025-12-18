"""OmenDB algorithm wrapper for ann-benchmarks.

Usage:
    1. Copy this directory into ann-benchmarks/ann_benchmarks/algorithms/
    2. Run: python install.py --algorithm omendb
    3. Run: python run.py --algorithm omendb --dataset sift-128-euclidean
"""

import numpy as np

import omendb

try:
    from ..base.module import BaseANN
except ImportError:
    # Standalone usage - define minimal base class
    class BaseANN:
        def done(self):
            pass


class OmenDB(BaseANN):
    """OmenDB HNSW implementation for ann-benchmarks."""

    def __init__(self, metric, m=16, ef_construction=100, quantization=None):
        self._metric = metric
        self._m = m
        self._ef_construction = ef_construction
        self._quantization = quantization
        self._ef = None
        self._db = None

    def fit(self, X):
        """Build index from training vectors."""
        # Map ann-benchmarks metric names to omendb
        metric_map = {"euclidean": "l2", "angular": "cosine"}
        omendb_metric = metric_map.get(self._metric, self._metric)

        kwargs = {
            "dimensions": X.shape[1],
            "m": self._m,
            "ef_construction": self._ef_construction,
            "metric": omendb_metric,
        }
        if self._quantization:
            kwargs["quantization"] = self._quantization
            kwargs["rescore"] = True

        self._db = omendb.open(":memory:", **kwargs)

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
        q = f", quantization={self._quantization}" if self._quantization else ""
        return f"OmenDB(m={self._m}, ef_construction={self._ef_construction}, ef={self._ef}{q})"

    def done(self):
        """Cleanup."""
        if self._db is not None:
            del self._db
            self._db = None


class OmenDBQuantized(OmenDB):
    """OmenDB with SQ8 quantization for ann-benchmarks."""

    def __init__(self, metric, m=16, ef_construction=100):
        super().__init__(metric, m, ef_construction, quantization="sq8")

    def __str__(self):
        return f"OmenDB-SQ8(m={self._m}, ef_construction={self._ef_construction}, ef={self._ef})"
