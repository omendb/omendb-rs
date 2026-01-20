#!/usr/bin/env python3
"""
Release Recall Validation - runs before publishing.

Validation using 10K SIFT/GloVe subsets in benchmarks/data with precomputed
ground truth to keep CI fast and deterministic.

Time budget: ~90 seconds on CI
Exit code: 0 on success, 1 on failure

Tests:
  1. L2 baseline (SIFT-10K): recall@10 >= 97%
  2. Cosine baseline (GloVe-10K): recall@10 >= 88%
  3. SQ8 quantization (SIFT-10K): recall@10 >= 95%
  4. Filtered search (SIFT-10K, 50%): recall@10 >= 92%
  5. Persistence (SIFT-10K): recall unchanged after save/load

Note: GloVe has lower recall than SIFT at scale - this is expected due to
its more uniform angular distribution which is harder for HNSW.

Usage:
    python release_validation.py
    uv run python benchmarks/release_validation.py  # from python/
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import omendb

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "sift": {
        "file": "sift-10k.npz",
        "metric": "l2",
    },
    "glove": {
        "file": "glove-10k.npz",
        "metric": "cosine",
    },
}

NUM_QUERIES = int(os.getenv("OMENDB_RELEASE_QUERIES", "200"))
K = 10  # recall@K

# Test configurations
TESTS = [
    {
        "name": "L2 baseline",
        "dataset": "sift",
        "quantization": False,
        "filtered": False,
        "threshold": 0.97,
    },
    {
        "name": "Cosine baseline",
        "dataset": "glove",
        "quantization": False,
        "filtered": False,
        "threshold": 0.88,
    },
    {
        "name": "SQ8",
        "dataset": "sift",
        "quantization": "sq8",
        "filtered": False,
        "threshold": 0.95,
    },
    {
        "name": "Filtered",
        "dataset": "sift",
        "quantization": False,
        "filtered": True,
        "threshold": 0.92,
    },
]


def load_subset(name: str) -> dict:
    """Load dataset subset with precomputed ground truth."""
    config = DATASETS[name]
    path = DATA_DIR / config["file"]
    if not path.exists():
        print(f"ERROR: Dataset not found: {path}")
        print("Run benchmarks/create_subsets.py to generate the data files.")
        sys.exit(1)

    data = np.load(path)
    vectors = data["vectors"].astype(np.float32)
    queries = data["queries"][:NUM_QUERIES].astype(np.float32)
    ground_truth = data["ground_truth"][:NUM_QUERIES]
    metadata = data["metadata"]
    filtered_ground_truth = data["filtered_ground_truth"][:NUM_QUERIES]

    return {
        "vectors": vectors,
        "queries": queries,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "filtered_ground_truth": filtered_ground_truth,
        "metric": config["metric"],
    }


def compute_recall(results: list, ground_truth: np.ndarray) -> float:
    """Compute recall@K."""
    gt_set = set(ground_truth[:K].tolist())
    result_ids = set()
    for r in results[:K]:
        if isinstance(r, dict):
            result_ids.add(int(r["id"]))
        else:
            result_ids.add(int(r.id))
    return len(gt_set & result_ids) / K


def run_test(config: dict, datasets: dict) -> tuple[bool, float, float]:
    """Run a single validation test."""
    dataset = datasets[config["dataset"]]
    vectors = dataset["vectors"]
    queries = dataset["queries"]
    metadata = dataset["metadata"]
    metric = dataset["metric"]

    if config["filtered"]:
        ground_truth = dataset["filtered_ground_truth"]
    else:
        ground_truth = dataset["ground_truth"]

    dim = vectors.shape[1]

    # Build index
    db = omendb.open(
        ":memory:",
        dimensions=dim,
        metric=metric,
        quantization=config["quantization"],
    )

    # Insert vectors with metadata
    records = [
        {
            "id": str(i),
            "vector": vec.tolist(),
            "metadata": {"category": int(metadata[i])},
        }
        for i, vec in enumerate(vectors)
    ]
    db.set(records)

    # Run queries
    start = time.perf_counter()
    recalls = []

    for i, query in enumerate(queries):
        if config["filtered"]:
            results = db.search(query.tolist(), k=K, filter={"category": {"$lt": 5}})
        else:
            results = db.search(query.tolist(), k=K)

        recall = compute_recall(results, ground_truth[i])
        recalls.append(recall)

    elapsed = time.perf_counter() - start
    qps = len(queries) / elapsed
    mean_recall = np.mean(recalls)

    passed = mean_recall >= config["threshold"]
    return passed, mean_recall, qps


def run_persistence_test(datasets: dict) -> tuple[bool, float, float]:
    """Test that recall is maintained after save/load."""
    dataset = datasets["sift"]
    vectors = dataset["vectors"]
    queries = dataset["queries"]
    ground_truth = dataset["ground_truth"]

    dim = vectors.shape[1]
    threshold = 0.97  # Same as L2 baseline

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_db"

        # Build and save
        db = omendb.open(
            str(db_path),
            dimensions=dim,
            metric="l2",
        )

        records = [
            {"id": str(i), "vector": vec.tolist()} for i, vec in enumerate(vectors)
        ]
        db.set(records)

        # Flush to disk and reopen
        db.flush()
        del db
        db = omendb.open(str(db_path), dimensions=dim, metric="l2")

        # Run queries
        start = time.perf_counter()
        recalls = []

        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=K)
            recall = compute_recall(results, ground_truth[i])
            recalls.append(recall)

        elapsed = time.perf_counter() - start
        qps = len(queries) / elapsed
        mean_recall = np.mean(recalls)

        del db
        del db_path

    passed = mean_recall >= threshold
    return passed, mean_recall, qps


def main():
    print("=" * 60)
    print("OmenDB Release Recall Validation")
    print("=" * 60)
    print()

    start_time = time.perf_counter()

    # Load datasets
    print("Loading datasets...")
    datasets = {}
    for name in ["sift", "glove"]:
        print(f"\n{name}:")
        datasets[name] = load_subset(name)
    print()

    all_passed = True
    results = []

    # Run main tests
    for config in TESTS:
        print(f"Testing: {config['name']}...")
        passed, recall, qps = run_test(config, datasets)

        status = "PASS" if passed else "FAIL"
        threshold = config["threshold"]

        print(f"  Recall@{K}: {recall:.3f} (threshold: {threshold:.2f}) [{status}]")
        print(f"  QPS: {qps:.0f}")
        print()

        results.append(
            {
                "name": config["name"],
                "passed": passed,
                "recall": recall,
                "threshold": threshold,
                "qps": qps,
            }
        )

        if not passed:
            all_passed = False

    # Run persistence test
    print("Testing: Persistence...")
    passed, recall, qps = run_persistence_test(datasets)

    status = "PASS" if passed else "FAIL"
    threshold = 0.97

    print(f"  Recall@{K}: {recall:.3f} (threshold: {threshold:.2f}) [{status}]")
    print(f"  QPS: {qps:.0f}")
    print()

    results.append(
        {
            "name": "Persistence",
            "passed": passed,
            "recall": recall,
            "threshold": threshold,
            "qps": qps,
        }
    )

    if not passed:
        all_passed = False

    elapsed = time.perf_counter() - start_time

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"{'Test':<20} {'Recall':>8} {'Threshold':>10} {'QPS':>8} {'Status':>8}")
    print("-" * 60)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"{r['name']:<20} {r['recall']:>8.3f} {r['threshold']:>10.2f} "
            f"{r['qps']:>8.0f} {status:>8}"
        )

    print()
    print(f"Total time: {elapsed:.1f}s")
    print()

    if all_passed:
        print("All tests PASSED")
        return 0
    print("Some tests FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
