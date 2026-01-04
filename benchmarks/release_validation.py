#!/usr/bin/env python3
"""
Release Recall Validation - runs before publishing.

Comprehensive validation using 50K subsets from SIFT and GloVe datasets.
Downloads datasets on first run (cached to ~/.cache/omendb/).

Time budget: ~5 minutes on CI
Exit code: 0 on success, 1 on failure

Tests:
  1. L2 baseline (SIFT-50K): recall@10 >= 97%
  2. Cosine baseline (GloVe-50K): recall@10 >= 88%
  3. SQ8 quantization (SIFT-50K): recall@10 >= 88%
  4. RaBitQ (SIFT-50K): recall@10 >= 93%
  5. Filtered search (SIFT-50K, 50%): recall@10 >= 92%
  6. Persistence (SIFT-50K): recall unchanged after save/load

Note: GloVe has lower recall than SIFT at scale - this is expected due to
its more uniform angular distribution which is harder for HNSW.

Usage:
    python release_validation.py
    uv run python benchmarks/release_validation.py  # from python/
"""

import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

import omendb

CACHE_DIR = Path.home() / ".cache" / "omendb"

DATASETS = {
    "sift": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "file": "sift-128-euclidean.hdf5",
        "metric": "l2",
    },
    "glove": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "file": "glove-100-angular.hdf5",
        "metric": "cosine",
    },
}

SUBSET_SIZE = 50_000
NUM_QUERIES = 500
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
        "name": "RaBitQ",
        "dataset": "sift",
        "quantization": "rabitq",
        "filtered": False,
        "threshold": 0.93,
    },
    {
        "name": "Filtered",
        "dataset": "sift",
        "quantization": False,
        "filtered": True,
        "threshold": 0.92,
    },
]


def download_dataset(name: str) -> Path:
    """Download dataset if not already cached."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config = DATASETS[name]
    path = CACHE_DIR / config["file"]

    if path.exists():
        print(f"  Using cached: {path}")
        return path

    url = config["url"]
    print(f"  Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")
    return path


def compute_ground_truth_l2(
    vectors: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Compute brute-force L2 KNN ground truth using argpartition (O(n) vs O(n log n))."""
    gt = np.zeros((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        distances = np.sum((vectors - q) ** 2, axis=1)  # Squared L2, skip sqrt
        # argpartition is O(n), finds k smallest without full sort
        top_k_unsorted = np.argpartition(distances, k)[:k]
        # Sort only the k candidates
        gt[i] = top_k_unsorted[np.argsort(distances[top_k_unsorted])]
    return gt


def compute_ground_truth_cosine(
    vectors: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Compute brute-force cosine KNN ground truth using argpartition (O(n) vs O(n log n))."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vectors_norm = vectors / norms

    gt = np.zeros((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        q_norm = q / np.linalg.norm(q) if np.linalg.norm(q) > 0 else q
        similarities = vectors_norm @ q_norm
        # argpartition on negated similarities for top-k largest
        top_k_unsorted = np.argpartition(-similarities, k)[:k]
        gt[i] = top_k_unsorted[np.argsort(-similarities[top_k_unsorted])]
    return gt


def load_subset(name: str) -> dict:
    """Load and prepare dataset subset with ground truth."""
    path = download_dataset(name)
    config = DATASETS[name]

    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])

    # Take subset
    vectors = train[:SUBSET_SIZE].astype(np.float32)
    queries = test[:NUM_QUERIES].astype(np.float32)

    print(f"  Computing ground truth for {len(vectors):,} vectors...")

    # Compute ground truth
    if config["metric"] == "l2":
        ground_truth = compute_ground_truth_l2(vectors, queries, K)
    else:
        ground_truth = compute_ground_truth_cosine(vectors, queries, K)

    # Synthetic metadata for filtered search
    metadata = (np.arange(len(vectors)) % 10).astype(np.int32)

    # Filtered ground truth
    mask = metadata < 5
    filtered_indices = np.where(mask)[0]
    filtered_vectors = vectors[mask]

    if config["metric"] == "l2":
        filtered_gt_local = compute_ground_truth_l2(filtered_vectors, queries, K)
    else:
        filtered_gt_local = compute_ground_truth_cosine(filtered_vectors, queries, K)

    filtered_ground_truth = filtered_indices[filtered_gt_local]

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

        # Clean up
        del db
        shutil.rmtree(db_path, ignore_errors=True)

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
    else:
        print("Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
