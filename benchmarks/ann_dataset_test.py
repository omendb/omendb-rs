#!/usr/bin/env python3
"""
Test OmenDB against real ann-benchmarks datasets.
Downloads datasets and runs recall/QPS benchmarks.

Usage:
    python ann_dataset_test.py --dataset sift-128-euclidean
    python ann_dataset_test.py --dataset glove-100-angular --test-adaptive
"""

import argparse
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np

import omendb

DATASETS = {
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-25-angular": "http://ann-benchmarks.com/glove-25-angular.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
}

DATA_DIR = Path(__file__).parent / "data"


def download_dataset(name: str) -> Path:
    """Download dataset if not already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.hdf5"

    if path.exists():
        print(f"Using cached dataset: {path}")
        return path

    url = DATASETS[name]
    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")
    return path


def load_dataset(path: Path):
    """Load dataset from HDF5 file."""
    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
        distances = np.array(f["distances"])
        metric = f.attrs.get("distance", "euclidean")
        if isinstance(metric, bytes):
            metric = metric.decode()
    return train, test, neighbors, distances, metric


def compute_recall(results: list, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    gt_set = set(ground_truth[:k])
    result_set = set(int(r["id"]) for r in results[:k])
    return len(gt_set & result_set) / k


def benchmark_omendb(
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    k: int = 10,
    ef_values: list = None,
    m: int = 16,
    ef_construction: int = 100,
    max_test: int = 1000,
    test_adaptive: bool = False,
):
    """Benchmark OmenDB on dataset."""
    if ef_values is None:
        ef_values = [10, 20, 40, 80, 100, 200]

    dim = train.shape[1]
    n_train = train.shape[0]
    n_test = min(test.shape[0], max_test)

    print(f"\n{'=' * 60}")
    print(f"OmenDB Benchmark: {n_train:,} train, {n_test:,} test, {dim}D")
    print(f"Config: m={m}, ef_construction={ef_construction}, k={k}")
    print(f"{'=' * 60}")

    # Build index
    print("\nBuilding index...")
    start = time.perf_counter()
    db = omendb.open(
        ":memory:",
        dimensions=dim,
        m=m,
        ef_construction=ef_construction,
    )

    # Insert in batches for progress
    batch_size = 10000
    for i in range(0, n_train, batch_size):
        end = min(i + batch_size, n_train)
        records = [{"id": str(j), "vector": train[j].tolist()} for j in range(i, end)]
        db.set(records)
        if (i // batch_size) % 10 == 0:
            print(f"  Inserted {end:,}/{n_train:,} vectors...")

    build_time = time.perf_counter() - start
    print(f"Build time: {build_time:.2f}s ({n_train / build_time:.0f} vec/s)")

    # Test queries
    test_subset = test[:n_test]
    gt_subset = neighbors[:n_test]

    print(f"\n{'ef':>6} | {'QPS':>8} | {'Recall@{}'.format(k):>10} | {'Latency':>10}")
    print("-" * 45)

    results = []
    for ef in ef_values:
        # Warm up
        for q in test_subset[:10]:
            db.search(q.tolist(), k=k, ef=ef)

        # Benchmark
        start = time.perf_counter()
        all_results = []
        for q in test_subset:
            res = db.search(q.tolist(), k=k, ef=ef)
            all_results.append(res)
        elapsed = time.perf_counter() - start

        # Compute metrics
        qps = n_test / elapsed
        recalls = [
            compute_recall(res, gt, k) for res, gt in zip(all_results, gt_subset)
        ]
        mean_recall = np.mean(recalls)
        latency_ms = elapsed * 1000 / n_test

        print(f"{ef:>6} | {qps:>8.0f} | {mean_recall:>10.3f} | {latency_ms:>8.2f}ms")
        results.append((ef, qps, mean_recall, latency_ms))

    # Test adaptive ef if requested
    if test_adaptive:
        print(f"\n{'=' * 60}")
        print("Testing adaptive ef (ef=None)...")
        print(f"{'=' * 60}")

        # Warm up
        for q in test_subset[:10]:
            db.search(q.tolist(), k=k)  # ef=None triggers adaptive

        # Benchmark adaptive
        start = time.perf_counter()
        all_results = []
        for q in test_subset:
            res = db.search(q.tolist(), k=k)  # ef=None
            all_results.append(res)
        elapsed = time.perf_counter() - start

        qps = n_test / elapsed
        recalls = [
            compute_recall(res, gt, k) for res, gt in zip(all_results, gt_subset)
        ]
        mean_recall = np.mean(recalls)
        latency_ms = elapsed * 1000 / n_test

        print(
            f"{'adapt':>6} | {qps:>8.0f} | {mean_recall:>10.3f} | {latency_ms:>8.2f}ms"
        )

        # Compare with fixed ef=100
        fixed_result = next((r for r in results if r[0] == 100), None)
        if fixed_result:
            fixed_qps = fixed_result[1]
            print(
                f"\nAdaptive vs fixed ef=100: {(qps / fixed_qps - 1) * 100:+.1f}% QPS"
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test OmenDB on ann-benchmarks datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="sift-128-euclidean",
        help="Dataset to test",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--m", type=int, default=16, help="HNSW M parameter")
    parser.add_argument(
        "--ef-construction", type=int, default=100, help="HNSW ef_construction"
    )
    parser.add_argument(
        "--max-test", type=int, default=1000, help="Max test queries to run"
    )
    parser.add_argument(
        "--test-adaptive",
        action="store_true",
        help="Also test adaptive ef selection",
    )
    args = parser.parse_args()

    # Download and load dataset
    path = download_dataset(args.dataset)
    train, test, neighbors, distances, metric = load_dataset(path)

    print(f"\nDataset: {args.dataset}")
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")
    print(f"  Metric: {metric}")

    # Run benchmark
    benchmark_omendb(
        train,
        test,
        neighbors,
        k=args.k,
        m=args.m,
        ef_construction=args.ef_construction,
        max_test=args.max_test,
        test_adaptive=args.test_adaptive,
    )


if __name__ == "__main__":
    main()
