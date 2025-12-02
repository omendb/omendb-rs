#!/usr/bin/env python3
"""
SIFT1M Standard Benchmark

Runs OmenDB against the SIFT1M dataset with ground truth verification.
This is the industry-standard benchmark for ANN algorithms.

Dataset: 1M 128D SIFT descriptors
- Base: 1,000,000 vectors (128D)
- Queries: 10,000 vectors
- Ground truth: 100 nearest neighbors per query

Source: http://corpus-texmex.irisa.fr/
Mirror: https://huggingface.co/datasets/qbo-odp/sift1m
"""

import os
import sys
import time
import struct
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# Add omendb to path (script is in omendb/python/benchmarks/)
sys.path.insert(0, str(Path(__file__).parent.parent))
import omendb

# Dataset configuration
SIFT1M_URLS = {
    "base": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs",
    "query": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs",
    "groundtruth": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs",
}

DATA_DIR = Path(__file__).parent / "data" / "sift1m"


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs file format (float vectors)"""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            # Read dimension (4 bytes, int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            # Read vector (dim * 4 bytes, float32)
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors)


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs file format (integer vectors for ground truth)"""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            # Read dimension (4 bytes, int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            # Read vector (dim * 4 bytes, int32)
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            vectors.append(vec)
        return np.array(vectors)


def download_dataset(force: bool = False) -> bool:
    """Download SIFT1M dataset if not present"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "sift_base.fvecs": SIFT1M_URLS["base"],
        "sift_query.fvecs": SIFT1M_URLS["query"],
        "sift_groundtruth.ivecs": SIFT1M_URLS["groundtruth"],
    }

    for filename, url in files.items():
        filepath = DATA_DIR / filename
        if filepath.exists() and not force:
            print(f"  {filename}: exists ({filepath.stat().st_size / 1e6:.1f} MB)")
            continue

        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  {filename}: downloaded ({filepath.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")
            return False

    return True


def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """Compute exact k-NN ground truth using brute force"""
    print(f"  Computing ground truth for {len(query_vectors)} queries...")
    ground_truth = []
    for query in query_vectors:
        # L2 distance
        dists = np.sum((base_vectors - query) ** 2, axis=1)
        neighbors = np.argsort(dists)[:k]
        ground_truth.append(neighbors)
    return np.array(ground_truth)


def compute_recall(results: list, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k against ground truth"""
    gt_set = set(ground_truth[:k])
    found_set = set(int(r["id"]) for r in results[:k])
    return len(gt_set & found_set) / k


def run_benchmark(
    n_vectors: Optional[int] = None,
    n_queries: int = 100,
    k: int = 10,
    ef_values: list = [10, 20, 50, 100, 200],
    m: int = 16,
    ef_construction: int = 200,
) -> dict:
    """
    Run SIFT1M benchmark with ground truth verification

    Args:
        n_vectors: Number of base vectors to use (None = all 1M)
        n_queries: Number of queries to run
        k: Number of neighbors to retrieve
        ef_values: ef_search values to test
        m: HNSW M parameter
        ef_construction: HNSW ef_construction parameter

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "=" * 60)
    print("SIFT1M Standard Benchmark")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    base_path = DATA_DIR / "sift_base.fvecs"
    query_path = DATA_DIR / "sift_query.fvecs"
    gt_path = DATA_DIR / "sift_groundtruth.ivecs"

    if not all(p.exists() for p in [base_path, query_path, gt_path]):
        print("Dataset not found. Downloading...")
        if not download_dataset():
            raise RuntimeError("Failed to download dataset")

    base_vectors = read_fvecs(str(base_path))
    query_vectors = read_fvecs(str(query_path))
    ground_truth = read_ivecs(str(gt_path))

    # Limit vectors if specified
    if n_vectors is not None and n_vectors < len(base_vectors):
        base_vectors = base_vectors[:n_vectors]
        print(f"Using {n_vectors:,} vectors (subset)")
        # Recompute ground truth locally for subset
        query_vectors = query_vectors[:n_queries]
        ground_truth = compute_ground_truth(base_vectors, query_vectors, k=100)
    else:
        query_vectors = query_vectors[:n_queries]
        ground_truth = ground_truth[:n_queries]

    n_base = len(base_vectors)
    dim = base_vectors.shape[1]
    n_q = len(query_vectors)

    print(f"  Base vectors: {n_base:,} x {dim}D")
    print(f"  Query vectors: {n_q:,}")
    print(f"  k={k}")

    # Build index
    print(f"\nBuilding index (M={m}, ef_construction={ef_construction})...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "sift_bench")

        config = {
            "hnsw": {
                "m": m,
                "ef_construction": ef_construction,
            }
        }

        start = time.perf_counter()
        db = omendb.open(db_path, dimensions=dim, config=config)

        # Insert in batches
        batch_size = 10000
        for i in range(0, n_base, batch_size):
            end_idx = min(i + batch_size, n_base)
            batch = [
                {"id": str(j), "embedding": base_vectors[j].tolist()}
                for j in range(i, end_idx)
            ]
            db.set(batch)
            if (i + batch_size) % 100000 == 0 or end_idx == n_base:
                print(f"  Inserted {end_idx:,}/{n_base:,} vectors")

        build_time = time.perf_counter() - start
        print(f"  Build time: {build_time:.2f}s ({n_base/build_time:.0f} vectors/sec)")

        # Run benchmarks at different ef values
        results = {
            "dataset": "SIFT1M",
            "n_vectors": n_base,
            "dimensions": dim,
            "n_queries": n_q,
            "k": k,
            "m": m,
            "ef_construction": ef_construction,
            "build_time_s": build_time,
            "results": [],
        }

        print(f"\n{'ef':>6} | {'Recall@'+str(k):>10} | {'QPS':>10} | {'Latency':>12}")
        print("-" * 50)

        for ef in ef_values:
            db.set_ef_search(ef)

            # Warm up
            _ = db.search(query_vectors[0].tolist(), k=k)

            # Run queries and measure
            recalls = []
            start = time.perf_counter()

            for i, query in enumerate(query_vectors):
                query_results = db.search(query.tolist(), k=k)
                recall = compute_recall(query_results, ground_truth[i], k)
                recalls.append(recall)

            elapsed = time.perf_counter() - start
            qps = n_q / elapsed
            mean_recall = np.mean(recalls)
            latency_ms = (elapsed / n_q) * 1000

            print(f"{ef:>6} | {mean_recall:>10.1%} | {qps:>10.0f} | {latency_ms:>10.2f}ms")

            results["results"].append({
                "ef": ef,
                "recall": mean_recall,
                "qps": qps,
                "latency_ms": latency_ms,
            })

        # Also test batch search
        print("\nBatch search performance:")
        db.set_ef_search(100)  # Standard ef for batch test

        queries_list = [q.tolist() for q in query_vectors]
        start = time.perf_counter()
        batch_results = db.search_batch(queries_list, k=k)
        batch_elapsed = time.perf_counter() - start
        batch_qps = n_q / batch_elapsed

        # Compute batch recall
        batch_recalls = []
        for i, res in enumerate(batch_results):
            recall = compute_recall(res, ground_truth[i], k)
            batch_recalls.append(recall)
        batch_recall = np.mean(batch_recalls)

        print(f"  ef=100: Recall={batch_recall:.1%}, QPS={batch_qps:.0f}")

        results["batch_results"] = {
            "ef": 100,
            "recall": batch_recall,
            "qps": batch_qps,
        }

    return results


def verify_correctness(n_vectors: int = 10000, k: int = 10) -> bool:
    """
    Verify OmenDB correctness against SIFT1M ground truth

    Uses a small subset for quick verification.
    Returns True if recall > 90% at ef=200 (expected for HNSW).
    """
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    results = run_benchmark(
        n_vectors=n_vectors,
        n_queries=100,
        k=k,
        ef_values=[200],  # High ef for accuracy
        m=16,
        ef_construction=200,
    )

    recall = results["results"][0]["recall"]
    passed = recall >= 0.90

    print(f"\n{'PASSED' if passed else 'FAILED'}: Recall@{k} = {recall:.1%} (threshold: 90%)")
    return passed


def print_expected_results():
    """Print expected recall ranges for SIFT1M"""
    print("""
Expected Recall@10 for SIFT1M (HNSW, M=16, ef_construction=200):

| ef_search | Expected Recall |
|-----------|-----------------|
| 10        | 60-75%          |
| 20        | 75-85%          |
| 50        | 85-93%          |
| 100       | 93-97%          |
| 200       | 97-99%          |

Lower recall indicates potential issues with:
- Distance calculation
- HNSW graph construction
- Neighbor selection heuristics

Higher recall than expected is fine (better algorithm!).
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SIFT1M benchmark for OmenDB")
    parser.add_argument("--download", action="store_true", help="Download dataset only")
    parser.add_argument("--verify", action="store_true", help="Quick correctness check")
    parser.add_argument("--full", action="store_true", help="Full 1M benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick 10K benchmark")
    parser.add_argument("-n", type=int, default=100000, help="Number of vectors")
    parser.add_argument("-q", type=int, default=100, help="Number of queries")
    parser.add_argument("-k", type=int, default=10, help="Number of neighbors")

    args = parser.parse_args()

    if args.download:
        print("Downloading SIFT1M dataset...")
        download_dataset()
    elif args.verify:
        passed = verify_correctness()
        sys.exit(0 if passed else 1)
    elif args.full:
        run_benchmark(n_vectors=None, n_queries=1000, k=10)
    elif args.quick:
        run_benchmark(n_vectors=10000, n_queries=100, k=10)
    else:
        run_benchmark(n_vectors=args.n, n_queries=args.q, k=args.k)
