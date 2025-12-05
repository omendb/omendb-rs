#!/usr/bin/env python3
"""OmenDB Benchmark Suite

Comprehensive benchmark for performance and correctness testing.

Usage:
    python benchmark.py                    # Quick synthetic benchmark
    python benchmark.py --full             # Full multi-dimension benchmark
    python benchmark.py --sift1m           # SIFT1M correctness verification
    python benchmark.py --sift1m --full    # Full SIFT1M (1M vectors)

Modes:
    Synthetic (default): Random vectors, measures throughput at various dimensions
    SIFT1M: Industry-standard dataset with ground truth for recall verification

For production benchmarks, use Fedora (quieter, x86-64):
    ssh fedora 'cd omendb/python && python benchmark.py --full'
"""

import argparse
import struct
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

import omendb

# SIFT1M dataset URLs (HuggingFace mirror)
SIFT1M_URLS = {
    "base": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs",
    "query": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs",
    "groundtruth": "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs",
}
DATA_DIR = Path(__file__).parent / "benchmarks" / "data" / "sift1m"


# =============================================================================
# Utilities
# =============================================================================


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random vectors."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def read_fvecs(filename: str) -> np.ndarray:
    """Read .fvecs file format (float vectors)."""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors)


def read_ivecs(filename: str) -> np.ndarray:
    """Read .ivecs file format (ground truth indices)."""
    with open(filename, "rb") as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            vectors.append(vec)
        return np.array(vectors)


def download_sift1m(force: bool = False) -> bool:
    """Download SIFT1M dataset if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "sift_base.fvecs": SIFT1M_URLS["base"],
        "sift_query.fvecs": SIFT1M_URLS["query"],
        "sift_groundtruth.ivecs": SIFT1M_URLS["groundtruth"],
    }

    for filename, url in files.items():
        filepath = DATA_DIR / filename
        if filepath.exists() and not force:
            continue

        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    return True


def compute_recall(results_ids: list, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k against ground truth."""
    gt_set = set(int(x) for x in ground_truth[:k])
    found_set = set(int(r) for r in results_ids[:k])
    return len(gt_set & found_set) / k


# =============================================================================
# Synthetic Benchmarks
# =============================================================================


def benchmark_synthetic(
    n_vectors: int = 10000,
    dim: int = 128,
    n_queries: int = 1000,
    k: int = 10,
) -> dict:
    """Run synthetic benchmark with random vectors."""
    print(f"\n{'=' * 60}")
    print(f"Synthetic Benchmark: {n_vectors:,} vectors, {dim}D")
    print(f"{'=' * 60}")

    vectors = generate_vectors(n_vectors, dim)
    queries = generate_vectors(n_queries, dim, seed=999)

    results = {"n_vectors": n_vectors, "dim": dim}

    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(f"{tmpdir}/bench", dimensions=dim)

        # Build
        batch = [
            {"id": str(i), "vector": vectors[i].tolist(), "metadata": {"cat": i % 10}}
            for i in range(n_vectors)
        ]
        start = time.perf_counter()
        db.set(batch)
        build_time = time.perf_counter() - start
        build_rate = n_vectors / build_time
        print(f"\nBuild:      {build_rate:>10,.0f} vec/s  ({build_time:.2f}s)")
        results["build_vec_per_s"] = build_rate

        # Warmup
        for q in queries[:10]:
            db.search(q.tolist(), k=k)

        # Single search
        latencies = []
        start = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            db.search(q.tolist(), k=k)
            latencies.append((time.perf_counter() - t0) * 1000)
        search_time = time.perf_counter() - start
        search_qps = n_queries / search_time
        latencies.sort()
        print(
            f"Search:     {search_qps:>10,.0f} QPS    (p50={latencies[len(latencies) // 2]:.2f}ms, p99={latencies[int(len(latencies) * 0.99)]:.2f}ms)"
        )
        results["search_qps"] = search_qps
        results["search_p99_ms"] = latencies[int(len(latencies) * 0.99)]

        # Batch search
        queries_list = [q.tolist() for q in queries]
        start = time.perf_counter()
        db.search_batch(queries_list, k=k)
        batch_time = time.perf_counter() - start
        batch_qps = n_queries / batch_time
        print(f"Batch:      {batch_qps:>10,.0f} QPS    ({batch_time:.3f}s total)")
        results["batch_qps"] = batch_qps

        # Filtered search (10% selectivity)
        start = time.perf_counter()
        for q in queries:
            db.search(q.tolist(), k=k, filter={"cat": 5})
        filtered_time = time.perf_counter() - start
        filtered_qps = n_queries / filtered_time
        print(f"Filtered:   {filtered_qps:>10,.0f} QPS    (10% selectivity)")
        results["filtered_qps"] = filtered_qps

    return results


# =============================================================================
# SIFT1M Benchmark (Correctness Verification)
# =============================================================================


def benchmark_sift1m(
    n_vectors: int = None,  # None = full 1M
    n_queries: int = 1000,
    k: int = 10,
    ef_values: list = [10, 50, 100, 200],
) -> dict:
    """Run SIFT1M benchmark with ground truth verification."""
    print(f"\n{'=' * 60}")
    print("SIFT1M Benchmark (Ground Truth Verification)")
    print(f"{'=' * 60}")

    # Check/download dataset
    base_path = DATA_DIR / "sift_base.fvecs"
    query_path = DATA_DIR / "sift_query.fvecs"
    gt_path = DATA_DIR / "sift_groundtruth.ivecs"

    if not all(p.exists() for p in [base_path, query_path, gt_path]):
        print("\nDownloading SIFT1M dataset (~525 MB)...")
        if not download_sift1m():
            print("ERROR: Failed to download dataset")
            return {}

    print("\nLoading dataset...")
    base_vectors = read_fvecs(str(base_path))
    query_vectors = read_fvecs(str(query_path))
    ground_truth = read_ivecs(str(gt_path))

    # Subset if requested
    if n_vectors is not None and n_vectors < len(base_vectors):
        base_vectors = base_vectors[:n_vectors]
        print(f"Using {n_vectors:,} vectors (subset)")
        # Recompute ground truth for subset
        print("Computing ground truth for subset...")
        new_gt = []
        for query in query_vectors[:n_queries]:
            dists = np.sum((base_vectors - query) ** 2, axis=1)
            new_gt.append(np.argsort(dists)[:100])
        ground_truth = np.array(new_gt)

    query_vectors = query_vectors[:n_queries]
    ground_truth = ground_truth[:n_queries]

    n_base = len(base_vectors)
    dim = base_vectors.shape[1]
    print(f"  Base: {n_base:,} x {dim}D")
    print(f"  Queries: {len(query_vectors):,}")

    results = {"n_vectors": n_base, "dim": dim, "results": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"hnsw": {"m": 16, "ef_construction": 200, "ef_search": 100}}
        db = omendb.open(f"{tmpdir}/sift", dimensions=dim, config=config)

        # Build
        print("\nBuilding index...")
        start = time.perf_counter()
        batch_size = 10000
        for i in range(0, n_base, batch_size):
            end_idx = min(i + batch_size, n_base)
            batch = [
                {"id": str(j), "vector": base_vectors[j].tolist()}
                for j in range(i, end_idx)
            ]
            db.set(batch)
        build_time = time.perf_counter() - start
        print(f"  Build: {build_time:.1f}s ({n_base / build_time:.0f} vec/s)")
        results["build_time_s"] = build_time

        # Test at different ef values
        print(f"\n{'ef':>6} | {'Recall@' + str(k):>10} | {'QPS':>10} | Status")
        print("-" * 50)

        for ef in ef_values:
            db.set_ef_search(ef)

            recalls = []
            start = time.perf_counter()
            for i, query in enumerate(query_vectors):
                res = db.search(query.tolist(), k=k)
                ids = [int(r["id"]) for r in res]
                recalls.append(compute_recall(ids, ground_truth[i], k))
            elapsed = time.perf_counter() - start

            qps = len(query_vectors) / elapsed
            recall = np.mean(recalls)

            # Expected recall ranges
            expected = {10: 0.60, 50: 0.85, 100: 0.93, 200: 0.97}
            status = "PASS" if recall >= expected.get(ef, 0.90) else "WARN"

            print(f"{ef:>6} | {recall:>10.1%} | {qps:>10.0f} | {status}")
            results["results"].append({"ef": ef, "recall": recall, "qps": qps})

        # Batch search
        print("\nBatch search:")
        db.set_ef_search(100)
        queries_list = [q.tolist() for q in query_vectors]
        start = time.perf_counter()
        batch_results = db.search_batch(queries_list, k=k)
        batch_time = time.perf_counter() - start
        batch_qps = len(query_vectors) / batch_time

        batch_recalls = []
        for i, res in enumerate(batch_results):
            ids = [int(r["id"]) for r in res]
            batch_recalls.append(compute_recall(ids, ground_truth[i], k))
        batch_recall = np.mean(batch_recalls)
        print(f"  ef=100: Recall={batch_recall:.1%}, QPS={batch_qps:.0f}")

        results["batch_qps"] = batch_qps
        results["batch_recall"] = batch_recall

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="OmenDB Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py                    # Quick synthetic (10K, 128D)
    python benchmark.py --full             # Full multi-dimension
    python benchmark.py --sift1m           # SIFT1M verification (100K)
    python benchmark.py --sift1m --full    # Full SIFT1M (1M vectors)
    python benchmark.py -d 768 -n 50000    # Custom: 50K vectors, 768D
""",
    )
    parser.add_argument("--full", action="store_true", help="Full benchmark suite")
    parser.add_argument(
        "--sift1m", action="store_true", help="SIFT1M correctness benchmark"
    )
    parser.add_argument("-d", "--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument(
        "-n", "--vectors", type=int, default=10000, help="Number of vectors"
    )
    parser.add_argument(
        "-q", "--queries", type=int, default=1000, help="Number of queries"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OmenDB Benchmark Suite")
    print("=" * 60)

    if args.sift1m:
        if args.full:
            # Full 1M vectors
            benchmark_sift1m(n_vectors=None, n_queries=1000)
        else:
            # Quick 100K vectors
            benchmark_sift1m(n_vectors=100000, n_queries=100)
    elif args.full:
        # Multi-dimension synthetic
        for dim in [128, 384, 768, 1536]:
            benchmark_synthetic(n_vectors=10000, dim=dim, n_queries=1000)

        # Scale test at 768D (embedding dimension)
        print("\n" + "=" * 60)
        print("Scale Test (768D)")
        print("=" * 60)
        for n in [10000, 50000, 100000]:
            benchmark_synthetic(n_vectors=n, dim=768, n_queries=1000)
    else:
        # Quick single benchmark
        benchmark_synthetic(
            n_vectors=args.vectors, dim=args.dim, n_queries=args.queries
        )

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
