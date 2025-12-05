#!/usr/bin/env python3
"""OmenDB Performance Benchmark

Measures build throughput, search QPS, and filtered search performance
at multiple embedding dimensions and dataset sizes.

Usage:
    python benchmark.py              # Quick benchmark (10K vectors)
    python benchmark.py --full       # Full benchmark (10K, 50K, 100K)
    python benchmark.py --dimension 1536  # Specific dimension
"""

import argparse
import platform
import subprocess
import tempfile
import time
from datetime import datetime

import numpy as np

import omendb


def get_benchmark_metadata() -> dict:
    """Get system and version info for reproducible benchmarks."""
    # Git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    return {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "omendb_version": getattr(omendb, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }


def print_metadata(metadata: dict):
    """Print benchmark metadata header."""
    print(f"Commit:   {metadata['commit']}")
    print(f"Version:  {metadata['omendb_version']}")
    print(f"Python:   {metadata['python']}")
    print(f"Platform: {metadata['platform']}")
    print(f"CPU:      {metadata['cpu']}")
    print(f"Time:     {metadata['timestamp']}")


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random vectors."""
    np.random.seed(seed)
    return np.random.randn(n, dim).astype(np.float32)


def benchmark_build(
    db_path: str, vectors: np.ndarray, with_metadata: bool = True
) -> dict:
    """Benchmark index build throughput."""
    n, dim = vectors.shape
    db = omendb.open(db_path, dimensions=dim)

    if with_metadata:
        batch = [
            {
                "id": f"d{i}",
                "vector": vectors[i].tolist(),
                "metadata": {"cat": i % 10},
            }
            for i in range(n)
        ]
    else:
        batch = [{"id": f"d{i}", "vector": vectors[i].tolist()} for i in range(n)]

    start = time.time()
    db.set(batch)
    elapsed = time.time() - start

    return {
        "vectors": n,
        "time_s": elapsed,
        "vec_per_s": n / elapsed,
        "db": db,
    }


def benchmark_search(db, queries: np.ndarray, k: int = 10, warmup: int = 10) -> dict:
    """Benchmark search QPS and latency."""
    n_queries = len(queries)

    # Warmup
    for q in queries[:warmup]:
        db.search(q.tolist(), k=k)

    # Benchmark
    latencies = []
    start = time.time()
    for q in queries:
        t0 = time.time()
        db.search(q.tolist(), k=k)
        latencies.append((time.time() - t0) * 1000)
    total = time.time() - start

    latencies.sort()
    return {
        "queries": n_queries,
        "time_s": total,
        "qps": n_queries / total,
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": latencies[len(latencies) // 2],
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)],
    }


def benchmark_filtered_search(
    db, queries: np.ndarray, filter_dict: dict, k: int = 10, warmup: int = 10
) -> dict:
    """Benchmark filtered search performance."""
    n_queries = len(queries)

    # Warmup
    for q in queries[:warmup]:
        db.search(q.tolist(), k=k, filter=filter_dict)

    # Benchmark
    start = time.time()
    for q in queries:
        db.search(q.tolist(), k=k, filter=filter_dict)
    total = time.time() - start

    return {
        "queries": n_queries,
        "time_s": total,
        "qps": n_queries / total,
        "latency_ms": (total / n_queries) * 1000,
    }


def benchmark_batch_search(db, queries: np.ndarray, k: int = 10) -> dict:
    """Benchmark batch search performance."""
    queries_list = [q.tolist() for q in queries]

    start = time.time()
    results = db.batch_search(queries_list, k=k)
    total = time.time() - start

    return {
        "queries": len(queries),
        "time_s": total,
        "qps": len(queries) / total,
        "latency_ms": (total / len(queries)) * 1000,
    }


def run_benchmark(n_vectors: int, dim: int, n_queries: int = 1000):
    """Run full benchmark suite for given parameters."""
    print(f"\n{'=' * 60}")
    print(f"OmenDB Benchmark: {n_vectors:,} vectors, {dim}D")
    print(f"{'=' * 60}")

    vectors = generate_vectors(n_vectors, dim)
    queries = generate_vectors(n_queries, dim, seed=999)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build
        build = benchmark_build(f"{tmpdir}/db", vectors)
        print(
            f"\nBuild:    {build['vec_per_s']:>10,.0f} vec/s  ({build['time_s']:.2f}s)"
        )

        db = build["db"]

        # Search
        search = benchmark_search(db, queries)
        print(
            f"Search:   {search['qps']:>10,.0f} QPS    ({search['latency_avg_ms']:.2f}ms avg, {search['latency_p99_ms']:.2f}ms p99)"
        )

        # Filtered search (10% selectivity)
        filtered = benchmark_filtered_search(db, queries, {"cat": {"$eq": 5}})
        print(
            f"Filtered: {filtered['qps']:>10,.0f} QPS    ({filtered['latency_ms']:.2f}ms, 10% selectivity)"
        )

        # Batch search
        batch = benchmark_batch_search(db, queries)
        print(
            f"Batch:    {batch['qps']:>10,.0f} QPS    ({batch['latency_ms']:.3f}ms per query)"
        )

    return {"build": build, "search": search, "filtered": filtered, "batch": batch}


def main():
    parser = argparse.ArgumentParser(description="OmenDB Performance Benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--dimension", type=int, default=128, help="Vector dimension")
    parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors")
    args = parser.parse_args()

    print("=" * 60)
    print("OmenDB Performance Benchmark")
    print("=" * 60)

    metadata = get_benchmark_metadata()
    print_metadata(metadata)

    if args.full:
        # Multiple dimensions
        for dim in [128, 384, 768, 1536]:
            run_benchmark(10000, dim)

        # Multiple scales at 128D
        print("\n" + "=" * 60)
        print("Scale Test (128D)")
        print("=" * 60)
        for n in [1000, 10000, 50000]:
            run_benchmark(n, 128)
    else:
        run_benchmark(args.vectors, args.dimension)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
