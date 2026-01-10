#!/usr/bin/env python3
"""OmenDB Performance Benchmark

Measures build throughput, search QPS, and filtered search performance
at multiple embedding dimensions and dataset sizes.

Usage:
    python benchmark.py              # Quick benchmark (10K vectors)
    python benchmark.py --full       # Full benchmark (10K, 50K, 100K)
    python benchmark.py --dimension 1536  # Specific dimension
    python benchmark.py --output results.json  # Save to JSON
"""

import argparse
import json
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

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


def generate_text_corpus(n: int, seed: int = 42) -> list[str]:
    """Generate random text documents for hybrid search benchmarks."""
    words = [
        "database",
        "vector",
        "search",
        "query",
        "index",
        "storage",
        "memory",
        "performance",
        "fast",
        "efficient",
        "scalable",
        "distributed",
        "embedded",
        "machine",
        "learning",
        "neural",
        "network",
        "model",
        "training",
        "inference",
        "algorithm",
        "optimization",
        "parallel",
        "concurrent",
        "async",
        "thread",
        "rust",
        "python",
        "javascript",
        "typescript",
        "server",
        "client",
    ]
    np.random.seed(seed)
    texts = []
    for _ in range(n):
        n_words = np.random.randint(5, 15)
        doc = " ".join(np.random.choice(words, n_words))
        texts.append(doc)
    return texts


def benchmark_build(
    db_path: str,
    vectors: np.ndarray,
    with_metadata: bool = True,
    quantize_bits: int = 0,
) -> dict:
    """Benchmark index build throughput."""
    n, dim = vectors.shape
    if quantize_bits > 0:
        db = omendb.open(db_path, dimensions=dim, quantization=quantize_bits)
    else:
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
    db.search_batch(queries_list, k=k)
    total = time.time() - start

    return {
        "queries": len(queries),
        "time_s": total,
        "qps": len(queries) / total,
        "latency_ms": (total / len(queries)) * 1000,
    }


def benchmark_text_search(db, query_texts: list[str], k: int = 10, warmup: int = 5) -> dict:
    """Benchmark text-only (BM25) search performance."""
    n_queries = len(query_texts)

    # Warmup
    for q in query_texts[:warmup]:
        db.search_text(q, k=k)

    # Benchmark
    latencies = []
    start = time.time()
    for q in query_texts:
        t0 = time.time()
        db.search_text(q, k=k)
        latencies.append((time.time() - t0) * 1000)
    total = time.time() - start

    latencies.sort()
    return {
        "queries": n_queries,
        "time_s": total,
        "qps": n_queries / total,
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
    }


def benchmark_hybrid_search(
    db,
    query_vectors: np.ndarray,
    query_texts: list[str],
    k: int = 10,
    alpha: float | None = None,
    warmup: int = 5,
) -> dict:
    """Benchmark hybrid (vector + text) search performance."""
    n_queries = len(query_vectors)

    # Warmup
    for i in range(min(warmup, n_queries)):
        db.search_hybrid(
            query_vectors[i].tolist(), query_texts[i % len(query_texts)], k=k, alpha=alpha
        )

    # Benchmark
    latencies = []
    start = time.time()
    for i in range(n_queries):
        t0 = time.time()
        db.search_hybrid(
            query_vectors[i].tolist(), query_texts[i % len(query_texts)], k=k, alpha=alpha
        )
        latencies.append((time.time() - t0) * 1000)
    total = time.time() - start

    latencies.sort()
    return {
        "queries": n_queries,
        "time_s": total,
        "qps": n_queries / total,
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_p99_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
    }


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute ground truth neighbors using brute-force L2 search."""
    n_queries = len(queries)
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)

    for i, q in enumerate(queries):
        # L2 distance to all vectors
        distances = np.sum((vectors - q) ** 2, axis=1)
        # Get k nearest indices
        ground_truth[i] = np.argpartition(distances, k)[:k]

    return ground_truth


def benchmark_recall(db, vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> dict:
    """Measure recall@k against brute-force ground truth."""
    n_queries = min(100, len(queries))  # Limit for speed
    queries_subset = queries[:n_queries]
    ground_truth = compute_ground_truth(vectors, queries_subset, k)

    total_recall = 0.0
    for i, q in enumerate(queries_subset):
        results = db.search(q.tolist(), k=k)
        returned_ids = {int(r["id"][1:]) for r in results}  # "d123" -> 123
        true_ids = set(ground_truth[i])
        recall = len(returned_ids & true_ids) / k
        total_recall += recall

    avg_recall = total_recall / n_queries
    return {"recall_at_k": avg_recall, "k": k, "n_queries": n_queries}


def run_benchmark(n_vectors: int, dim: int, n_queries: int = 1000, quantize_bits: int = 0):
    """Run full benchmark suite for given parameters."""
    mode = f"RaBitQ-{quantize_bits}bit" if quantize_bits > 0 else "f32"
    print(f"\n{'=' * 60}")
    print(f"OmenDB Benchmark: {n_vectors:,} vectors, {dim}D ({mode})")
    print(f"{'=' * 60}")

    vectors = generate_vectors(n_vectors, dim)
    queries = generate_vectors(n_queries, dim, seed=999)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build
        build = benchmark_build(f"{tmpdir}/db", vectors, quantize_bits=quantize_bits)
        print(f"\nBuild:    {build['vec_per_s']:>10,.0f} vec/s  ({build['time_s']:.2f}s)")

        db = build["db"]

        # Search
        search = benchmark_search(db, queries)
        print(
            f"Search:   {search['qps']:>10,.0f} QPS    ({search['latency_avg_ms']:.2f}ms avg, {search['latency_p99_ms']:.2f}ms p99)"
        )

        # Recall measurement (graph quality indicator)
        recall = benchmark_recall(db, vectors, queries)
        print(f"Recall:   {recall['recall_at_k']:>10.1%} @{recall['k']}")

        # Filtered search (10% selectivity)
        filtered = benchmark_filtered_search(db, queries, {"cat": 5})
        print(
            f"Filtered: {filtered['qps']:>10,.0f} QPS    ({filtered['latency_ms']:.2f}ms, 10% selectivity)"
        )

        # Batch search
        batch = benchmark_batch_search(db, queries)
        print(f"Batch:    {batch['qps']:>10,.0f} QPS    ({batch['latency_ms']:.3f}ms per query)")

    # Return serializable results (no db object)
    return {
        "config": {
            "n_vectors": n_vectors,
            "dimensions": dim,
            "n_queries": n_queries,
            "quantize_bits": quantize_bits,
        },
        "build": {k: v for k, v in build.items() if k != "db"},
        "search": search,
        "recall": recall,
        "filtered": filtered,
        "batch": batch,
    }


def run_hybrid_benchmark(n_vectors: int, dim: int, n_queries: int = 100):
    """Run hybrid search benchmark suite."""
    print(f"\n{'=' * 60}")
    print(f"OmenDB Hybrid Benchmark: {n_vectors:,} vectors, {dim}D")
    print(f"{'=' * 60}")

    vectors = generate_vectors(n_vectors, dim)
    texts = generate_text_corpus(n_vectors, seed=42)
    query_vectors = generate_vectors(n_queries, dim, seed=999)
    query_texts = ["vector database", "machine learning", "rust performance", "search query"]

    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(f"{tmpdir}/db", dimensions=dim)
        db.enable_text_search()

        # Build with text
        batch = [
            {
                "id": f"d{i}",
                "vector": vectors[i].tolist(),
                "text": texts[i],
                "metadata": {"cat": i % 10},
            }
            for i in range(n_vectors)
        ]
        start = time.time()
        db.set(batch)
        build_time = time.time() - start
        print(f"\nBuild:    {n_vectors / build_time:>10,.0f} vec/s  ({build_time:.2f}s)")

        # Text search (BM25 only)
        text_result = benchmark_text_search(db, query_texts * (n_queries // 4))
        print(
            f"Text:     {text_result['qps']:>10,.0f} QPS    ({text_result['latency_avg_ms']:.2f}ms avg)"
        )

        # Hybrid search (balanced alpha=0.5)
        hybrid_result = benchmark_hybrid_search(db, query_vectors, query_texts, alpha=0.5)
        print(
            f"Hybrid:   {hybrid_result['qps']:>10,.0f} QPS    ({hybrid_result['latency_avg_ms']:.2f}ms avg)"
        )

        # Hybrid text-only (alpha=0.0)
        text_only = benchmark_hybrid_search(db, query_vectors, query_texts, alpha=0.0)
        print(
            f"α=0.0:    {text_only['qps']:>10,.0f} QPS    ({text_only['latency_avg_ms']:.2f}ms avg)"
        )

        # Hybrid vector-only (alpha=1.0)
        vec_only = benchmark_hybrid_search(db, query_vectors, query_texts, alpha=1.0)
        print(
            f"α=1.0:    {vec_only['qps']:>10,.0f} QPS    ({vec_only['latency_avg_ms']:.2f}ms avg)"
        )

    return {
        "config": {"n_vectors": n_vectors, "dimensions": dim, "n_queries": n_queries},
        "build_time_s": build_time,
        "text_search": text_result,
        "hybrid_balanced": hybrid_result,
        "hybrid_text_only": text_only,
        "hybrid_vector_only": vec_only,
    }


def save_results(output_path: str, metadata: dict, results: list):
    """Save benchmark results to JSON file."""
    output = {"metadata": metadata, "results": results}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {path}")


def append_to_history(metadata: dict, results: list):
    """Append results to benchmarks/history.json for tracking over time."""
    history_path = Path(__file__).parent / "benchmarks" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing history or create new
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    # Append new entry
    entry = {
        "metadata": metadata,
        "results": results,
    }
    history.append(entry)

    # Keep last 100 entries to avoid unbounded growth
    history = history[-100:]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"History updated: {history_path} ({len(history)} entries)")


def main():
    parser = argparse.ArgumentParser(description="OmenDB Performance Benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid search benchmarks")
    parser.add_argument("--dimension", type=int, default=128, help="Vector dimension")
    parser.add_argument("--vectors", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--queries", type=int, default=1000, help="Number of queries")
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[0, 2, 4, 8],
        default=0,
        help="RaBitQ quantization bits (0=none, 2/4/8=quantized)",
    )
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument("--no-history", action="store_true", help="Don't append to history.json")
    args = parser.parse_args()

    print("=" * 60)
    print("OmenDB Performance Benchmark")
    print("=" * 60)

    metadata = get_benchmark_metadata()
    print_metadata(metadata)

    all_results = []

    if args.hybrid:
        # Hybrid search benchmarks
        result = run_hybrid_benchmark(args.vectors, args.dimension)
        all_results.append(result)
    elif args.full:
        # Multiple dimensions
        for dim in [128, 384, 768, 1536]:
            result = run_benchmark(10000, dim)
            all_results.append(result)

        # Multiple scales at 768D
        print("\n" + "=" * 60)
        print("Scale Test (768D)")
        print("=" * 60)
        for n in [10000, 50000, 100000]:
            result = run_benchmark(n, 768)
            all_results.append(result)

        # Hybrid search at 384D (common embedding dim)
        print("\n" + "=" * 60)
        print("Hybrid Search Test")
        print("=" * 60)
        result = run_hybrid_benchmark(10000, 384)
        all_results.append(result)
    else:
        result = run_benchmark(
            args.vectors, args.dimension, n_queries=args.queries, quantize_bits=args.quantize
        )
        all_results.append(result)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)

    # Save to JSON if output specified
    if args.output:
        save_results(args.output, metadata, all_results)

    # Append to history unless disabled
    if not args.no_history:
        append_to_history(metadata, all_results)


if __name__ == "__main__":
    main()
