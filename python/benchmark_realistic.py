#!/usr/bin/env python3
"""Realistic query-heavy benchmark for OmenDB Python bindings

Real-world pattern: Build once, query repeatedly
"""

import omendb
import tempfile
import os
import time
import numpy as np
import cProfile
import pstats
from io import StringIO


def build_index(db, n_vectors, dimensions):
    """One-time index build"""
    print(f"Building index with {n_vectors:,} vectors...")
    vectors = [{
        "id": f"vec{i}",
        "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
        "metadata": {"category": f"cat_{i % 10}", "value": i}
    } for i in range(n_vectors)]

    start = time.time()
    db.set(vectors)
    build_time = time.time() - start

    print(f"  Build time: {build_time:.2f}s ({n_vectors / build_time:,.0f} vec/s)")
    return build_time


def query_workload(db, n_queries, k, dimensions, with_filter=False):
    """Realistic query workload - repeated searches"""
    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(n_queries)]

    filter_dict = {"category": "cat_5"} if with_filter else None
    filter_desc = " with filter" if with_filter else ""

    print(f"\nQuery workload: {n_queries:,} queries (k={k}){filter_desc}")

    # Warmup
    for _ in range(10):
        db.search(queries[0], k=k, filter=filter_dict)

    # Benchmark
    start = time.time()
    for query in queries:
        _ = db.search(query, k=k, filter=filter_dict)
    elapsed = time.time() - start

    qps = n_queries / elapsed
    latency_ms = (elapsed / n_queries) * 1000

    print(f"  QPS: {qps:,.0f}")
    print(f"  Latency: {latency_ms:.3f}ms (mean)")
    print(f"  p50: {latency_ms:.3f}ms (approx)")

    return qps, latency_ms


def profile_query_workload(db, n_queries, k, dimensions):
    """Profile query workload to find bottlenecks"""
    print("\n" + "="*60)
    print("PROFILING QUERY WORKLOAD")
    print("="*60)

    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(n_queries)]

    profiler = cProfile.Profile()
    profiler.enable()

    for query in queries:
        _ = db.search(query, k=k)

    profiler.disable()

    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())


def benchmark_incremental_updates(db, dimensions):
    """Benchmark incremental updates (more realistic than batch rebuild)"""
    print("\n" + "="*60)
    print("INCREMENTAL UPDATE PATTERN")
    print("="*60)

    # Simulate adding 100 vectors at a time (more realistic)
    batches = [100, 500, 1000]

    for batch_size in batches:
        vectors = [{
            "id": f"inc{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {}
        } for i in range(batch_size)]

        start = time.time()
        db.set(vectors)
        elapsed = time.time() - start

        print(f"  Add {batch_size} vectors: {elapsed*1000:.1f}ms ({batch_size/elapsed:,.0f} vec/s)")


def main():
    print("="*60)
    print("REALISTIC QUERY-HEAVY BENCHMARK")
    print("="*60)
    print("\nPattern: Build once, query repeatedly (real-world use)")

    dimensions = 128
    n_vectors = 10_000

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.db")
        db = omendb.open(db_path, dimensions=dimensions)

        # Phase 1: Build index (one-time cost)
        print("\n" + "="*60)
        print("PHASE 1: INDEX BUILD (ONE-TIME)")
        print("="*60)
        build_time = build_index(db, n_vectors, dimensions)

        # Phase 2: Query workload (repeated, main use case)
        print("\n" + "="*60)
        print("PHASE 2: QUERY WORKLOAD (MAIN USE CASE)")
        print("="*60)

        # Test different query patterns
        qps_k10, lat_k10 = query_workload(db, 10_000, k=10, dimensions=dimensions)
        qps_k100, lat_k100 = query_workload(db, 1_000, k=100, dimensions=dimensions)

        # With filtering
        qps_filtered, lat_filtered = query_workload(db, 5_000, k=10, dimensions=dimensions, with_filter=True)

        # Phase 3: Incremental updates (realistic pattern)
        benchmark_incremental_updates(db, dimensions)

        # Phase 4: Profile to find bottlenecks
        profile_query_workload(db, 1000, k=10, dimensions=dimensions)

        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"\nBuild: {n_vectors/build_time:,.0f} vec/s")
        print(f"Query (k=10): {qps_k10:,.0f} QPS @ {lat_k10:.3f}ms")
        print(f"Query (k=100): {qps_k100:,.0f} QPS @ {lat_k100:.3f}ms")
        print(f"Query (filtered): {qps_filtered:,.0f} QPS @ {lat_filtered:.3f}ms")

        print("\n" + "="*60)
        print("BOTTLENECK ANALYSIS")
        print("="*60)
        print("\nLikely bottlenecks:")
        print("  1. Python/Rust boundary overhead (PyO3 conversions)")
        print("  2. JSON metadata serialization/deserialization")
        print("  3. Result vector allocations in Python")
        print("  4. GIL overhead (even with release)")
        print("\nNext steps:")
        print("  → Profile with py-spy or cProfile to identify hotspots")
        print("  → Optimize PyO3 conversions (use Cow, avoid clones)")
        print("  → Consider bulk query API (batch queries)")
        print("  → Benchmark without metadata to isolate JSON overhead")


if __name__ == "__main__":
    main()
