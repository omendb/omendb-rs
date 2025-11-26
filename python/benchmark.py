#!/usr/bin/env python3
"""Performance benchmark for OmenDB Python bindings

Target: 20K-30K vec/s @ 10K vectors (from Week 1 design)
"""

import omendb
import tempfile
import os
import time
import numpy as np


def benchmark_insert(db, n_vectors, dimensions):
    """Benchmark insert performance"""
    print(f"\n{'='*60}")
    print(f"INSERT BENCHMARK: {n_vectors:,} vectors ({dimensions} dims)")
    print(f"{'='*60}")

    # Generate random vectors
    vectors = []
    for i in range(n_vectors):
        embedding = np.random.randn(dimensions).astype(np.float32).tolist()
        vectors.append({
            "id": f"vec{i}",
            "embedding": embedding,
            "metadata": {"index": i}
        })

    # Benchmark insert
    start = time.time()
    db.set(vectors)
    elapsed = time.time() - start

    vec_per_sec = n_vectors / elapsed

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {vec_per_sec:,.0f} vec/s")
    print(f"  Latency: {(elapsed / n_vectors) * 1000:.3f}ms per vector")

    return vec_per_sec


def benchmark_search(db, n_queries, k, dimensions):
    """Benchmark search performance"""
    print(f"\n{'='*60}")
    print(f"SEARCH BENCHMARK: {n_queries} queries (k={k})")
    print(f"{'='*60}")

    # Generate random query vectors
    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(n_queries)]

    # Benchmark search
    start = time.time()
    for query in queries:
        results = db.search(query, k=k)
    elapsed = time.time() - start

    qps = n_queries / elapsed
    latency_ms = (elapsed / n_queries) * 1000

    print(f"  Time: {elapsed:.2f}s")
    print(f"  QPS: {qps:,.0f}")
    print(f"  Latency (mean): {latency_ms:.3f}ms")

    return qps, latency_ms


def benchmark_batch_operations(dimensions):
    """Benchmark batch operations"""
    print(f"\n{'='*60}")
    print("BATCH OPERATIONS BENCHMARK")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.db")
        db = omendb.open(db_path, dimensions=dimensions)

        # Test different batch sizes
        batch_sizes = [100, 1000, 10000]

        for batch_size in batch_sizes:
            vectors = [{
                "id": f"vec{i}",
                "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
                "metadata": {}
            } for i in range(batch_size)]

            start = time.time()
            db.set(vectors)
            elapsed = time.time() - start

            print(f"  Batch size {batch_size:,}: {batch_size / elapsed:,.0f} vec/s")


def benchmark_persistence(dimensions, n_vectors):
    """Benchmark save/load operations"""
    print(f"\n{'='*60}")
    print(f"PERSISTENCE BENCHMARK: {n_vectors:,} vectors")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "persist.db")

        # Create and populate database
        db = omendb.open(db_path, dimensions=dimensions)
        vectors = [{
            "id": f"vec{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {"index": i}
        } for i in range(n_vectors)]

        db.set(vectors)

        # Benchmark save
        start = time.time()
        db.save()
        save_time = time.time() - start

        print(f"  Save time: {save_time:.2f}s")
        print(f"  Save throughput: {n_vectors / save_time:,.0f} vec/s")

        del db

        # Benchmark load
        start = time.time()
        db2 = omendb.open(db_path, dimensions=dimensions)
        load_time = time.time() - start

        print(f"  Load time: {load_time:.2f}s")
        print(f"  Load throughput: {n_vectors / load_time:,.0f} vec/s")
        print(f"  Speedup vs rebuild: {load_time:.2f}s load vs building HNSW from scratch")


def main():
    print("\n" + "="*60)
    print("OADB PYTHON BINDINGS PERFORMANCE BENCHMARK")
    print("="*60)
    print(f"Target: 20K-30K vec/s @ 10K vectors (from Week 1 design)")
    print()

    dimensions = 128

    # Test 1: Insert performance @ 10K vectors (target scale)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.db")
        db = omendb.open(db_path, dimensions=dimensions)

        insert_throughput = benchmark_insert(db, 10_000, dimensions)

        # Verify target met
        if insert_throughput >= 20_000:
            print(f"\n  ✅ PASS: {insert_throughput:,.0f} vec/s >= 20K target")
        else:
            print(f"\n  ⚠️  BELOW TARGET: {insert_throughput:,.0f} vec/s < 20K target")

        # Test 2: Search performance
        benchmark_search(db, 1000, k=10, dimensions=dimensions)

    # Test 3: Larger scale (100K vectors)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench_large.db")
        db = omendb.open(db_path, dimensions=dimensions)

        print(f"\n{'='*60}")
        print("LARGER SCALE: 100K vectors")
        print(f"{'='*60}")

        throughput_100k = benchmark_insert(db, 100_000, dimensions)
        benchmark_search(db, 100, k=10, dimensions=dimensions)

    # Test 4: Batch operations
    benchmark_batch_operations(dimensions)

    # Test 5: Persistence
    benchmark_persistence(dimensions, 10_000)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Insert @ 10K: {insert_throughput:,.0f} vec/s")
    print(f"  Insert @ 100K: {throughput_100k:,.0f} vec/s")
    print(f"  Target (20K-30K vec/s): {'✅ PASSED' if insert_throughput >= 20_000 else '⚠️ BELOW TARGET'}")
    print()


if __name__ == "__main__":
    main()
