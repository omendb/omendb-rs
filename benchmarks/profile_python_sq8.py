#!/usr/bin/env python3
"""Profile Python SQ8 vs FP32 single-query performance to find hotspot."""

import time
import numpy as np
import omendb

# Parameters
N_VECTORS = 10_000
N_QUERIES = 1000
DIMENSIONS = 768
K = 10


def benchmark_search(db, queries, name: str) -> float:
    """Benchmark single-query search and return QPS."""
    # Warmup
    for q in queries[:10]:
        db.search(q.tolist(), k=K)

    # Timed run
    start = time.perf_counter()
    for q in queries:
        db.search(q.tolist(), k=K)
    elapsed = time.perf_counter() - start

    qps = len(queries) / elapsed
    print(f"{name}: {qps:.0f} QPS ({elapsed * 1000 / len(queries):.2f} ms/query)")
    return qps


def benchmark_batch(db, queries, name: str) -> float:
    """Benchmark batch search and return QPS."""
    # Convert to 2D array
    queries_2d = np.array(queries)

    # Warmup
    db.search_batch(queries_2d[:10], k=K)

    # Timed run
    start = time.perf_counter()
    db.search_batch(queries_2d, k=K)
    elapsed = time.perf_counter() - start

    qps = len(queries) / elapsed
    print(f"{name} batch: {qps:.0f} QPS ({elapsed * 1000 / len(queries):.2f} ms/query)")
    return qps


def benchmark_numpy_vs_list(db, queries, name: str):
    """Compare numpy array vs list input."""
    # List input
    list_queries = [q.tolist() for q in queries[:100]]
    start = time.perf_counter()
    for q in list_queries:
        db.search(q, k=K)
    list_elapsed = time.perf_counter() - start
    list_qps = 100 / list_elapsed

    # Numpy input
    start = time.perf_counter()
    for q in queries[:100]:
        db.search(q, k=K)
    np_elapsed = time.perf_counter() - start
    np_qps = 100 / np_elapsed

    print(f"{name} list input: {list_qps:.0f} QPS, numpy input: {np_qps:.0f} QPS")


def profile_search_components(db, query, name: str):
    """Profile individual components of search."""
    # Test just the search call overhead
    n = 100

    # Cold call
    db.search(query.tolist(), k=K)

    # Time many calls
    start = time.perf_counter()
    for _ in range(n):
        db.search(query.tolist(), k=K)
    elapsed = time.perf_counter() - start

    avg_us = elapsed * 1_000_000 / n
    print(f"{name} avg search time: {avg_us:.0f} us")


def main():
    np.random.seed(42)

    # Generate random data
    vectors = np.random.randn(N_VECTORS, DIMENSIONS).astype(np.float32)
    queries = np.random.randn(N_QUERIES, DIMENSIONS).astype(np.float32)

    # Prepare items
    items = [
        {"id": f"vec_{i}", "vector": vectors[i].tolist()} for i in range(N_VECTORS)
    ]

    print("=== Python SQ8 vs FP32 Profile ===")
    print(f"Vectors: {N_VECTORS}, Dimensions: {DIMENSIONS}, K: {K}")
    print()

    # Test FP32
    print("--- Full Precision (fp32) ---")
    db_fp32 = omendb.open(":memory:", dimensions=DIMENSIONS)
    db_fp32.set(items)

    profile_search_components(db_fp32, queries[0], "fp32")
    benchmark_numpy_vs_list(db_fp32, queries, "fp32")
    fp32_qps = benchmark_search(db_fp32, queries, "fp32 single")
    fp32_batch = benchmark_batch(db_fp32, queries, "fp32")
    print()

    # Test SQ8
    print("--- SQ8 Quantization ---")
    db_sq8 = omendb.open(":memory:", dimensions=DIMENSIONS, quantization="sq8")
    db_sq8.set(items)

    profile_search_components(db_sq8, queries[0], "sq8")
    benchmark_numpy_vs_list(db_sq8, queries, "sq8")
    sq8_qps = benchmark_search(db_sq8, queries, "sq8 single")
    sq8_batch = benchmark_batch(db_sq8, queries, "sq8")
    print()

    # Test SQ8 with rescore=False
    print("--- SQ8 no rescore ---")
    db_sq8_nr = omendb.open(
        ":memory:", dimensions=DIMENSIONS, quantization="sq8", rescore=False
    )
    db_sq8_nr.set(items)

    profile_search_components(db_sq8_nr, queries[0], "sq8-nr")
    sq8_nr_qps = benchmark_search(db_sq8_nr, queries, "sq8-nr single")
    sq8_nr_batch = benchmark_batch(db_sq8_nr, queries, "sq8-nr")
    print()

    # Test RaBitQ
    print("--- RaBitQ Quantization ---")
    db_rabitq = omendb.open(":memory:", dimensions=DIMENSIONS, quantization="rabitq")
    db_rabitq.set(items)

    profile_search_components(db_rabitq, queries[0], "rabitq")
    rabitq_qps = benchmark_search(db_rabitq, queries, "rabitq single")
    rabitq_batch = benchmark_batch(db_rabitq, queries, "rabitq")
    print()

    # Summary
    print("=== Summary ===")
    print(f"{'Mode':<15} {'Single QPS':>12} {'Batch QPS':>12} {'vs FP32':>10}")
    print(f"{'fp32':<15} {fp32_qps:>12.0f} {fp32_batch:>12.0f} {'1.0x':>10}")
    print(
        f"{'sq8':<15} {sq8_qps:>12.0f} {sq8_batch:>12.0f} {sq8_qps / fp32_qps:>10.2f}x"
    )
    print(
        f"{'sq8-nr':<15} {sq8_nr_qps:>12.0f} {sq8_nr_batch:>12.0f} {sq8_nr_qps / fp32_qps:>10.2f}x"
    )
    print(
        f"{'rabitq':<15} {rabitq_qps:>12.0f} {rabitq_batch:>12.0f} {rabitq_qps / fp32_qps:>10.2f}x"
    )


if __name__ == "__main__":
    main()
