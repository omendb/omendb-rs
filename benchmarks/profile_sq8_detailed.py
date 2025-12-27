#!/usr/bin/env python3
"""Detailed profiling of SQ8 search to find slowdown source."""

import time
import numpy as np
import omendb

N_VECTORS = 10_000
DIM = 768
K = 10
N_QUERIES = 100

np.random.seed(42)
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)

items = [{"id": f"v{i}", "vector": vectors[i].tolist()} for i in range(N_VECTORS)]

print("=" * 60)
print(f"Setup: {N_VECTORS} vectors, {DIM}D, k={K}, {N_QUERIES} queries")
print("=" * 60)


def benchmark(db, label, queries, warmup=50):
    """Run benchmark with timing breakdown."""
    # Warmup
    for i in range(warmup):
        _ = db.search(queries[i % len(queries)].tolist(), k=K)

    # Measure individual queries
    times_us = []
    for i in range(N_QUERIES):
        q = queries[i].tolist()
        start = time.perf_counter()
        _ = db.search(q, k=K)
        elapsed = (time.perf_counter() - start) * 1_000_000
        times_us.append(elapsed)

    avg_us = sum(times_us) / len(times_us)
    min_us = min(times_us)
    max_us = max(times_us)
    qps = 1_000_000 / avg_us

    print(f"\n{label}:")
    print(f"  Avg: {avg_us:.0f} us  ({qps:.0f} QPS)")
    print(f"  Min: {min_us:.0f} us  Max: {max_us:.0f} us")

    # Check for cold start
    cold_avg = sum(times_us[:5]) / 5
    warm_avg = sum(times_us[-20:]) / 20
    if cold_avg > warm_avg * 1.5:
        print(
            f"  Cold start detected: first 5 avg {cold_avg:.0f} us, last 20 avg {warm_avg:.0f} us"
        )

    return avg_us


print("\n--- Test 1: FP32 (baseline) ---")
db_fp32 = omendb.open(":memory:", dimensions=DIM)
db_fp32.set(items)
t_fp32 = benchmark(db_fp32, "FP32", queries)

print("\n--- Test 2: SQ8 with rescore (default) ---")
db_sq8 = omendb.open(":memory:", dimensions=DIM, quantization="sq8")
db_sq8.set(items)
t_sq8 = benchmark(db_sq8, "SQ8 (rescore=True)", queries)

print("\n--- Test 3: SQ8 without rescore ---")
db_sq8_nr = omendb.open(":memory:", dimensions=DIM, quantization="sq8", rescore=False)
db_sq8_nr.set(items)
t_sq8_nr = benchmark(db_sq8_nr, "SQ8 (rescore=False)", queries)

print("\n--- Summary ---")
print(f"FP32:            {t_fp32:.0f} us")
print(f"SQ8 (rescore):   {t_sq8:.0f} us  ({t_sq8 / t_fp32:.2f}x FP32)")
print(f"SQ8 (no rescore): {t_sq8_nr:.0f} us  ({t_sq8_nr / t_fp32:.2f}x FP32)")

print("\n--- Test 4: Numpy array vs list input ---")
# Test if input format affects performance
times_list = []
times_numpy = []
for i in range(50):
    q_list = queries[i].tolist()
    q_numpy = queries[i]

    start = time.perf_counter()
    _ = db_sq8.search(q_list, k=K)
    times_list.append((time.perf_counter() - start) * 1_000_000)

    start = time.perf_counter()
    _ = db_sq8.search(q_numpy, k=K)
    times_numpy.append((time.perf_counter() - start) * 1_000_000)

print(f"List input:  {sum(times_list) / len(times_list):.0f} us avg")
print(f"Numpy input: {sum(times_numpy) / len(times_numpy):.0f} us avg")

print("\n--- Test 5: Batch search ---")
batch_queries = queries[:10]
start = time.perf_counter()
for _ in range(10):
    _ = db_sq8.search_batch(batch_queries, k=K)
batch_time = (time.perf_counter() - start) / 10 / len(batch_queries) * 1_000_000
print(f"Batch SQ8: {batch_time:.0f} us per query")
print(f"Single vs Batch speedup: {t_sq8 / batch_time:.2f}x")
