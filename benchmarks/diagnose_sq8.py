#!/usr/bin/env python3
"""Diagnose SQ8 slowdown with detailed timing."""

import time
import numpy as np
import omendb

# Smaller test for faster iteration
N_VECTORS = 1_000
DIM = 128  # Smaller dimension
K = 10
N_ITERS = 500

np.random.seed(42)
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
query = np.random.randn(DIM).astype(np.float32)
query_list = query.tolist()  # Pre-convert to avoid measuring this

items = [{"id": f"v{i}", "vector": vectors[i].tolist()} for i in range(N_VECTORS)]

print(f"Setup: {N_VECTORS} vectors, {DIM}D, k={K}, {N_ITERS} iterations")
print()

# Test 1: FP32
print("Test 1: FP32")
db_fp32 = omendb.open(":memory:", dimensions=DIM)
db_fp32.set(items)

# Warmup
for _ in range(100):
    _ = db_fp32.search(query_list, k=K)

# Measure
start = time.perf_counter()
for _ in range(N_ITERS):
    _ = db_fp32.search(query_list, k=K)
fp32_time = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  FP32: {fp32_time:.0f} us/query")

# Test 2: SQ8
print("\nTest 2: SQ8")
db_sq8 = omendb.open(":memory:", dimensions=DIM, quantization="sq8")
db_sq8.set(items)

# Warmup
for _ in range(100):
    _ = db_sq8.search(query_list, k=K)

# Measure
start = time.perf_counter()
for _ in range(N_ITERS):
    _ = db_sq8.search(query_list, k=K)
sq8_time = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  SQ8: {sq8_time:.0f} us/query")

print(f"\nRatio: SQ8 is {sq8_time / fp32_time:.2f}x FP32")

# Test 3: Measure without result conversion (approximate)
print("\nTest 3: Compare with tight loop (no per-iteration overhead)")
start = time.perf_counter()
for _ in range(N_ITERS):
    pass  # Baseline loop overhead
loop_overhead = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  Empty loop: {loop_overhead:.3f} us/iter")

# Test 4: Profile call overhead
print("\nTest 4: Call overhead comparison")

# Measure len() call overhead (minimal work)
start = time.perf_counter()
for _ in range(N_ITERS):
    _ = len(db_sq8)
len_time = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  len() call: {len_time:.2f} us")

# Measure search with k=1 (minimal results)
start = time.perf_counter()
for _ in range(N_ITERS):
    _ = db_sq8.search(query_list, k=1)
k1_time = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  SQ8 k=1: {k1_time:.0f} us")

start = time.perf_counter()
for _ in range(N_ITERS):
    _ = db_fp32.search(query_list, k=1)
fp32_k1_time = (time.perf_counter() - start) / N_ITERS * 1_000_000
print(f"  FP32 k=1: {fp32_k1_time:.0f} us")
print(f"  k=1 ratio: SQ8 is {k1_time / fp32_k1_time:.2f}x FP32")
