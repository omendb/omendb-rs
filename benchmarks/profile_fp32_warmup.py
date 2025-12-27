#!/usr/bin/env python3
"""Profile FP32 with extended warmup to compare with SQ8."""

import time
import numpy as np
import omendb

N_VECTORS = 10_000
DIM = 768
K = 10

np.random.seed(42)
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
query = np.random.randn(DIM).astype(np.float32)

items = [{"id": f"v{i}", "vector": vectors[i].tolist()} for i in range(N_VECTORS)]

print("=== FP32 Extended Warmup Test ===")
db = omendb.open(":memory:", dimensions=DIM)
db.set(items)

# First 10 calls
print("\nFirst 10 calls (cold):")
times = []
for i in range(10):
    start = time.perf_counter()
    db.search(query.tolist(), k=K)
    elapsed = (time.perf_counter() - start) * 1_000_000
    times.append(elapsed)
    print(f"  Call {i + 1}: {elapsed:.0f} us")
print(f"  Avg: {sum(times) / len(times):.0f} us")

# After 100 warmup calls
print("\nAfter 100 warmup calls:")
for _ in range(100):
    db.search(query.tolist(), k=K)

times = []
for i in range(10):
    start = time.perf_counter()
    db.search(query.tolist(), k=K)
    elapsed = (time.perf_counter() - start) * 1_000_000
    times.append(elapsed)
print(
    f"  Avg: {sum(times) / len(times):.0f} us, range: {min(times):.0f}-{max(times):.0f} us"
)

# Tight loop
print("\nTight loop (5000 calls, measuring total):")
n = 5000
start = time.perf_counter()
for _ in range(n):
    db.search(query.tolist(), k=K)
elapsed = time.perf_counter() - start
avg_us = elapsed * 1_000_000 / n
qps = n / elapsed
print(f"  Avg: {avg_us:.0f} us ({qps:.0f} QPS)")
