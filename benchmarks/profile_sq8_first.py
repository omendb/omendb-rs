#!/usr/bin/env python3
"""Profile Python SQ8 timing breakdown - SQ8 first to capture timing output."""

import numpy as np
import omendb

N_VECTORS = 10_000
N_QUERIES = 100
DIM = 768
K = 10

np.random.seed(42)
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)

items = [{"id": f"v{i}", "vector": vectors[i].tolist()} for i in range(N_VECTORS)]

print("=== Testing SQ8 (will show timing for calls 50-54) ===")
db_sq8 = omendb.open(":memory:", dimensions=DIM, quantization="sq8")
db_sq8.set(items)

for i, q in enumerate(queries):
    db_sq8.search(q.tolist(), k=K)
    if i == 60:
        break

print("\n=== Now testing FP32 (calls will be 61+, no timing output) ===")
db_fp32 = omendb.open(":memory:", dimensions=DIM)
db_fp32.set(items)

for i, q in enumerate(queries):
    db_fp32.search(q.tolist(), k=K)
