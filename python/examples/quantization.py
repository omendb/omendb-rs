#!/usr/bin/env python3
"""
OmenDB quantization example — SQ8 and RaBitQ compression.

Demonstrates:
- SQ8 quantization (4x compression, ~99% recall)
- RaBitQ quantization (32x compression)
- Comparing results between modes
"""

import random

import omendb


def random_vectors(n, dim):
    """Generate random vectors for demonstration."""
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


dim = 64
vectors = random_vectors(100, dim)

# Full precision (default)
db_full = omendb.open(":memory:", dimensions=dim)
db_full.set([{"id": f"v{i}", "vector": v} for i, v in enumerate(vectors)])

# SQ8 quantization — 4x smaller, ~99% recall
db_sq8 = omendb.open(":memory:", dimensions=dim, quantization="sq8")
db_sq8.set([{"id": f"v{i}", "vector": v} for i, v in enumerate(vectors)])

# RaBitQ quantization — 32x smaller (needs 256+ vectors for training, falls back to exact for fewer)
db_rabitq = omendb.open(":memory:", dimensions=dim, quantization="rabitq")
db_rabitq.set([{"id": f"v{i}", "vector": v} for i, v in enumerate(vectors)])

# Compare search results
query = vectors[0]
k = 5

print("Full precision results:")
for r in db_full.search(query, k=k):
    print(f"  {r['id']}: {r['distance']:.6f}")

print("\nSQ8 results:")
for r in db_sq8.search(query, k=k):
    print(f"  {r['id']}: {r['distance']:.6f}")

print("\nRaBitQ results:")
for r in db_rabitq.search(query, k=k):
    print(f"  {r['id']}: {r['distance']:.6f}")
