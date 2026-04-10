#!/usr/bin/env python3
"""
OmenDB quantization example — SQ8 compression.

Demonstrates:
- SQ8 quantization (4x compression, ~99% recall)
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
db_full = omendb.create(":memory:", {"dense": {"dim": dim}})
db_full.set([{"id": f"v{i}", "vector": v} for i, v in enumerate(vectors)])

# SQ8 quantization — 4x smaller, ~99% recall
db_sq8 = omendb.create(":memory:", {"dense": {"dim": dim, "quantization": "sq8"}})
db_sq8.set([{"id": f"v{i}", "vector": v} for i, v in enumerate(vectors)])

# Compare search results
query = vectors[0]
k = 5

print("Full precision results:")
for r in db_full.search(query, k=k):
    print(f"  {r['id']}: {r['distance']:.6f}")

print("\nSQ8 results:")
for r in db_sq8.search(query, k=k):
    print(f"  {r['id']}: {r['distance']:.6f}")
