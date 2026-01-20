#!/usr/bin/env python3
"""
OmenDB Quickstart - Minimal working example.

This is the simplest possible example to get you started.
For more features, see basic.py, filters.py, and rag.py.
"""

import omendb

# Open or create database (128-dimensional vectors)
db = omendb.open(":memory:", dimensions=128)

# Add some vectors with metadata
db.set(
    [
        {"id": "doc1", "vector": [0.1] * 128, "metadata": {"title": "First doc"}},
        {"id": "doc2", "vector": [0.2] * 128, "metadata": {"title": "Second doc"}},
        {"id": "doc3", "vector": [0.15] * 128, "metadata": {"title": "Third doc"}},
    ]
)

# Search for similar vectors
results = db.search(query=[0.1] * 128, k=2)

for r in results:
    print(f"{r['id']}: {r['metadata']['title']} (distance: {r['distance']:.4f})")

# Output:
# doc1: First doc (distance: 0.0000)
# doc3: Third doc (distance: 0.5657)
