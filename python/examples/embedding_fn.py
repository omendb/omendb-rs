#!/usr/bin/env python3
"""
OmenDB embedding_fn example — auto-embed documents and queries.

Demonstrates:
- Setting up a database with an embedding function
- Adding documents (auto-embedded)
- Searching with text queries (auto-embedded)
- Filtered search with text queries
"""

import omendb


def fake_embedder(texts: list[str]) -> list[list[float]]:
    """Mock embedding function. Replace with your model (OpenAI, sentence-transformers, etc.)."""
    return [[float(ord(c) % 10) / 10.0 for c in text[:4].ljust(4)] for text in texts]


# Create with embedding function
db = omendb.create(":memory:", {"dense": {"dim": 4}}, embedding_fn=fake_embedder)

# Add documents — vectors computed automatically
db.set(
    [
        {
            "id": "doc1",
            "document": "Paris is the capital of France",
            "metadata": {"topic": "geography"},
        },
        {
            "id": "doc2",
            "document": "The mitochondria is the powerhouse of the cell",
            "metadata": {"topic": "biology"},
        },
        {
            "id": "doc3",
            "document": "Python was created by Guido van Rossum",
            "metadata": {"topic": "programming"},
        },
    ]
)

print(f"Added {len(db)} documents")

# Search with text — query auto-embedded
results = db.search("capital city", k=2)
for r in results:
    print(f"  {r['id']}: distance={r['distance']:.4f}")

# Filtered search with text query
results = db.search("science", k=2, filter={"topic": "biology"})
print(f"\nFiltered results: {len(results)}")
