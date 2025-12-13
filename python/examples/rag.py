#!/usr/bin/env python3
"""
OmenDB RAG (Retrieval-Augmented Generation) Example

Demonstrates a basic RAG workflow:
1. Store document chunks with embeddings
2. Retrieve relevant context for a query
3. (In production) Generate answer with LLM

Note: Uses mock embeddings. Replace with real model in production:
- sentence-transformers
- OpenAI embeddings
- Cohere embed
"""

import hashlib

import omendb


# --- Mock Embedding (replace with real model) ---
def embed(text: str, dim: int = 128) -> list[float]:
    """Hash-based mock embedding. Deterministic for same text."""
    h = hashlib.sha256(text.encode()).digest()
    raw = [(h[i % len(h)] / 255.0) * 2 - 1 for i in range(dim)]
    mag = sum(x * x for x in raw) ** 0.5
    return [x / mag for x in raw]


# --- Sample Documents ---
DOCS = [
    {
        "id": "intro",
        "text": "OmenDB is a fast embedded vector database for Python.",
        "source": "readme",
    },
    {
        "id": "install",
        "text": "Install with: pip install omendb. Requires Python 3.9+.",
        "source": "quickstart",
    },
    {
        "id": "search",
        "text": "Use db.search(query, k=10) to find similar vectors.",
        "source": "api",
    },
    {
        "id": "filters",
        "text": "Filter search with MongoDB-style operators: $eq, $gt, $in.",
        "source": "api",
    },
    {
        "id": "persist",
        "text": "Data persists automatically. Use db.flush() to flush.",
        "source": "api",
    },
]


def main():
    # Create database
    db = omendb.open(":memory:", dimensions=128)

    # Index documents
    items = [{"id": doc["id"], "vector": embed(doc["text"]), "metadata": doc} for doc in DOCS]
    db.set(items)
    print(f"Indexed {len(db)} documents\n")

    # Query
    queries = [
        "How do I install OmenDB?",
        "How do I search for vectors?",
        "Does data persist?",
    ]

    for q in queries:
        print(f"Q: {q}")
        results = db.search(query=embed(q), k=2)
        for r in results:
            print(f"  [{r['metadata']['source']}] {r['metadata']['text'][:60]}...")
        print()

    # Filter by source
    print("Filtered search (source=api):")
    results = db.search(query=embed("how to use"), k=3, filter={"source": "api"})
    for r in results:
        print(f"  {r['id']}: {r['metadata']['text'][:50]}...")


if __name__ == "__main__":
    main()
