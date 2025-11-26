# OmenDB

Fast embedded vector database with HNSW + ACORN-1 filtered search.

## Features

- **9.2x faster** than ChromaDB at 10K vectors
- **ACORN-1** filtered search (37.79x speedup)
- **RaBitQ** 2/4/8-bit quantization (8x compression, 100% recall)
- **seerdb** LSM-tree persistent storage

## Installation

```bash
pip install omendb
```

## Quick Start

```python
import omendb

# Open database (creates if doesn't exist)
db = omendb.open("./my_vectors", dimensions=384)

# Add vectors with metadata
db.set([
    {"id": "doc1", "embedding": [0.1] * 384, "metadata": {"title": "Hello"}},
    {"id": "doc2", "embedding": [0.2] * 384, "metadata": {"title": "World"}},
])

# Search
results = db.search([0.15] * 384, k=10)

# Filtered search (ACORN-1)
results = db.search(
    [0.15] * 384,
    k=10,
    filter={"title": {"$eq": "Hello"}}
)
```

## API

```python
db.set(items)              # Store vectors (insert or replace)
db.get(ids)                # Get by ID
db.delete(ids)             # Delete vectors
db.update(ids, metadata)   # Update metadata only
db.search(query, k, filter) # Vector search
len(db) / db.count()       # Count vectors
db.save()                  # Persist to disk
```

## Configuration

```python
db = omendb.open(
    "./my_vectors",
    dimensions=384,
    config={
        "hnsw": {"m": 16, "ef_construction": 200, "ef_search": 100},
        "quantization": {"bits": 4}  # 2, 4, or 8
    }
)
```

## Benchmarks

| Database | QPS (10K) | vs OmenDB |
|----------|-----------|-----------|
| OmenDB   | ~19,000   | 1.0x      |
| ChromaDB | 2,065     | 9.2x slower |
| LanceDB  | 763       | 24.9x slower |

## License

Apache-2.0
