# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Fast embedded vector database for AI applications.

Built in Rust with Python bindings. HNSW indexing, ACORN-1 filtered search, and RaBitQ compression.

## Installation

```bash
pip install omendb
```

## Quick Start

```python
import omendb

# Open database (auto-persists to disk)
db = omendb.open("./my_vectors", dimensions=384)

# Add vectors with metadata
db.set([
    {"id": "doc1", "embedding": [...], "metadata": {"category": "science"}},
    {"id": "doc2", "embedding": [...], "metadata": {"category": "art"}},
])

# Search
results = db.search(query_embedding, k=10)

# Filtered search (ACORN-1)
results = db.search(
    query_embedding,
    k=10,
    filter={"category": {"$eq": "science"}}
)
```

## Features

| Feature | Description |
|---------|-------------|
| **HNSW Index** | ~19,000 QPS at 10K vectors |
| **ACORN-1** | Filtered search, 37x faster than post-filtering |
| **RaBitQ** | 2/4/8-bit quantization (8x compression, 100% recall) |
| **Persistent** | Auto-saves via seerdb LSM-tree |
| **Collections** | Organize vectors into namespaces |

## API Reference

```python
# Database
db = omendb.open(path, dimensions, config={})
db.set(items)                  # Insert/replace vectors
db.get(ids)                    # Get by ID
db.delete(ids)                 # Delete vectors
db.update(ids, metadata)       # Update metadata
db.search(query, k, filter)    # Vector search
db.count()                     # Count vectors

# Collections
coll = db.collection("name")   # Get or create collection
coll.set(items)                # Same API as db
db.collections()               # List collections
db.delete_collection("name")   # Delete collection
```

## Configuration

```python
db = omendb.open(
    "./vectors",
    dimensions=384,
    config={
        "hnsw": {
            "m": 16,                # Connections per node
            "ef_construction": 200, # Build quality
            "ef_search": 100        # Search quality
        },
        "quantization": {
            "bits": 4               # 2, 4, or 8
        }
    }
)
```

## Filter Operators

```python
{"field": {"$eq": value}}      # Equals
{"field": {"$gt": value}}      # Greater than
{"field": {"$gte": value}}     # Greater or equal
{"field": {"$lt": value}}      # Less than
{"field": {"$lte": value}}     # Less or equal
{"$and": [...]}                # Logical AND
{"$or": [...]}                 # Logical OR
```

## Integrations

Works with LangChain and LlamaIndex:

```bash
pip install omendb[langchain]
pip install omendb[llamaindex]
```

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0
