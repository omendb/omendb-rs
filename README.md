# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Embedded vector database for Python. No server, no setup, just `pip install`.

```bash
pip install omendb
```

## Quick Start

```python
import omendb

# Create database (persistent)
db = omendb.open("./vectors", dimensions=128)

# Add vectors with metadata
db.set([
    {"id": "doc1", "vector": [0.1] * 128, "metadata": {"category": "science"}},
    {"id": "doc2", "vector": [0.2] * 128, "metadata": {"category": "history"}},
])

# Search
results = db.search([0.1] * 128, k=5)

# Filtered search
results = db.search([0.1] * 128, k=5, filter={"category": "science"})
```

## Features

- **Embedded** - Runs in-process, no server needed
- **Persistent** - Data survives restarts automatically
- **Filtered search** - Query by metadata using ACORN-1 algorithm
- **Hybrid search** - Combine vector similarity with BM25 text search
- **Compression** - RaBitQ quantization for 4-8x memory reduction

## API

```python
# Database
db = omendb.open(path, dimensions)      # Open or create
db = omendb.open(":memory:", dimensions)  # In-memory (ephemeral)

# CRUD
db.set(items)                           # Insert/update vectors
db.get(id)                              # Get by ID
db.delete(ids)                          # Delete by IDs
db.update(id, metadata)                 # Update metadata only

# Search
db.search(query, k)                     # Vector search
db.search(query, k, filter={...})       # Filtered search
db.search_batch(queries, k)             # Batch search (parallel)

# Hybrid search (requires text field in vectors)
db.search_hybrid(query_vector, query_text, k)
db.search_text(query_text, k)           # Text-only BM25
```

## Filters

```python
# Equality
{"field": "value"}                      # Shorthand
{"field": {"$eq": "value"}}             # Explicit

# Comparison
{"field": {"$ne": "value"}}             # Not equal
{"field": {"$gt": 10}}                  # Greater than
{"field": {"$gte": 10}}                 # Greater or equal
{"field": {"$lt": 10}}                  # Less than
{"field": {"$lte": 10}}                 # Less or equal

# Membership
{"field": {"$in": ["a", "b"]}}          # In list
{"field": {"$contains": "sub"}}         # String contains

# Logical
{"$and": [{...}, {...}]}                # AND
{"$or": [{...}, {...}]}                 # OR
```

## Configuration

```python
db = omendb.open(
    "./vectors",
    dimensions=384,
    m=16,                # HNSW connections per node (default: 16)
    ef_construction=200, # Index build quality (default: 100)
    ef_search=100,       # Search quality (default: 50)
    quantization=4,      # RaBitQ bits: 2, 4, or 8 (default: None)
)
```

## Examples

See [`python/examples/`](python/examples/) for complete working examples:

- `quickstart.py` - Minimal working example
- `basic.py` - CRUD operations and persistence
- `filters.py` - All filter operators
- `rag.py` - RAG workflow with mock embeddings

## Integrations

### LangChain

```bash
pip install omendb[langchain]
```

```python
from langchain_openai import OpenAIEmbeddings
from omendb.langchain import OmenDBVectorStore

store = OmenDBVectorStore.from_texts(
    texts=["Paris is the capital of France"],
    embedding=OpenAIEmbeddings(),
    path="./langchain_vectors",
)
docs = store.similarity_search("capital of France", k=1)
```

### LlamaIndex

```bash
pip install omendb[llamaindex]
```

```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from omendb.llamaindex import OmenDBVectorStore

vector_store = OmenDBVectorStore(path="./llama_vectors")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [Document(text="OmenDB is fast")],
    storage_context=storage_context,
)
response = index.as_query_engine().query("What is OmenDB?")
```

## License

[Apache-2.0](LICENSE)
