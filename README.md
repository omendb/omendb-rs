# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![License](https://img.shields.io/badge/License-Elastic_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Embedded vector database for Python and Node.js. No server, no setup, just install.

```bash
pip install omendb
```

## Quick Start

```python
import omendb

# Create database (persistent) - creates ./mydb.omen file
db = omendb.open("./mydb", dimensions=128)

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
- **Filtered search** - Query by metadata with JSON-style filters
- **Hybrid search** - Combine vector similarity with BM25 text search
- **Quantization** - 4-8x smaller indexes with minimal recall loss

## Platforms

| Platform                     | Status       |
| ---------------------------- | ------------ |
| Linux (x86_64, ARM64)        | Supported    |
| macOS (Intel, Apple Silicon) | Supported    |
| Windows (x86_64)             | Experimental |

## API

```python
# Database
db = omendb.open(path, dimensions)      # Open or create
db = omendb.open(":memory:", dimensions)  # In-memory (ephemeral)

# CRUD
db.set(items)                           # Insert/update vectors
db.get(id)                              # Get by ID
db.get_batch(ids)                       # Batch get by IDs
db.delete(ids)                          # Delete by IDs
db.delete_by_filter(filter)             # Delete by metadata filter
db.update(id, metadata)                 # Update metadata only

# Iteration
len(db)                                 # Number of vectors
db.count()                              # Same as len(db)
db.count(filter={...})                  # Count matching filter
db.ids()                                # Iterate all IDs (lazy)
db.items()                              # Get all items as list
db.exists(id)                           # Check if ID exists
"id" in db                              # Same as exists()
for item in db: ...                     # Iterate all items (lazy)

# Search
db.search(query, k)                     # Vector search
db.search(query, k, filter={...})       # Filtered search
db.search(query, k, max_distance=0.5)   # Only results with distance <= 0.5
db.search_batch(queries, k)             # Batch search (parallel)

# Hybrid search (requires text field in vectors)
db.search_hybrid(query_vector, query_text, k)
db.search_hybrid(query_vector, query_text, k, alpha=0.7)  # 70% vector, 30% text
db.search_hybrid(query_vector, query_text, k, subscores=True)  # Return separate scores
db.search_text(query_text, k)           # Text-only BM25

# Persistence
db.flush()                              # Flush to disk
```

## Distance Filtering

Use `max_distance` to filter out low-relevance results (prevents "context rot" in RAG):

```python
# Only return results with distance <= 0.5
results = db.search(query, k=10, max_distance=0.5)

# Combine with metadata filter
results = db.search(query, k=10, filter={"type": "doc"}, max_distance=0.5)
```

This ensures your RAG pipeline only receives highly relevant context, avoiding distractors that can hurt LLM performance.

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
    "./mydb",              # Creates ./mydb.omen + ./mydb.wal
    dimensions=384,
    m=16,                # HNSW connections per node (default: 16)
    ef_construction=200, # Index build quality (default: 100)
    ef_search=100,       # Search quality (default: 100)
    quantization=True,   # SQ8 quantization (default: None)
    metric="cosine",     # Distance metric (default: "l2")
)

# Quantization options:
# - True or "sq8": SQ8 ~4x smaller, ~99% recall (recommended)
# - "rabitq": RaBitQ ~8x smaller, ~98% recall
# - None/False: Full precision (default)

# Distance metric options:
# - "l2" or "euclidean": Euclidean distance (default)
# - "cosine": Cosine distance (1 - cosine similarity)
# - "dot" or "ip": Inner product (for MIPS)

# Context manager (auto-flush on exit)
with omendb.open("./db", dimensions=768) as db:
    db.set([...])

# Hybrid search with alpha (0=text, 1=vector, default=0.5)
db.search_hybrid(query_vec, "query text", k=10, alpha=0.7)

# Get separate keyword and semantic scores for debugging/tuning
results = db.search_hybrid(query_vec, "query text", k=10, subscores=True)
# Returns: {"id": "...", "score": 0.85, "keyword_score": 0.92, "semantic_score": 0.78}
```

## Performance

**10K vectors, Apple M3 Max** (m=16, ef=100, k=10):

| Dimension | Single QPS | Batch QPS | Speedup |
| --------- | ---------- | --------- | ------- |
| 128D      | 12,000+    | 87,000+   | 7.2x    |
| 768D      | 3,800+     | 20,500+   | 5.4x    |
| 1536D     | 1,600+     | 6,200+    | 3.8x    |

**SIFT-1M** (1M vectors, 128D, m=16, ef=100, k=10):

| Machine      | QPS   | Recall |
| ------------ | ----- | ------ |
| i9-13900KF   | 4,591 | 98.6%  |
| Apple M3 Max | 3,216 | 98.4%  |

**Quantization** reduces memory with minimal recall loss:

| Mode   | Compression | Use Case                       |
| ------ | ----------- | ------------------------------ |
| f32    | 1x          | Default, highest recall        |
| sq8    | 4x          | Recommended for most users     |
| rabitq | 8x          | Large datasets, cost-sensitive |

```python
db = omendb.open("./db", dimensions=768, quantization=True)  # Enable SQ8
```

<details>
<summary>Benchmark methodology</summary>

- **Parameters**: m=16, ef_construction=100, ef_search=100
- **Batch**: Uses Rayon for parallel search across all cores
- **Recall**: Validated against brute-force ground truth on SIFT/GloVe
- **Reproduce**:
  - Quick (10K): `uv run python benchmarks/run.py`
  - SIFT-1M: `uv run python benchmarks/ann_dataset_test.py --dataset sift-128-euclidean`

</details>

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

[Elastic License 2.0](LICENSE) - Free to use, modify, and embed. The only restriction: you can't offer OmenDB as a managed service to third parties.
