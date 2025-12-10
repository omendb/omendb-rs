# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Embedded vector database for Python. No server, no setup, just `pip install`.

> **v0.0.9**: Performance optimizations and stability. API may still evolve. [Report issues](https://github.com/omendb/omendb/issues).

## Quick Start

```python
import omendb

# Create database
db = omendb.open("./vectors", dimensions=4)

# Add vectors
db.set([
    {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"type": "dog"}},
    {"id": "b", "vector": [0.9, 0.1, 0.0, 0.0], "metadata": {"type": "dog"}},
    {"id": "c", "vector": [0.0, 0.0, 1.0, 0.0], "metadata": {"type": "cat"}},
])

# Search
results = db.search([1.0, 0.0, 0.0, 0.0], k=2)
# [{"id": "a", "distance": 0.0, "metadata": {"type": "dog"}},
#  {"id": "b", "distance": 0.02, "metadata": {"type": "dog"}}]

# Filtered search
results = db.search([1.0, 0.0, 0.0, 0.0], k=2, filter={"type": {"$eq": "cat"}})
# [{"id": "c", "distance": 2.0, "metadata": {"type": "cat"}}]
```

## Features

- **Embedded** - Runs in your process, no server needed
- **Persistent** - Data survives restarts, no manual saves
- **Filtered search** - Query by metadata using ACORN-1 algorithm
- **Compression** - RaBitQ quantization reduces memory 4-8x
- **HNSW indexing** - Approximate nearest neighbor search

### Performance

On a 10K vector dataset (M3 Max):

| Dimension | Batch Search | Single Search |
| --------- | ------------ | ------------- |
| 128D      | ~50,000 QPS  | ~7,700 QPS    |
| 768D      | ~12,600 QPS  | ~2,500 QPS    |
| 1536D     | ~4,700 QPS   | ~1,200 QPS    |

Run `python benchmark.py` to measure on your hardware.

## Hybrid Search

Combine vector similarity with BM25 text search using RRF fusion:

```python
db = omendb.open("./docs", dimensions=384)

# Add vectors with text for hybrid search
db.set([
    {"id": "doc1", "vector": embedding1, "text": "Machine learning basics", "metadata": {"topic": "ml"}},
    {"id": "doc2", "vector": embedding2, "text": "Deep learning neural networks", "metadata": {"topic": "dl"}},
])

# Hybrid search (vector + text)
results = db.search_hybrid(
    query_vector=query_embedding,
    query_text="neural networks",
    k=10,
    alpha=0.5,  # 0=text only, 1=vector only
)

# Text-only search (BM25)
results = db.search_text("machine learning", k=10)
```

## With LangChain

```python
from langchain_openai import OpenAIEmbeddings
from omendb.langchain import OmenDBVectorStore

# Create from documents
store = OmenDBVectorStore.from_texts(
    texts=["Paris is the capital of France", "Berlin is in Germany"],
    embedding=OpenAIEmbeddings(),
    path="./langchain_vectors",
)

# Search
docs = store.similarity_search("capital of France", k=1)
print(docs[0].page_content)  # "Paris is the capital of France"
```

```bash
pip install omendb[langchain]
```

## With LlamaIndex

```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from omendb.llamaindex import OmenDBVectorStore

# Create index
vector_store = OmenDBVectorStore(path="./llama_vectors")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [Document(text="OmenDB is fast")],
    storage_context=storage_context,
)

# Query
response = index.as_query_engine().query("What is OmenDB?")
```

```bash
pip install omendb[llamaindex]
```

## API

```python
db = omendb.open(path, dimensions)

db.set(items)                  # Add/replace vectors (with optional text)
db.get(ids)                    # Get by ID
db.delete(ids)                 # Delete
db.update(ids, metadata)       # Update metadata only
db.search(query, k, filter)    # Vector search
db.search_batch(queries, k)    # Batch vector search (parallel)
db.search_hybrid(vec, text, k) # Hybrid vector + text search
db.search_text(text, k)        # Text-only BM25 search
db.count()                     # Count vectors
```

## Filters

```python
{"field": {"$eq": "value"}}    # equals
{"field": {"$gt": 10}}         # greater than
{"field": {"$gte": 10}}        # greater or equal
{"field": {"$lt": 10}}         # less than
{"field": {"$lte": 10}}        # less or equal
{"$and": [{...}, {...}]}       # AND
{"$or": [{...}, {...}]}        # OR
```

## Configuration

```python
db = omendb.open(
    "./vectors",
    dimensions=384,
    config={
        "hnsw": {"m": 16, "ef_construction": 200, "ef_search": 100},
        "quantization": {"bits": 4}  # 2, 4, or 8
    }
)
```

## License

Apache-2.0
