# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Embedded vector database. No server, no setup, just `pip install`.

## Quick Start

```python
import omendb
import numpy as np

# Create database
db = omendb.open("./vectors", dimensions=4)

# Add vectors
db.set([
    {"id": "a", "embedding": [1.0, 0.0, 0.0, 0.0], "metadata": {"type": "dog"}},
    {"id": "b", "embedding": [0.9, 0.1, 0.0, 0.0], "metadata": {"type": "dog"}},
    {"id": "c", "embedding": [0.0, 0.0, 1.0, 0.0], "metadata": {"type": "cat"}},
])

# Search
results = db.search([1.0, 0.0, 0.0, 0.0], k=2)
# [{"id": "a", "distance": 0.0, "metadata": {"type": "dog"}},
#  {"id": "b", "distance": 0.02, "metadata": {"type": "dog"}}]

# Filtered search
results = db.search([1.0, 0.0, 0.0, 0.0], k=2, filter={"type": {"$eq": "cat"}})
# [{"id": "c", "distance": 2.0, "metadata": {"type": "cat"}}]
```

## Why OmenDB?

- **No server** - Runs in your process, data stays local
- **Fast** - 50,000 QPS batch, 5,700 QPS single (10K vectors, 128D)
- **Memory efficient** - 8x compression, same accuracy
- **Filtered search** - Query by metadata without scanning everything
- **Persistent** - Survives restarts, no manual saves needed

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

db.set(items)                  # Add/replace vectors
db.get(ids)                    # Get by ID
db.delete(ids)                 # Delete
db.update(ids, metadata)       # Update metadata only
db.search(query, k, filter)    # Search single query
db.search_batch(queries, k)    # Search batch (parallel)
db.count()                     # Count vectors

# Collections (namespaces)
users = db.collection("users")
users.set([...])
users.search([...], k=5)
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
