# OmenDB

[![PyPI](https://img.shields.io/pypi/v/omendb)](https://pypi.org/project/omendb/)
[![npm](https://img.shields.io/npm/v/omendb)](https://www.npmjs.com/package/omendb)
[![License](https://img.shields.io/badge/License-Elastic_2.0-blue.svg)](https://github.com/omendb/omendb/blob/main/LICENSE)

Embedded vector database for Python and Node.js. No server, no setup, just install.

- **21K QPS** single-threaded search with 100% recall (SIFT-10K)
- **67K vec/s** insert throughput (1.4x hnswlib)
- **SQ8** (4x) and **RaBitQ** (32x) quantization
- **ACORN-1** predicate-aware filtered search
- **Hybrid search** -- BM25 text + vector with RRF fusion
- **Multi-vector** -- ColBERT/MaxSim with MUVERA and token pooling
- **Auto-embedding** -- pass a function, store documents, search with strings

```bash
pip install omendb       # Python
npm install omendb       # Node.js
cargo install omendb     # CLI
```

## Quick Start

### Python

**With auto-embedding** -- pass an embedding function, work with documents and strings:

```python
import omendb

def embed(texts):
    # Your embedding model here (OpenAI, sentence-transformers, etc.)
    return [[0.1] * 384 for _ in texts]

db = omendb.open("./mydb", dimensions=384, embedding_fn=embed)

# Add documents -- auto-embedded
db.set([
    {"id": "doc1", "document": "Paris is the capital of France", "metadata": {"topic": "geography"}},
    {"id": "doc2", "document": "The mitochondria is the powerhouse of the cell", "metadata": {"topic": "biology"}},
])

# Search with text -- auto-embedded
results = db.search("capital of France", k=5)
```

**With vectors** -- bring your own embeddings:

```python
db = omendb.open("./mydb", dimensions=128)

db.set([
    {"id": "doc1", "vector": [0.1] * 128, "metadata": {"category": "science"}},
    {"id": "doc2", "vector": [0.2] * 128, "metadata": {"category": "history"}},
])

results = db.search([0.1] * 128, k=5)
results = db.search([0.1] * 128, k=5, filter={"category": "science"})
```

### Node.js

**With auto-embedding:**

```javascript
const omendb = require("omendb");

const db = omendb.open("./mydb", 384, { embeddingFn: embed });
db.set([{ id: "doc1", document: "Paris is the capital of France" }]);
const results = db.search("capital of France", 5);
```

**With vectors:**

```javascript
const db = omendb.open("./mydb", 128);
db.set([{ id: "doc1", vector: new Float32Array(128).fill(0.1) }]);
const results = db.search(new Float32Array(128).fill(0.1), 5);
```

## Features

- **HNSW graph indexing** -- SIMD-accelerated distance computation
- **ACORN-1 filtered search** -- predicate-aware graph traversal, 37.79x speedup over post-filtering
- **SQ8 quantization** -- 4x compression, ~99% recall
- **RaBitQ quantization** -- 32x compression (1-bit with FFHT rotation), rescore restores exact recall
- **BM25 text search** -- full-text search via Tantivy
- **Hybrid search** -- RRF fusion of vector + text results
- **Multi-vector / ColBERT** -- MUVERA + MaxSim scoring for token-level retrieval
- **Token pooling** -- k-means clustering, 50% storage reduction for multi-vector
- **Auto-embedding** -- `embedding_fn` (Python) / `embeddingFn` (Node.js) for document-in, text-query workflows
- **Collections** -- namespaced sub-databases within a single file
- **Persistence** -- WAL + atomic checkpoints
- **O(1) lazy delete + compaction** -- deleted records cleaned up in background
- **Segment-based architecture** -- background merging for sustained write throughput
- **Context manager** (Python) / `close()` (Node.js) for resource cleanup
- **CLI tool** -- info, stats, search, bench, export, compact, and more

## Platforms

| Platform                     | Status       |
| ---------------------------- | ------------ |
| Linux (x86_64, ARM64)        | Supported    |
| macOS (Intel, Apple Silicon) | Supported    |
| Windows (x86_64)             | Experimental |

## API Reference

### Python

```python
# Database
db = omendb.open(path, dimensions, embedding_fn=fn)  # With auto-embedding
db = omendb.open(path, dimensions)                    # Manual vectors
db = omendb.open(":memory:", dimensions)              # In-memory

# CRUD
db.set(items)                           # Insert/update (vectors or documents)
db.set("id", vector, metadata)          # Single insert
db.get(id)                              # Get by ID
db.get_batch(ids)                       # Batch get
db.delete(ids)                          # Delete by IDs
db.delete_by_filter(filter)             # Delete by metadata filter
db.update(id, vector, metadata, text)   # Update fields

# Search
db.search(query, k)                     # Vector or string query
db.search(query, k, filter={...})       # Filtered search (ACORN-1)
db.search(query, k, max_distance=0.5)   # Distance threshold
db.search_batch(queries, k)             # Batch search (parallel)

# Hybrid search
db.search_hybrid(query_vector, query_text, k)
db.search_hybrid("query text", k=10)    # String query (auto-embeds both)
db.search_text(query_text, k)           # Text-only BM25

# Iteration
len(db)                                 # Count
db.count(filter={...})                  # Filtered count
db.ids()                                # Lazy ID iterator
db.items()                              # All items (loads to memory)
for item in db: ...                     # Lazy iteration
"id" in db                              # Existence check

# Collections
col = db.collection("users")            # Create/get collection
db.collections()                        # List collections
db.delete_collection("users")           # Delete collection

# Persistence
db.flush()                              # Flush to disk
db.close()                              # Close
db.compact()                            # Remove deleted records
db.optimize()                           # Reorder for cache locality
db.merge_from(other_db)                 # Merge databases

# Config
db.get_ef_search()                      # Get search quality
db.set_ef_search(200)                   # Set search quality
db.dimensions                           # Vector dimensionality
db.stats()                              # Database statistics
```

### Node.js

```javascript
// Database
const db = omendb.open(path, dimensions, { embeddingFn: fn });
const db = omendb.open(path, dimensions);

// CRUD
db.set(items);
db.get(id);
db.getBatch(ids);
db.delete(ids);
db.deleteByFilter(filter);
db.update(id, { vector, metadata, text });

// Search
db.search(query, k);
db.search(query, k, { filter, maxDistance, ef });
db.searchBatch(queries, k);

// Hybrid
db.searchHybrid(queryVector, queryText, k);
db.searchText(queryText, k);

// Collections
db.collection("users");
db.collections();
db.deleteCollection("users");

// Persistence
db.flush();
db.close();
db.compact();
db.optimize();
```

## Configuration

```python
db = omendb.open(
    "./mydb",                # Creates ./mydb.omen + ./mydb.wal
    dimensions=384,
    m=16,                    # HNSW connections per node (default: 16)
    ef_construction=200,     # Index build quality (default: 100)
    ef_search=100,           # Search quality (default: 100)
    quantization=True,       # SQ8 quantization (default: None)
    metric="cosine",         # Distance metric (default: "l2")
    embedding_fn=embed,      # Auto-embed documents and string queries
)

# Quantization options:
# - True or "sq8": SQ8 ~4x smaller, ~99% recall (recommended)
# - "rabitq": RaBitQ ~32x smaller, ~95% recall (rescore restores exact recall)
# - None/False: Full precision (default)

# Distance metric options:
# - "l2" or "euclidean": Euclidean distance (default)
# - "cosine": Cosine distance (1 - cosine similarity)
# - "dot" or "ip": Inner product (for MIPS)

# Context manager (auto-flush on exit)
with omendb.open("./db", dimensions=768) as db:
    db.set([...])
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

## Hybrid Search

Combine vector similarity with BM25 full-text search using RRF fusion:

```python
# With embedding_fn -- pass a string for both vector and text query
db = omendb.open("./mydb", dimensions=384, embedding_fn=embed)
db.set([
    {"id": "doc1", "document": "Paris is the capital of France", "metadata": {"topic": "geography"}},
])

results = db.search_hybrid("capital of France", k=10)

# With manual vectors
db.search_hybrid(query_vector, "query text", k=10)

# Tune alpha: 0 = text only, 1 = vector only, default = 0.5
db.search_hybrid(query_vector, "query text", k=10, alpha=0.7)

# Get separate keyword and semantic scores for debugging/tuning
results = db.search_hybrid(query_vector, "query text", k=10, subscores=True)
# Returns: {"id": "...", "score": 0.85, "keyword_score": 0.92, "semantic_score": 0.78}

# Text-only BM25
db.search_text("capital of France", k=10)
```

## Multi-vector (ColBERT)

MUVERA with MaxSim scoring for ColBERT-style token-level retrieval. Token pooling via k-means reduces storage by 50%.

```python
mvdb = omendb.open(":memory:", dimensions=128, multi_vector=True)
mvdb.set([{
    "id": "doc1",
    "vectors": [[0.1]*128, [0.2]*128, [0.3]*128],  # Token embeddings
}])
results = mvdb.search([[0.1]*128, [0.15]*128], k=5)  # MaxSim scoring
```

## CLI

```bash
cargo install omendb

omendb info ./mydb                        # Database info
omendb stats ./mydb                       # Detailed statistics
omendb count ./mydb                       # Vector count
omendb ids ./mydb                         # List all IDs
omendb search ./mydb -q 0.1,0.2,... -k 5 # Search
omendb get ./mydb doc1                    # Get by ID
omendb collections ./mydb                 # List collections
omendb bench ./mydb                       # Benchmark
omendb export ./mydb -o data.json         # Export
omendb compact ./mydb                     # Compact database
```

## Performance

**SIFT-10K** (128D, M=16, ef=100, k=10, Apple M3 Max):

| Metric      | OmenDB | hnswlib | Ratio |
| ----------- | ------ | ------- | ----- |
| Search QPS  | 21,335 | 14,776  | 1.44x |
| Build vec/s | 67,000 | 47,000  | 1.43x |
| Recall@10   | 100.0% | 99.5%   | --    |
| Batch QPS   | 92,000 | --      | --    |

**SIFT-1M** (1M vectors, 128D, M=16, ef=100, k=10):

| Machine      | QPS   | Recall |
| ------------ | ----- | ------ |
| i9-13900KF   | 4,591 | 98.6%  |
| Apple M3 Max | 3,216 | 98.4%  |

**Quantization:**

| Mode   | Compression | Recall                   | Use Case             |
| ------ | ----------- | ------------------------ | -------------------- |
| f32    | 1x          | 100%                     | Default              |
| SQ8    | 4x          | ~99%                     | Recommended for most |
| RaBitQ | 32x         | ~95% (100% with rescore) | Large datasets       |

```python
db = omendb.open("./db", dimensions=768, quantization=True)          # SQ8
db = omendb.open("./db", dimensions=768, quantization="rabitq")      # RaBitQ
```

**Filtered search** (ACORN-1, SIFT-10K, 10% selectivity):

| Method  | QPS | Recall | Speedup               |
| ------- | --- | ------ | --------------------- |
| ACORN-1 | --  | --     | 37.79x vs post-filter |

<details>
<summary>Benchmark methodology</summary>

- **Parameters**: m=16, ef_construction=100, ef_search=100
- **Batch**: Uses Rayon for parallel search across all cores
- **Recall**: Validated against brute-force ground truth on SIFT/GloVe
- **Reproduce**:
  - Quick (10K): `uv run python benchmarks/run.py`
  - SIFT-1M: `uv run python benchmarks/ann_dataset_test.py --dataset sift-128-euclidean`

</details>

## Tuning

The `ef_search` parameter controls the recall/speed tradeoff at query time. Higher values explore more candidates, improving recall but slowing search.

**Rules of thumb:**

- `ef_search` must be >= k (number of results requested)
- For 128D embeddings: ef=100 usually achieves 90%+ recall
- For 768D+ embeddings: increase to ef=200-400 for better recall
- If recall drops at scale (50K+), increase both ef_search and ef_construction

**Runtime tuning:**

```python
# Check current value
print(db.get_ef_search())  # 100

# Increase for better recall (slower)
db.set_ef_search(200)

# Decrease for speed (may reduce recall)
db.set_ef_search(50)

# Per-query override
results = db.search(query, k=10, ef=300)
```

**Recommended settings by use case:**

| Use Case            | ef_search | Expected Recall |
| ------------------- | --------- | --------------- |
| Fast search (128D)  | 64        | ~85%            |
| Balanced (default)  | 100       | ~90%            |
| High recall (768D+) | 200-300   | ~95%+           |
| Maximum recall      | 500+      | ~98%+           |

## Examples

See complete working examples:

- [`python/examples/quickstart.py`](python/examples/quickstart.py) -- Minimal Python example
- [`python/examples/basic.py`](python/examples/basic.py) -- CRUD operations and persistence
- [`python/examples/filters.py`](python/examples/filters.py) -- All filter operators
- [`python/examples/rag.py`](python/examples/rag.py) -- RAG workflow with mock embeddings
- [`python/examples/embedding_fn.py`](python/examples/embedding_fn.py) -- Auto-embedding with embedding_fn
- [`python/examples/quantization.py`](python/examples/quantization.py) -- SQ8 and RaBitQ quantization
- [`node/examples/quickstart.js`](node/examples/quickstart.js) -- Minimal Node.js example
- [`node/examples/embedding_fn.js`](node/examples/embedding_fn.js) -- Auto-embedding with embeddingFn
- [`node/examples/multivector.ts`](node/examples/multivector.ts) -- Multi-vector / ColBERT

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

[Elastic License 2.0](LICENSE) -- Free to use, modify, and embed. The only restriction: you can't offer OmenDB as a managed service to third parties.
