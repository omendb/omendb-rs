# oadb - Fast Embedded Vector Database

**Fast embedded vector search with Extended RaBitQ compression**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](../LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

Install via pip (coming soon to PyPI):

```bash
pip install oadb
```

**5-line example:**

```python
import oadb

db = oadb.open("./my_vectors", dimensions=3)
db.set(["doc1"], [[0.1, 0.2, 0.3]], [{"title": "Hello"}])
results = db.search(query=[0.1, 0.2, 0.3], k=5)
print(results[0])  # {'id': 'doc1', 'distance': 0.0, 'metadata': {'title': 'Hello'}}
```

## Performance

**21% faster than ChromaDB at 10K vectors (in-memory, ef_search=50):**

| Metric | oadb (in-memory) | oadb (persistent) | ChromaDB |
|--------|------------------|-------------------|----------|
| **Search QPS** | **3,415** | 2,542 | 2,812 |
| **Latency** | **0.29ms** | 0.39ms | 0.36ms |
| **Insert** | 27K vec/s | 20K vec/s | 26K vec/s |
| **Recall** | ~93% (ef=50) | ~93% (ef=50) | ~88% |

**Tune speed vs accuracy:**
```python
db.set_ef_search(50)   # Fast: 3,415 QPS, ~93% recall
db.set_ef_search(100)  # Balanced: 2,226 QPS, ~99% recall
db.set_ef_search(200)  # Accurate: 1,510 QPS, ~100% recall
```

**Algorithm**: HNSW + Extended RaBitQ (8x memory compression)

**Quick Performance Test** (<10 seconds):
```bash
# Run quick comparison (1K vectors)
cd python
uv run python quick_test.py
```

Compares oadb vs ChromaDB vs hnswlib (C++ baseline). Use during development to quickly validate changes.

## Key Features

- ✅ **Fast search**: 3,415 QPS @ 10K vectors, 0.29ms latency (21% faster than ChromaDB)
- ✅ **Tunable recall**: 93-100% via ef_search parameter
- ✅ **Memory efficient**: Extended RaBitQ with 8x compression (4-bit quantization)
- ✅ **Persistent**: seerdb LSM backend with auto-persist
- ✅ **Metadata filtering**: MongoDB-style filters ($eq, $ne, $gt, $in, $and, $or) + ACORN-1 (37x speedup)
- ✅ **Integrations**: LangChain + LlamaIndex VectorStore
- ✅ **Production-ready**: 513 tests passing (353 Rust + 160 Python)

## Installation

### From PyPI (coming soon)

```bash
pip install oadb
```

### From source

```bash
git clone https://github.com/omendb/omendb.git
cd omendb/python
pip install maturin
maturin develop --release
```

## Usage

### Basic Operations

```python
import oadb

# Open or create database
db = oadb.open("./data", dimensions=3)

# Store vectors (insert or replace)
db.set(
    ["doc1", "doc2"],                           # IDs
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],        # Vectors
    [{"title": "First", "year": 2024},          # Metadata
     {"title": "Second", "year": 2023}]
)

# Search (k nearest neighbors)
results = db.search(query=[0.1, 0.2, 0.3], k=5)
for result in results:
    print(f"{result['id']}: distance={result['distance']:.3f}")

# Search with metadata filter (ACORN-1: 37x faster than post-filter)
results = db.search(
    query=[0.1, 0.2, 0.3],
    k=10,
    filter={"year": {"$gte": 2024}}
)

# Get vector by ID
doc = db.get(["doc1"])[0]
print(doc["vector"])  # [0.1, 0.2, 0.3]

# Update metadata only (keeps existing vector)
db.update(["doc1"], [{"title": "Updated Title", "year": 2024}])

# Delete vectors
db.delete(["doc1", "doc2"])

# Get count
print(len(db))     # Pythonic
print(db.count())  # Explicit
```

### Metadata Filtering

Supports MongoDB-style filter operators:

```python
# Equality
filter={"category": "research"}

# Comparison
filter={"year": {"$gte": 2020, "$lt": 2025}}

# Contains
filter={"tags": {"$in": ["python", "rust"]}}

# Logical operators
filter={
    "$and": [
        {"year": {"$gte": 2020}},
        {"category": "research"}
    ]
}
```

## API Reference

See docstrings for detailed API documentation:

```python
import oadb
help(oadb.open)
help(oadb.VectorDatabase)
```

## Performance Benchmarks

### Search Performance (10K vectors, 128 dimensions)

| ef_search | QPS (in-memory) | QPS (persistent) | Recall |
|-----------|-----------------|------------------|--------|
| 50 | 3,415 | 2,542 | ~93% |
| 100 | 2,226 | 1,785 | ~99% |
| 200 | 1,510 | 1,168 | ~100% |

**vs ChromaDB**: 2,812 QPS (~88% recall)

### Insert Performance (10K vectors, 128 dimensions)

- **In-memory**: 27,000 vectors/second
- **Persistent**: 20,000 vectors/second
- **Batch operations**: Much more efficient than individual inserts

### Memory

- **Extended RaBitQ**: 8x compression with ~100% recall @ 4-bit
- **ACORN-1 filtered search**: 37x speedup vs post-filtering

## Examples

See the `examples/` directory (coming soon) for:

- **basic.py**: Simple vector insert and search
- **rag.py**: Retrieval-augmented generation (RAG) workflow
- **filters.py**: Advanced metadata filtering

## Documentation

Full documentation available at: [https://github.com/omendb/omendb](https://github.com/omendb/omendb)

- [Architecture Design](../ai/oadb/design/python_api.md)
- [Performance Analysis](../ai/oadb/performance/)
- [Project Status](../ai/oadb/STATUS.md)

## Requirements

- Python 3.8+
- No additional dependencies (numpy is included)

## License

**Apache License 2.0** (open source)

- ✅ Free to use, modify, and redistribute
- ✅ Commercial use allowed
- ✅ No restrictions on embedding in your app

See [LICENSE](../LICENSE) for full terms.

## Contributing

Contributions welcome! Please see the main repository for guidelines:

- [Contributing Guide](../CLAUDE.md)
- [Development Environment](../CLAUDE.md#development-environment)
- [Testing Requirements](../CLAUDE.md#contributing--development-principles)

## Support

- **Issues**: [GitHub Issues](https://github.com/omendb/omendb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/omendb/omendb/discussions)

## Roadmap

**Current**: Pre-release polish (November 2025)

**Next**:
- RaBitQ FHT optimization (10-80x encoding speedup)
- API stability review
- PyPI packaging (`pip install oadb`)

See [TODO.md](../ai/oadb/TODO.md) for detailed roadmap.

---

**Status**: Alpha (v0.1.0) - 513 tests passing, pre-release in progress

Built with Rust + PyO3
