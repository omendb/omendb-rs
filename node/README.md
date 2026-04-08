# omendb

Fast embedded vector database with HNSW indexing for Node.js and Bun.

## Installation

```bash
npm install @omendb/omendb
```

## Quick Start

```typescript
import { open } from "omendb";

// Open or create a database
const db = open("./vectors");

// Insert vectors
db.set([
  {
    id: "doc1",
    vector: new Float32Array([0.1, 0.1, 0.1]),
    metadata: { title: "Hello" },
  },
  {
    id: "doc2",
    vector: new Float32Array([0.2, 0.2, 0.2]),
    metadata: { category: "news" },
  },
]);

// Search
const results = db.search(new Float32Array([0.15, 0.15, 0.15]), 5);
// [{ id: 'doc1', distance: 0.05, metadata: { title: 'Hello' } }, ...]

// Batch search (async, parallel)
const batchResults = await db.searchBatch(queries, 10);

// Close when done (releases file locks)
db.close();
```

## Features

- HNSW indexing for fast approximate nearest neighbor search
- ACORN-1 filtered search
- SQ8 quantization (4x compression, ~99% recall)
- Hybrid search (vector + BM25 text)
- Collections for multi-tenancy
- Persistent storage with auto-save
- Works with Node.js 18+ and Bun

## API

### Opening a Database

```typescript
import { open } from "omendb";

// Basic
const db = open("./vectors", { dimensions: 384 });

// Single-vector dimensions can be inferred on first insert
const inferred = open(":memory:");

// In-memory
const memDb = open(":memory:", { dimensions: 128 });

// Full options
const db = open("./vectors", {
  dimensions: 768,
  m: 16, // HNSW connections per node (default: 16)
  efConstruction: 100, // Build quality (default: 100)
  efSearch: 100, // Search quality (default: 100)
  quantization: true, // SQ8: 4x compression, ~99% recall
  metric: "cosine", // "l2", "cosine", or "dot"
});

// Multi-vector stores still require explicit token dimensions
const mvdb = open(":memory:", {
  dimensions: 128,
  multiVector: true,
  textSearch: { tokenizer: "code", writerBufferMb: 64 },
});
```

### Core Operations

#### `db.set(items)`

Insert or update vectors.

```typescript
db.set([
  { id: "doc1", vector: Float32Array, metadata?: object },
  { id: "doc2", vector: Float32Array, metadata?: object },
]);
```

#### `db.get(id)`

Get a vector by ID.

```typescript
const item = db.get("doc1");
// { id: "doc1", vector: Float32Array, metadata: {...} } or null
```

#### `db.getBatch(ids)`

Get multiple vectors by ID.

```typescript
const items = db.getBatch(["doc1", "doc2"]);
// [{ id, vector, metadata } | null, ...]
```

#### `db.update(id, options)`

Update a vector's data and/or metadata.

```typescript
db.update("doc1", {
  vector: newVector, // Optional
  metadata: { title: "New" }, // Optional
});
```

#### `db.delete(ids)`

Delete vectors by ID.

```typescript
const deleted = db.delete(["doc1", "doc2"]);
// Returns number deleted
```

#### `db.deleteByFilter(filter)`

Delete vectors matching a filter.

```typescript
const deleted = db.deleteByFilter({ category: "old" });
const deleted = db.deleteByFilter({
  $and: [{ type: "draft" }, { age: { $gt: 30 } }],
});
```

### Search

#### `db.search(query, k, options?)`

Search for k nearest neighbors (sync).

```typescript
const results = db.search(queryVector, 10); // Basic
const results = db.search(queryVector, 10, {
  ef: 200, // Search quality (higher = better recall)
  filter: { category: "news" }, // Metadata filter
  maxDistance: 0.5, // Distance threshold
});
// [{ id, distance, metadata }, ...]
```

#### `db.searchBatch(queries, k, ef?)`

Batch search with parallel execution (async).

```typescript
const results = await db.searchBatch(queries, 10, 100);
// [[{ id, distance, metadata }, ...], ...]
```

### Text & Hybrid Search

#### `db.enableTextSearch(config?)`

Enable text indexing for hybrid search.

```typescript
db.enableTextSearch(); // Default buffer + tokenizer
db.enableTextSearch({ bufferMb: 128 }); // Custom buffer size
db.enableTextSearch({ tokenizer: "code" }); // Code-aware tokenization
db.enableTextSearch({ writerBufferMb: 64, tokenizer: "raw" });
```

#### `db.hasTextSearch`

Check if text search is enabled.

```typescript
if (db.hasTextSearch) { ... }
```

#### `db.setWithText(items)`

Insert vectors with text content.

```typescript
db.setWithText([
  { id: "doc1", vector: vec, text: "Machine learning tutorial", metadata: {...} }
]);
```

#### `db.searchText(query, k)`

BM25 text-only search.

```typescript
const results = db.searchText("machine learning", 10);
// [{ id, score, metadata }, ...]
```

#### `db.searchHybrid(queryVector, queryText, k, options?)`

Combined vector + text search using Reciprocal Rank Fusion.

```typescript
// Basic
const results = db.searchHybrid(queryVector, "machine learning", 10);

// With options
const results = db.searchHybrid(queryVector, "machine learning", 10, {
  alpha: 0.7, // 0=text only, 1=vector only (default: 0.5)
  rrfK: 60, // RRF constant (default: 60)
  filter: { category: "ml" },
  subscores: true, // Include separate scores
});
// [{ id, score, metadata, keywordScore?, semanticScore? }, ...]
```

### Collections

#### `db.collection(name)`

Get or create a named collection.

```typescript
const users = db.collection("users");
users.set([...]);
users.search(query, 5);
```

#### `db.collections()`

List all collections.

```typescript
const names = db.collections();
// ["users", "products", ...]
```

#### `db.deleteCollection(name)`

Delete a collection.

```typescript
db.deleteCollection("old_collection");
```

### Properties

```typescript
db.length; // Number of vectors
db.dimensions; // Vector dimensionality
db.efSearch; // Get/set search quality parameter

db.efSearch = 200; // Tune for better recall
```

### Utility Methods

#### `db.count(filter?)`

Count vectors, optionally with filter.

```typescript
const total = db.count();
const filtered = db.count({ category: "news" });
```

#### `db.isEmpty()`

Check if database is empty.

#### `db.exists(id)`

Check if an ID exists.

```typescript
if (db.exists("doc1")) { ... }
```

#### `db.ids()`

Get all vector IDs.

```typescript
const allIds = db.ids();
```

#### `db.items()`

Get all vectors with metadata.

```typescript
const allItems = db.items();
// [{ id, vector, metadata }, ...]
```

#### `db.stats()`

Get index statistics.

```typescript
const stats = db.stats();
// { numVectors, dimensions, maxLevel, avgNeighborsL0, ... }
```

### Persistence

#### `db.flush()`

Force write pending changes to disk.

```typescript
db.flush();
```

#### `db.compact()`

Remove deleted records and reclaim space.

```typescript
const removed = db.compact();
```

#### `db.optimize()`

Reorder graph for better cache locality (6-40% speedup).

```typescript
const reordered = db.optimize();
```

#### `db.close()`

Close database and release file locks.

```typescript
db.close();
// Can now reopen the same path
```

#### `db.mergeFrom(other)`

Merge another database into this one.

```typescript
const merged = db.mergeFrom(otherDb);
```

### Filter Operators

```typescript
// Equality
{ field: "value" }                    // Shorthand
{ field: { $eq: "value" } }           // Explicit

// Comparison
{ field: { $ne: "value" } }           // Not equal
{ field: { $gt: 10 } }                // Greater than
{ field: { $gte: 10 } }               // Greater or equal
{ field: { $lt: 10 } }                // Less than
{ field: { $lte: 10 } }               // Less or equal

// Membership
{ field: { $in: ["a", "b"] } }        // In list
{ field: { $nin: ["a", "b"] } }       // Not in list

// Logical
{ $and: [{...}, {...}] }              // AND
{ $or: [{...}, {...}] }               // OR
```

## Performance

Node bindings call the same Rust core as the Rust and Python APIs, so authoritative ANN numbers are tracked at the repo root with the shared SIFT benchmark on Fedora/Linux medians.

Use the shared benchmark for comparable performance claims:

```bash
cd python && uv run python benchmark.py
cd python && uv run python benchmark.py --publish
```

Older Apple M3 Max wrapper numbers were local reference points only and are no longer treated as the authoritative baseline.

## License

[Elastic License 2.0](../LICENSE)
