# omendb

Fast embedded vector database with HNSW indexing for Node.js and Bun.

## Installation

```bash
npm install omendb
```

## Usage

```typescript
import { open } from "omendb";

// Open or create a database
const db = open("./vectors", { dimensions: 384 });

// Insert vectors
db.set([
  {
    id: "doc1",
    vector: new Float32Array(384).fill(0.1),
    metadata: { title: "Hello" },
  },
  {
    id: "doc2",
    vector: new Float32Array(384).fill(0.2),
    metadata: { title: "World" },
  },
]);

// Search
const results = db.search(new Float32Array(384).fill(0.15), { k: 5 });
console.log(results);
// [{ id: 'doc1', distance: 0.05, metadata: { title: 'Hello' } }, ...]

// Batch search (async, parallel)
const batchResults = await db.searchBatch(queries, { k: 10 });
```

## Features

- HNSW indexing for fast approximate nearest neighbor search
- ACORN-1 filtered search
- SQ8 quantization (4x compression, ~99% recall)
- Collections for multi-tenancy
- Persistent storage with auto-save
- Works with Node.js 18+ and Bun

## API

### `open(path, options?)`

Open or create a vector database.

- `path`: Database directory path
- `options.dimensions`: Vector dimensionality (default: 128)

### `db.set(items)`

Insert or update vectors.

### `db.search(query, options)`

Search for k nearest neighbors (sync).

### `db.searchBatch(queries, options)`

Batch search with parallel execution (async).

### `db.get(id)`

Get a vector by ID.

### `db.delete(ids)`

Delete vectors by ID.

## License

Apache-2.0
