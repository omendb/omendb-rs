# OmenDB JavaScript Bindings Plan

## Overview

Native Node.js/Bun bindings for OmenDB using napi-rs v3. Target: TUI apps, agents, web servers.

## Technology Choice: napi-rs v3

| Criteria    | napi-rs v3                    | Alternatives                    |
| ----------- | ----------------------------- | ------------------------------- |
| Performance | Native speed, lowest overhead | Neon 2x slower, WASM ~2x slower |
| Bun compat  | 98% (v1.2.5+)                 | Partial for others              |
| TypeScript  | Auto-generated .d.ts          | Manual for Neon                 |
| Async       | Native tokio support          | Limited in Neon                 |
| Adoption    | Next.js, Rspack, Oxc, Turbo   | -                               |

**Decision**: napi-rs v3 (current stable as of 2025)

## Project Structure

```
node/
├── .cargo/
│   └── config.toml           # Cross-compilation config
├── .github/
│   └── workflows/
│       └── CI.yml            # Multi-platform builds
├── src/
│   └── lib.rs                # Rust bindings (~800-1000 lines)
├── npm/                      # Auto-generated platform packages
│   ├── darwin-arm64/
│   ├── darwin-x64/
│   ├── linux-arm64-gnu/
│   ├── linux-x64-gnu/
│   └── win32-x64-msvc/
├── __test__/
│   └── index.spec.ts         # Vitest tests
├── build.rs                  # napi_build::setup()
├── Cargo.toml
├── package.json
├── index.js                  # Auto-generated loader
├── index.d.ts                # Auto-generated types
└── README.md
```

## API Design

Port Python API 1:1 for consistency. TypeScript types auto-generated.

```typescript
// Generated types (index.d.ts)
export interface SearchResult {
  id: string;
  distance: number;
  metadata: Record<string, unknown>;
}

export interface VectorItem {
  id: string;
  embedding: number[] | Float32Array;
  metadata?: Record<string, unknown>;
  document?: string;
}

export interface SearchOptions {
  k: number;
  ef?: number;
  filter?: MetadataFilter;
}

export class VectorDatabase {
  // Core operations
  set(items: VectorItem[]): number[];
  set(id: string, embedding: number[], metadata?: object): number[];

  search(
    query: number[] | Float32Array,
    options: SearchOptions,
  ): SearchResult[];
  searchBatch(
    queries: number[][] | Float32Array[],
    options: SearchOptions,
  ): SearchResult[][];

  get(id: string): VectorItem | null;
  delete(ids: string[]): number;
  update(id: string, embedding: number[], metadata?: object): void;

  // Persistence
  save(): void;

  // Collections
  collection(name: string): VectorDatabase;
  collections(): string[];
  deleteCollection(name: string): void;

  // Tuning
  get efSearch(): number;
  set efSearch(value: number);
  get count(): number;

  // Merge
  mergeFrom(other: VectorDatabase): number;
}

// Module-level
export function open(path: string, options?: OpenOptions): VectorDatabase;
```

## Implementation Mapping

| Python (PyO3)         | Node (napi-rs)              | Notes                   |
| --------------------- | --------------------------- | ----------------------- |
| `#[pyclass]`          | `#[napi]` on struct         | Same pattern            |
| `#[pymethods]`        | `#[napi]` on impl           | Same pattern            |
| `PyResult<T>`         | `Result<T>`                 | napi handles conversion |
| `Python<'_>`          | `Env`                       | JS environment handle   |
| `Bound<'_, PyAny>`    | `JsUnknown`                 | Dynamic JS value        |
| `PyDict`              | `Object`                    | JS object               |
| `Vec<f32>`            | `Float32Array` / `Vec<f64>` | TypedArray preferred    |
| `parking_lot::RwLock` | Same                        | Thread safety           |
| `py.allow_threads()`  | Automatic in async          | GIL not applicable      |

## Key Implementation Details

### 1. Buffer Handling (Critical for Performance)

```rust
use napi::bindgen_prelude::*;

#[napi]
impl VectorDatabase {
  // Accept both Array and Float32Array for flexibility
  #[napi]
  pub fn search(&self, query: Either<Vec<f64>, Float32Array>, k: u32) -> Result<Vec<SearchResult>> {
    let query_vec: Vec<f32> = match query {
      Either::A(arr) => arr.into_iter().map(|x| x as f32).collect(),
      Either::B(typed) => typed.to_vec(),  // Zero-copy when possible
    };
    // ...
  }
}
```

### 2. Async Batch Search (Release Event Loop)

```rust
use napi::tokio;

#[napi]
impl VectorDatabase {
  #[napi]
  pub async fn search_batch(
    &self,
    queries: Vec<Float32Array>,
    k: u32,
  ) -> Result<Vec<Vec<SearchResult>>> {
    // Runs on tokio runtime, doesn't block Node event loop
    let results = self.inner.read().store
      .batch_search_parallel_with_metadata(&query_vecs, k as usize, None);
    // ...
  }
}
```

### 3. Error Handling

```rust
use napi::{Error, Status};

fn convert_error(err: anyhow::Error) -> Error {
  let msg = err.to_string();
  if msg.contains("dimension") {
    Error::new(Status::InvalidArg, msg)
  } else {
    Error::new(Status::GenericFailure, msg)
  }
}
```

### 4. Thread Safety (Same as Python)

```rust
use parking_lot::RwLock;

#[napi]
pub struct VectorDatabase {
  inner: RwLock<VectorDatabaseInner>,
  path: String,
  dimensions: u32,
  is_persistent: bool,
}
```

## Dependencies

```toml
# node/Cargo.toml
[package]
name = "omendb-node"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "3", features = ["async", "tokio_rt"] }
napi-derive = "3"
omendb = { path = ".." }
parking_lot = "0.12"
serde_json = "1"

[build-dependencies]
napi-build = "2"

[profile.release]
lto = true
codegen-units = 1
strip = true
```

```json
// node/package.json
{
  "name": "@omendb/node",
  "version": "0.1.0",
  "main": "index.js",
  "types": "index.d.ts",
  "napi": {
    "binaryName": "omendb",
    "targets": [
      "x86_64-apple-darwin",
      "aarch64-apple-darwin",
      "x86_64-unknown-linux-gnu",
      "aarch64-unknown-linux-gnu",
      "x86_64-pc-windows-msvc"
    ]
  },
  "scripts": {
    "build": "napi build --release",
    "build:debug": "napi build",
    "test": "vitest run",
    "prepublishOnly": "napi prepublish -t npm"
  },
  "devDependencies": {
    "@napi-rs/cli": "^3.0.0",
    "vitest": "^2.0.0"
  },
  "files": ["index.js", "index.d.ts"]
}
```

## Implementation Steps

### Phase 1: Scaffold (~1 hour)

1. Create `node/` directory structure
2. Set up Cargo.toml with napi-rs v3
3. Set up package.json with napi config
4. Create build.rs
5. Verify `napi build` works

### Phase 2: Core API (~3-4 hours)

1. Implement `open()` function
2. Implement `VectorDatabase` struct with RwLock
3. Implement `set()` - single and batch
4. Implement `search()` - single query
5. Implement `searchBatch()` - parallel async
6. Implement `get()`, `delete()`, `update()`
7. Implement `save()`

### Phase 3: Collections & Advanced (~1-2 hours)

1. Implement `collection()`, `collections()`, `deleteCollection()`
2. Implement `mergeFrom()`
3. Implement `efSearch` getter/setter
4. Implement metadata filter parsing

### Phase 4: Testing (~1-2 hours)

1. Basic CRUD tests
2. Search accuracy tests
3. Batch search tests
4. Collection tests
5. Bun compatibility verification

### Phase 5: CI/CD (~1 hour)

1. GitHub Actions workflow for multi-platform builds
2. npm publish workflow (manual trigger)

## Estimated Total: 7-10 hours

## Testing Strategy

```typescript
// __test__/index.spec.ts
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { open, VectorDatabase } from "../index";
import { tmpdir } from "os";
import { mkdtemp, rm } from "fs/promises";
import { join } from "path";

describe("VectorDatabase", () => {
  let db: VectorDatabase;
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await mkdtemp(join(tmpdir(), "omendb-test-"));
    db = open(join(tempDir, "test"), { dimensions: 128 });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true });
  });

  it("should set and get vectors", () => {
    const embedding = new Float32Array(128).fill(0.1);
    db.set([{ id: "doc1", embedding, metadata: { title: "Hello" } }]);

    const result = db.get("doc1");
    expect(result?.id).toBe("doc1");
    expect(result?.metadata.title).toBe("Hello");
  });

  it("should search similar vectors", () => {
    // Insert vectors
    const vectors = Array.from({ length: 100 }, (_, i) => ({
      id: `doc${i}`,
      embedding: new Float32Array(128).fill(i / 100),
    }));
    db.set(vectors);

    // Search
    const query = new Float32Array(128).fill(0.5);
    const results = db.search(query, { k: 5 });

    expect(results).toHaveLength(5);
    expect(results[0].id).toBe("doc50"); // Closest to 0.5
  });
});
```

## Bun Compatibility Notes

- napi-rs modules work in Bun v1.2.5+ (98% Node-API compat)
- Test with both `bun test` and `vitest`
- No special handling needed - same binary works

## Design Decisions (Finalized)

### 1. Package Name: `omendb`

**Decision**: Unscoped `omendb`

| Option         | Pros                                                                         | Cons                                          |
| -------------- | ---------------------------------------------------------------------------- | --------------------------------------------- |
| `omendb`       | Clean import, matches Python, follows usearch/lancedb/better-sqlite3 pattern | Platform sub-packages need careful publishing |
| `@omendb/node` | npm spam detection avoided, explicit "this is Node"                          | Verbose import, unnecessary suffix            |

**Rationale**:

- `usearch`, `lancedb`, `vectordb`, `better-sqlite3` all use simple unscoped names
- Users write `import { open } from 'omendb'` - clean and obvious
- Platform packages: `@omendb/linux-x64-gnu`, `@omendb/darwin-arm64` (scoped to avoid spam detection)
- Matches Python package name for cross-language consistency

### 2. Float Precision: Accept Both, Prefer Float32Array

**Decision**: Accept `number[] | Float32Array`, convert internally to `f32`

```typescript
// User-facing API
search(query: number[] | Float32Array, options: SearchOptions): SearchResult[]
set(items: Array<{ embedding: number[] | Float32Array, ... }>): number[]
```

**Implementation**:

```rust
use napi::Either;

#[napi]
pub fn search(&self, query: Either<Vec<f64>, Float32Array>, k: u32) -> Result<Vec<SearchResult>> {
    let query_vec: Vec<f32> = match query {
        Either::A(arr) => arr.into_iter().map(|x| x as f32).collect(),  // f64 → f32
        Either::B(typed) => typed.to_vec(),  // Already f32, minimal copy
    };
    // ...
}
```

**Rationale**:

- `Float32Array` is 2x memory efficient and enables zero-copy in some cases
- `number[]` is more ergonomic (no constructor needed)
- OmenDB stores f32 internally anyway - precision loss from f64→f32 is negligible for embeddings
- Power users can use `Float32Array` for max performance; casual users can use arrays

### 3. Sync vs Async: Sync Default, Async for Batch

**Decision**: All sync except `searchBatch()`

| Method          | Sync/Async | Rationale                                      |
| --------------- | ---------- | ---------------------------------------------- |
| `open()`        | Sync       | Fast, one-time setup                           |
| `set()`         | Sync       | Fast enough for typical use                    |
| `search()`      | Sync       | Single query <10ms typical                     |
| `searchBatch()` | **Async**  | Parallel execution via rayon, can take 100ms+  |
| `get()`         | Sync       | O(1) lookup                                    |
| `delete()`      | Sync       | Fast                                           |
| `update()`      | Sync       | Fast                                           |
| `save()`        | Sync       | Embedded DB, fast flush                        |
| `collection()`  | Sync       | Fast                                           |
| `mergeFrom()`   | Sync       | Keep simple, user can wrap in worker if needed |

**Rationale** (from better-sqlite3 philosophy):

- Async overhead (Promise creation, microtask scheduling) can exceed operation time for fast ops
- Single `search()` on 100K vectors: ~1-5ms — async overhead not worth it
- Batch `searchBatch()` with 100 queries: ~50-200ms — async allows event loop to continue
- Sync API is simpler to use, easier to reason about, fewer bugs
- Matches better-sqlite3's proven approach for embedded databases

**searchBatch implementation**:

```rust
#[napi]
impl VectorDatabase {
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<Float32Array>,
        k: u32,
    ) -> Result<Vec<Vec<SearchResult>>> {
        // Runs on tokio runtime, parallel via rayon internally
        // Returns Promise to JS, doesn't block event loop
    }
}
```

Usage:

```typescript
// Sync - simple, fast single query
const results = db.search(query, { k: 10 });

// Async - parallel batch, non-blocking
const batchResults = await db.searchBatch(queries, { k: 10 });
```
