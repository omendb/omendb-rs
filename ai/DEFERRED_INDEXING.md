# Deferred HNSW Indexing

**Status**: Planning
**Priority**: P2
**Target**: 0.0.22 (current released: 0.0.20, unreleased: 0.0.21)
**Downstream**: cloud-q0rl (blocked on this)

## Problem

HNSW insertion is O(n log n) - each insert searches the graph to find neighbors. At scale:

- 100K vectors: 795 vec/sec
- 1M vectors: 391 vec/sec (2x slower)

For bulk loads, this is the bottleneck.

## Solution

Deferred indexing: store vectors first, build graph once at the end.

Competitors doing this:

- **Qdrant**: `m=0` during bulk, re-enable after
- **Milvus**: Separate insert/index phases

## Proposed API

```rust
// Option A: Builder flag
let mut index = HNSWBuilder::new(dims)
    .deferred_indexing(true)
    .build()?;

index.insert(&vec)?;      // O(1) - just stores vector
index.insert(&vec2)?;     // O(1)
index.build_index()?;     // O(n log n) - builds graph once

// Option B: Separate methods (more explicit)
index.insert_deferred(&vec)?;  // Store only
index.build_index()?;          // Build graph
```

## Implementation Plan

### Phase 1: Core API

1. Add `deferred: bool` field to `HNSWIndex`
2. Add `pending_vectors: Vec<(u32, Vec<f32>)>` for deferred storage
3. Modify `insert()` to check deferred flag
4. Add `build_index()` method that:
   - Takes all pending vectors
   - Calls existing `batch_insert()` logic
   - Clears pending queue

### Phase 2: Search behavior

- If `deferred && !pending_vectors.is_empty()`:
  - Option A: Return error "index not built"
  - Option B: Auto-build on first search (lazy)
  - Option C: Brute-force search on pending (slow but works)

Recommend: Option A for clarity, Option B as opt-in.

### Phase 3: Incremental builds

- `build_index()` can be called multiple times
- Each call builds graph for NEW pending vectors only
- Merges with existing graph

## Files to Modify

| File                              | Changes                                  |
| --------------------------------- | ---------------------------------------- |
| `src/vector/hnsw_index.rs`        | Add builder option, deferred field       |
| `src/vector/hnsw/index/mod.rs`    | Add pending storage                      |
| `src/vector/hnsw/index/insert.rs` | Add `insert_deferred()`, `build_index()` |
| `src/vector/hnsw/index/search.rs` | Handle deferred state                    |

## Testing

1. Unit tests for deferred insert/build cycle
2. Benchmark: deferred vs regular batch_insert at 100K, 1M
3. Test search-before-build error handling
4. Test incremental build_index() calls

## Expected Results

- Bulk insert: 10-50x faster (matching Qdrant claims)
- Memory: Similar (still need to store vectors)
- Search: Same performance after build_index()

## Dependencies

None - self-contained feature in omendb.

## Downstream

cloud-lsm-vec (cloud-q0rl) will use this for L0 batch inserts.
