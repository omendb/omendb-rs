# Deferred HNSW Indexing

**Status**: Implemented (pending release)
**Priority**: P2
**Target**: 0.0.22 (current released: 0.0.20, unreleased: 0.0.21)
**Downstream**: cloud-q0rl (blocked on this)

## Problem

Single-vector `insert()` is O(log n) per vector due to graph traversal. At scale:

- 100K vectors: 795 vec/sec (incremental insert)
- 1M vectors: 391 vec/sec (2x slower)

For bulk loads, this is the bottleneck.

## Key Insight

**The existing `batch_insert()` already uses deferred-style internally:**

```rust
// src/vector/hnsw/index/insert.rs lines 249-510
pub fn batch_insert(&mut self, vectors: Vec<Vec<f32>>) -> Result<Vec<u32>> {
    // Phase 1: Store all vectors (sequential, fast)
    for vector in vectors { ... }

    // Phase 2: Build graph in parallel (rayon)
    nodes_to_insert.par_iter().try_for_each(...)

    // Phase 3: Prune over-connected nodes
    for node_id in 0..max_node_id { ... }
}
```

**The problem is API ergonomics**, not algorithm. Users need to:

1. Collect ALL vectors upfront
2. Call `batch_insert()` once

This doesn't work for streaming inserts where vectors arrive over time.

## Solution: Streaming Deferred API

Add a thin API layer that accumulates vectors, then calls `batch_insert()`:

```rust
// Builder flag approach
let mut index = HNSWIndex::builder()
    .dimensions(128)
    .deferred(true)  // Enable deferred mode
    .build()?;

index.insert(&vec1)?;  // O(1) - just stores in pending
index.insert(&vec2)?;  // O(1) - just stores in pending
index.build_index()?;  // Calls batch_insert() internally
```

## Implementation (Minimal)

### Changes to `src/vector/hnsw_index.rs`

```rust
pub struct HNSWIndex {
    // ... existing fields ...

    /// Deferred mode: vectors stored but graph not built
    deferred: bool,

    /// Pending vectors awaiting graph construction
    pending_vectors: Vec<Vec<f32>>,
}

impl HNSWIndex {
    /// Insert vector (deferred mode: store only; normal mode: full insert)
    pub fn insert(&mut self, vector: &[f32]) -> Result<usize> {
        if self.deferred {
            self.pending_vectors.push(vector.to_vec());
            return Ok(self.pending_vectors.len() - 1);
        }
        // ... existing insert logic ...
    }

    /// Build graph for all pending vectors
    pub fn build_index(&mut self) -> Result<Vec<u32>> {
        if self.pending_vectors.is_empty() {
            return Ok(vec![]);
        }
        let vectors = std::mem::take(&mut self.pending_vectors);
        self.index.batch_insert(vectors)
    }

    /// Check if there are pending vectors
    pub fn has_pending(&self) -> bool {
        !self.pending_vectors.is_empty()
    }

    /// Get count of pending vectors
    pub fn pending_count(&self) -> usize {
        self.pending_vectors.len()
    }
}
```

### Builder addition

```rust
impl HNSWIndexBuilder {
    /// Enable deferred indexing mode
    pub fn deferred(mut self, deferred: bool) -> Self {
        self.deferred = deferred;
        self
    }
}
```

### Search behavior

```rust
pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
    if self.has_pending() {
        anyhow::bail!("Index has {} pending vectors. Call build_index() first.",
                      self.pending_count());
    }
    // ... existing search ...
}
```

## Files to Modify

| File                       | Changes                                        |
| -------------------------- | ---------------------------------------------- |
| `src/vector/hnsw_index.rs` | Add deferred field, pending vec, build_index() |

**That's it.** No changes to core HNSW algorithms needed.

## Testing

1. Deferred insert accumulates vectors without graph construction
2. `build_index()` processes all pending vectors
3. Search before build returns error
4. Multiple build_index() calls work (incremental)
5. Benchmark: deferred vs incremental at 100K

## Expected Results

Same performance as `batch_insert()` (10-50x faster than incremental).

## Downstream

cloud-lsm-vec will use this for L0 flush: accumulate vectors during writes,
call `build_index()` when flushing to SSTable.
