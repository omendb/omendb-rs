# OmenDB Embedded Code Review

**Date:** 2026-01-09
**Reviewer:** Claude Opus 4.5
**Scope:** Full codebase review of `/Users/nick/github/omendb/omendb/src/`
**Test Results:** 273 tests passed

---

## Summary

Comprehensive review of the OmenDB embedded vector database. The codebase is well-structured with solid implementations of HNSW, quantization (SQ8, RaBitQ), and hybrid search. However, several issues were identified ranging from a critical compilation bug in FFI to algorithm correctness issues in graph merging.

| Severity | Count |
| -------- | ----- |
| Critical | 3     |
| High     | 2     |
| Medium   | 3     |
| Low      | 4     |

**Verified:** `cargo check --features ffi` confirms compilation failures for issues #1 and #1a.

---

## Critical Issues (Must Fix)

### 1. FFI: omendb_save takes const pointer but flush requires mutable reference

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Lines:** 474-488
**Confidence:** 95%

The function signature takes `*const OmenDB` but calls `flush()` which requires `&mut self`:

```rust
// Line 474
pub unsafe extern "C" fn omendb_save(db: *const OmenDB) -> i32 {
    // ...
    let Some(db) = db.as_ref() else {  // Line 477 - gives &OmenDB (immutable)
        // ...
    };
    match db.store.flush() {  // Line 482 - flush() requires &mut self!
```

This will fail to compile when the `ffi` feature is enabled. The code compiles in tests because `#[cfg(feature = "ffi")]` gates the module.

**Fix:**

```rust
pub unsafe extern "C" fn omendb_save(db: *mut OmenDB) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.flush() {
        // ...
    }
}
```

---

### 1a. FFI: MetadataFilter::from_json method does not exist

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Line:** 818
**Confidence:** 100% (verified by cargo check)

The `omendb_hybrid_search` function calls `MetadataFilter::from_json()` which doesn't exist:

```rust
// Line 817-818
Ok(v) => match crate::vector::store::MetadataFilter::from_json(&v) {
```

Compiler error:

```
error[E0599]: no variant or associated item named `from_json` found for enum `MetadataFilter` in the current scope
```

**Fix:** Implement `from_json` on `MetadataFilter` enum in `/Users/nick/github/omendb/omendb/src/vector/store/filter.rs`, or use a different parsing approach (e.g., serde deserialize).

---

### 2. IGTM Graph Merge: entry_points use wrong graph IDs

**File:** `/Users/nick/github/omendb/omendb/src/vector/hnsw/merge.rs`
**Lines:** 194-208
**Confidence:** 90%

When merging graphs, the code finds neighbors in the small graph and uses them as entry_points for fast insertion. However, these IDs are from the small graph - not the large graph where vectors were just inserted:

```rust
// Lines 194-199
let small_neighbors = small.get_neighbors_level0(node_id);
let entry_points: Vec<u32> = small_neighbors
    .iter()
    .filter(|&&n| join_set.contains(&n))  // Small graph IDs
    .copied()
    .collect();

// Line 208 - entry_points are small graph IDs, but large graph needs large graph IDs!
large.insert_with_hints(vector, &entry_points, fast_ef)?;
```

The comment on line 206-207 says "These vectors were already inserted, so we can find them in large graph" - but the IDs don't match. When join set vectors are inserted into the large graph, they get NEW IDs assigned by the large graph.

**Fix:** Track mapping from small graph IDs to large graph IDs:

```rust
// Phase 2: Insert join set vectors and track ID mapping
let mut small_to_large: HashMap<u32, u32> = HashMap::new();
for &node_id in &join_set {
    let vector = small.get_vector(node_id).ok_or(...)?;
    let large_id = large.insert(vector)?;  // Returns new ID in large graph
    small_to_large.insert(node_id, large_id);
}

// Phase 3: Map entry points using the ID mapping
let entry_points: Vec<u32> = small_neighbors
    .iter()
    .filter(|&&n| join_set.contains(&n))
    .filter_map(|&n| small_to_large.get(&n).copied())  // Map to large graph IDs
    .collect();
```

---

## High Severity Issues (Should Fix)

### 3. FFI: Filter parameter accepted but ignored in omendb_search

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Lines:** 374-407
**Confidence:** 95%

The `omendb_search` function accepts a `filter_json` parameter but completely ignores it:

```rust
// Lines 375-388
let filter: Option<MetadataFilter> = if filter_json.is_null() {
    None
} else {
    let filter_str = match CStr::from_ptr(filter_json).to_str() { ... };
    // Filter parsing not yet implemented in FFI
    let _ = filter_str;  // Line 386 - IGNORED!
    None
};

// Lines 390-406 - Both branches call knn_search without filter
let search_results = if filter.is_some() {
    // Filtered search not yet exposed via FFI (filter ignored)
    match db.store.knn_search(&query, k) { ... }  // Still uses knn_search!
} else {
    match db.store.knn_search(&query, k) { ... }
};
```

This is misleading to API consumers who expect filtering to work.

**Fix:** Either implement filter parsing (note: `omendb_hybrid_search` at line 809-832 shows how to parse filters correctly) or document that the parameter is not yet implemented:

```rust
if !filter_json.is_null() {
    set_last_error("Filter not yet implemented in omendb_search. Use omendb_hybrid_search.".to_string());
    return -1;
}
```

---

### 4. FFI: Unused config_json parameter in omendb_open

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Lines:** 69-105
**Confidence:** 100%

The `config_json` parameter is declared with leading underscore and never used:

```rust
// Line 81-82
pub unsafe extern "C" fn omendb_open(
    path: *const c_char,
    dimensions: usize,
    _config_json: *const c_char,  // Never used
) -> *mut OmenDB {
```

**Fix:** Either implement config parsing or remove the parameter and update documentation.

---

## Medium Severity Issues (Consider Fixing)

### 5. VectorStore::merge_from rebuilds entire index instead of using IGTM

**File:** `/Users/nick/github/omendb/omendb/src/vector/store/mod.rs`
**Lines:** 1806-1868
**Confidence:** 85%

The `merge_from` method has an IGTM-based `GraphMerger` available in `merge.rs` but instead rebuilds the entire index:

```rust
// Lines 1863-1866
// Always rebuild index after merge to ensure consistency
// (HNSW merge would include conflicting vectors that were skipped above)
self.rebuild_index()?;
```

The comment mentions "conflicting vectors" but the GraphMerger handles this. Rebuilding is O(n log n) vs IGTM's expected 1.3-1.7x speedup.

**Fix:** Use `GraphMerger` from `merge.rs` to merge the HNSW indices directly.

---

### 6. Thread-local FFI error can be clobbered

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Lines:** 44-56
**Confidence:** 100%

The `LAST_ERROR` uses thread-local storage, but if multiple FFI calls occur before reading the error, earlier errors are lost:

```rust
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}
```

This is a common pattern in C FFI but should be documented. Users must call `omendb_last_error()` immediately after a function returns an error code.

**Note:** This is a design limitation, not a bug. Document in header file.

---

### 7. TextIndex changes require explicit commit()

**File:** `/Users/nick/github/omendb/omendb/src/text/mod.rs`
**Lines:** 154-185

After `index_document()`, changes are not visible to searchers until `commit()` is called. While `VectorStore::flush()` calls commit, mid-session queries won't see uncommitted documents.

**Note:** This is documented behavior but could cause confusion. Consider auto-commit option.

---

## Low Severity Issues (Optional Fix)

### 8. FFI version hardcoded

**File:** `/Users/nick/github/omendb/omendb/src/ffi.rs`
**Line:** 518

Version is hardcoded instead of using Cargo.toml:

```rust
static VERSION: &[u8] = b"0.0.21\0";
```

**Fix:**

```rust
// At module level
const VERSION_WITH_NULL: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

pub extern "C" fn omendb_version() -> *const c_char {
    VERSION_WITH_NULL.as_ptr().cast::<c_char>()
}
```

---

### 9. cosine_distance returns 1.0 for zero vectors

**File:** `/Users/nick/github/omendb/omendb/src/distance/ops.rs`
**Lines:** 129-133

When either vector has zero norm, returns 1.0 (maximum distance). This is reasonable but should be documented:

```rust
if norm_a == 0.0 || norm_b == 0.0 {
    return 1.0;  // Maximum distance for degenerate case
}
```

**Note:** Document this behavior in the function's doc comment.

---

### 10. Neighbor code storage silently masks missing data

**File:** `/Users/nick/github/omendb/omendb/src/vector/hnsw/storage.rs`
**Lines:** 700-710

When building interleaved codes, if a neighbor_id points beyond quantized_data length, it silently uses 0 (padding):

```rust
if code_start + sq < quantized_data.len() {
    codes[block_start + sq * FASTSCAN_BATCH_SIZE + n] =
        quantized_data[code_start + sq];
}
// Else: padding with 0 (already initialized)
```

**Note:** This is intentional but could add debug logging for debugging data corruption.

---

### 11. batch_insert progress tracking division

**File:** `/Users/nick/github/omendb/omendb/src/vector/hnsw/index/insert.rs`
**Lines:** 362-366

```rust
let progress_interval = if batch_size >= 1000 {
    batch_size / 10
} else {
    batch_size
};
```

For batch_size < 10, `batch_size / 10 = 0` which would cause modulo by zero if the check was `>= 10`. Current code is safe but the logic is fragile.

---

## Positive Observations

1. **Well-structured HNSW implementation** - The monomorphized distance dispatch and ACORN-1 filtered search are well-implemented.

2. **Comprehensive quantization support** - SQ8 and RaBitQ with proper fallback paths.

3. **Thread-safe storage** - Lock-free reads via ArcSwap for search performance.

4. **Good error handling** - Custom error types with proper propagation.

5. **Platform-aware optimizations** - Prefetch disabled on Apple Silicon (DMP handles it).

6. **MN-RU deletion algorithm** - Proper graph repair after deletions.

---

## Recommendations

1. **Enable FFI feature in CI tests** to catch compilation issues like #1.

2. **Add integration tests for graph merging** to validate ID mapping.

3. **Consider auto-commit mode for TextIndex** to improve usability.

4. **Add version validation** when loading persisted indices.

---

## Files Reviewed

- `src/lib.rs`
- `src/ffi.rs`
- `src/distance/ops.rs`
- `src/compression/scalar.rs`
- `src/compression/rabitq.rs`
- `src/omen/file.rs`
- `src/omen/wal.rs`
- `src/omen/header.rs`
- `src/text/mod.rs`
- `src/vector/store/mod.rs`
- `src/vector/hnsw/storage.rs`
- `src/vector/hnsw/graph_storage.rs`
- `src/vector/hnsw/merge.rs`
- `src/vector/hnsw/index/mod.rs`
- `src/vector/hnsw/index/search.rs`
- `src/vector/hnsw/index/insert.rs`
- `src/vector/hnsw/index/delete.rs`
