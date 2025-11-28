# Data Loss Test Plan

Comprehensive test plan to verify omendb has no data loss bugs before publishing to crates.io.

## Critical Areas

### 1. HNSW Incremental Insert Bug (Fixed Nov 2025)

**Bug**: Entry point updated before graph edges were constructed, causing vectors inserted after first search to be unreachable.

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| INC-1 | Insert 100 vectors, search, insert 100 more, verify all 200 searchable | ✅ `test_incremental_set_batch` |
| INC-2 | Interleave inserts and searches (insert 10, search, repeat 10x) | ✅ `test_interleaved_insert_search` |
| INC-3 | Insert batch, search, single insert, search, verify new vector reachable | ✅ `test_batch_then_single_insert` |
| INC-4 | Empty index -> insert -> search -> insert -> search cycle | ✅ `test_insert_search_cycle_from_empty` |

### 2. Persistence (Save/Load)

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| PER-1 | Save index, load, verify all vectors present | ✅ `test_save_load_roundtrip` |
| PER-2 | Save index, load, verify search returns same results | ✅ `test_persistent_search` |
| PER-3 | Save with quantization, load, verify recall | ✅ `test_quantization_persistence` |
| PER-4 | Save after inserts, load, insert more, verify all searchable | ✅ `test_persistence_across_reopen` |
| PER-5 | Reopen database multiple times, verify data integrity | ✅ `test_persistence_across_reopen` |
| PER-6 | Large index (100K+ vectors) save/load roundtrip | P2 - Future |

### 3. Disk Storage

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| DSK-1 | Memory -> DiskStorage serialization roundtrip | ✅ `test_disk_storage_persistence` |
| DSK-2 | WritableDiskStorage incremental writes | ✅ `test_writable_disk_storage_basic` |
| DSK-3 | CachedStorage consistency with DiskStorage | ✅ `test_memory_disk_cached_consistency` |
| DSK-4 | Large graph (10K+ nodes) disk serialization | ✅ `test_large_graph_disk_roundtrip` |
| DSK-5 | Disk-backed index search accuracy | ✅ `test_disk_backed_workflow_end_to_end` |

### 4. Deletion

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| DEL-1 | Delete vector, verify not in search results | ✅ `test_persistent_delete` |
| DEL-2 | Delete, save, load, verify still deleted | ✅ `test_persistent_delete` (includes reopen) |
| DEL-3 | Delete vector, insert new, verify graph integrity | P2 - Future |
| DEL-4 | Delete all vectors, verify empty state | P2 - Future |
| DEL-5 | Delete entry point, verify new entry point selected | P2 - Future |

### 5. Quantization (RaBitQ)

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| QNT-1 | 4-bit quantization recall > 95% | ✅ `test_quantization_search_accuracy` |
| QNT-2 | 2-bit quantization recall > 80% | P2 - Future |
| QNT-3 | FHT-enabled quantization correctness | ✅ `test_fht_*` (multiple FHT tests) |
| QNT-4 | Quantization + persistence roundtrip | ✅ `test_quantization_persistence` |
| QNT-5 | Original vectors recoverable when keep_original=true | P2 - Future |

### 6. Filtered Search (ACORN-1)

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| FLT-1 | Basic metadata filter (equals) | ✅ `test_metadata_filter_eq` |
| FLT-2 | Range filter (greater than, less than) | ✅ `test_metadata_filter_gte` |
| FLT-3 | Complex filter (AND, OR) | ✅ `test_metadata_filter_and` |
| FLT-4 | High selectivity filter (0.1%) | P2 - Future |
| FLT-5 | Filter returns no results | P2 - Future |
| FLT-6 | Filter with persistence | ✅ `test_persistence_with_metadata` |

### 7. Edge Cases

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| EDG-1 | Single vector insert/search | ✅ `test_hnsw_index_insert_single`, `test_hnsw_index_search_single` |
| EDG-2 | Identical vectors (duplicates) | P2 - Future |
| EDG-3 | Zero vector (all zeros) | ✅ `test_zero_vectors`, `test_distance_zero_vectors` |
| EDG-4 | Very high dimensional (4096D) | P2 - Future |
| EDG-5 | Empty search on empty index | ✅ `test_hnsw_index_search_empty`, `test_empty_index_serialization` |
| EDG-6 | k > num_vectors search | P2 - Future |

### 8. Concurrency

**Test Cases**:
| ID | Test | Status |
|----|------|--------|
| CON-1 | Parallel inserts via batch_insert | ✅ Via rayon |
| CON-2 | Concurrent reads during build | P2 - Future |
| CON-3 | Thread-local query buffers isolation | ✅ `test_thread_local_isolation` |

---

## Test Priority

### P0 - Must Pass Before Publish ✅ ALL PASS
- INC-1, INC-2, INC-3, INC-4 (Incremental insert - recently fixed bug)
- PER-1, PER-2, PER-3, PER-4, PER-5 (Core persistence)
- DSK-1 through DSK-5 (Disk storage)

### P1 - Should Pass ✅ ALL PASS
- DEL-1, DEL-2 (Deletion persistence)
- QNT-1, QNT-3, QNT-4 (Quantization)
- FLT-1, FLT-2, FLT-3, FLT-6 (Filtered search)
- EDG-1, EDG-3, EDG-5 (Edge cases)
- CON-1, CON-3 (Concurrency)

### P2 - Nice to Have (Future)
- Large scale tests (100K+)
- DEL-3, DEL-4, DEL-5 (Advanced deletion)
- QNT-2, QNT-5 (2-bit, keep_original)
- FLT-4, FLT-5 (Edge filter cases)
- EDG-2, EDG-4, EDG-6 (Advanced edge cases)
- CON-2 (Concurrent reads during build)

---

## Running Tests

```bash
# Run all tests
cargo test --lib

# Run specific category
cargo test incremental
cargo test persist
cargo test disk
cargo test quantization

# Run with logging
RUST_LOG=debug cargo test test_name -- --nocapture
```

---

## Test Implementation Status

| Category | Existing | Needed (P2) | Coverage |
|----------|----------|-------------|----------|
| Incremental Insert | 4 | 0 | 100% |
| Persistence | 5 | 1 | 83% |
| Disk Storage | 5 | 0 | 100% |
| Deletion | 2 | 3 | 40% |
| Quantization | 3 | 2 | 60% |
| Filtered Search | 4 | 2 | 67% |
| Edge Cases | 3 | 3 | 50% |
| Concurrency | 2 | 1 | 67% |

**Overall**: ~28 existing tests, ~12 P2 (future) = ~70% coverage on data loss scenarios

**P0 (Must Pass) Status**: ✅ All critical tests pass
**P1 (Should Pass) Status**: ✅ All pass

---

## Verified: November 28, 2025

- omendb: 256 tests pass
- cloud lib: 166 tests pass
- cloud integration: All pass

**Ready for PyPI 0.0.1-alpha release**

---

*Created: 2025-11-27*
*Last Updated: 2025-11-28*
