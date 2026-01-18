# OmenDB Embedded Review Checklist

## Prompt

You are reviewing the OmenDB embedded vector database codebase (25K lines Rust).

**Your task:** Systematically review every file listed below and report ALL issues you find. Be thorough - we recently found several broken examples and stale references that slipped through previous reviews.

**IMPORTANT: Report only, do not fix.** Do not modify any files. Output a structured list of findings with file paths and line numbers. A Claude agent will review your findings and make fixes.

**Search the entire codebase for these removed features** (grep/search all .rs files):

- `rabitq`, `RaBitQ`
- `search_asymmetric`
- `with_ef_search`
- `neighbor_codes`, `NeighborCodeStorage`
- `FastScan`
- `build_adc_table`
- `asymmetric_l2_squared`

**For each file**, check:

1. Do doc comments match the actual code behavior?
2. Are there unused functions, imports, or fields?
3. Are there references to features that don't exist?
4. Are there logic errors or bugs?
5. Are variable/function names clear?

**Output format** - For each issue:

```
FILE: path/to/file.rs:LINE
TYPE: [stale_ref|dead_code|doc_mismatch|bug|naming|other]
ISSUE: Description
SUGGESTED_FIX: What should change (do not apply)
```

Start by grepping for the removed features list, then review files by priority (P0 first). Output all findings to stdout - do not edit any files.

---

## Reference

Review the entire codebase for issues. Focus on finding:

1. **Dead code** - unused functions, structs, fields, imports
2. **Stale references** - mentions of removed features (see list below)
3. **Doc/comment accuracy** - do comments match the code?
4. **API consistency** - do signatures match documentation?
5. **Logic errors** - bugs, off-by-one, incorrect calculations
6. **Naming issues** - unclear or misleading names

## Removed Features to Search For

These were removed but references may remain:

```
rabitq
RaBitQ
search_asymmetric
with_ef_search
neighbor_codes
NeighborCodeStorage
FastScan
build_adc_table
asymmetric_l2_squared
```

## Files by Priority

### P0 - Public API (check thoroughly)

| File                      | Lines | Purpose              |
| ------------------------- | ----- | -------------------- |
| `src/vector/store/mod.rs` | 2,477 | Main VectorStore API |
| `python/src/lib.rs`       | 2,465 | Python bindings      |
| `node/src/lib.rs`         | 1,359 | Node.js bindings     |

### P1 - Core Implementation

| File                               | Lines | Purpose                     |
| ---------------------------------- | ----- | --------------------------- |
| `src/vector/hnsw/storage.rs`       | 2,388 | Vector and neighbor storage |
| `src/omen/file.rs`                 | 1,283 | Persistence file format     |
| `src/vector/hnsw_index.rs`         | 940   | Thread-safe HNSW wrapper    |
| `src/vector/hnsw/index/search.rs`  | 862   | Search implementation       |
| `src/vector/store/record_store.rs` | 675   | Record/slot management      |
| `src/compression/scalar.rs`        | 610   | SQ8 quantization            |
| `src/vector/hnsw/index/insert.rs`  | 602   | Insert implementation       |
| `src/vector/hnsw/index/delete.rs`  | 562   | Delete implementation       |
| `src/vector/hnsw/types.rs`         | 493   | HNSW types and params       |
| `src/distance/ops.rs`              | 473   | SIMD distance functions     |
| `src/vector/hnsw/index/mod.rs`     | 362   | HNSW index core             |
| `src/vector/hnsw/graph_storage.rs` | 343   | Graph neighbor storage      |

### P2 - Supporting Systems

| File                             | Lines | Purpose                 |
| -------------------------------- | ----- | ----------------------- |
| `src/vector/store/tests.rs`      | 1,870 | VectorStore tests       |
| `src/vector/hnsw/index/tests.rs` | 1,569 | HNSW tests              |
| `src/omen/metadata.rs`           | 831   | Metadata storage        |
| `src/omen/wal.rs`                | 540   | Write-ahead log         |
| `src/text/tests.rs`              | 496   | Text search tests       |
| `src/vector/hnsw/merge.rs`       | 445   | Index merging           |
| `src/vector/store/filter.rs`     | 403   | Metadata filter parsing |
| `src/text/mod.rs`                | 390   | BM25 text search        |

### P3 - Smaller Files

| File                                   | Lines | Purpose              |
| -------------------------------------- | ----- | -------------------- |
| `src/vector/hnsw/query_buffers.rs`     | 309   | Thread-local buffers |
| `src/vector/store/thread_safe.rs`      | 291   | Thread-safe wrapper  |
| `src/vector/hnsw/index/persistence.rs` | 277   | HNSW serialization   |
| `src/omen/header.rs`                   | 268   | File header format   |
| `src/vector/store/options.rs`          | 251   | VectorStoreOptions   |
| `src/omen/manifest.rs`                 | 237   | Manifest file        |
| `src/lib.rs`                           | ~200  | Public exports       |
| `src/vector/mod.rs`                    | ~150  | Module exports       |
| `src/vector/hnsw/mod.rs`               | ~100  | HNSW exports         |
| `src/vector/hnsw/prefetch.rs`          | 40    | Prefetch config      |
| `src/vector/hnsw/error.rs`             | ~50   | Error types          |

### Examples

| File                          | Purpose            |
| ----------------------------- | ------------------ |
| `examples/basic.rs`           | Basic usage        |
| `examples/filtered_search.rs` | Metadata filtering |
| `examples/persistence.rs`     | Save/load          |
| `examples/bench_*.rs`         | Benchmarks         |

### Documentation

| File               | Purpose                                 |
| ------------------ | --------------------------------------- |
| `README.md`        | Main docs (symlink to python/README.md) |
| `python/README.md` | Python API docs                         |
| `node/README.md`   | Node.js API docs                        |

## Known Issues Already Fixed

- `examples/filtered_search.rs` - was using tuple destructuring on struct
- `examples/bench_sq8_scaling.rs` - referenced rabitq
- `examples/profile_rabitq.rs` - deleted (used non-existent API)
- `examples/bench_adc_x86.rs` - deleted (used non-existent API)
- `examples/profile_sq8.rs` - deleted (used removed search_asymmetric)
- `src/omen/file.rs` - doc comments mentioned rabitq modes
- `src/vector/hnsw/prefetch.rs` - had dead code cache_line_size()
- `python/README.md` - mentioned rabitq quantization option
- `node/README.md` - was missing most API methods

## Output Format

For each issue found, report (do not fix, only report):

```
FILE: path/to/file.rs
LINE: 123
TYPE: [dead_code|stale_ref|doc_mismatch|bug|naming]
DESCRIPTION: Brief description
SUGGESTED_FIX: What should change (do not apply)
```
