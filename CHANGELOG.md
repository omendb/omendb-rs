# Changelog

## v0.0.32

### Durability and Recovery

- Fix manifest-first checkpoint recovery so committed writes are not lost on reopen
- Recover legitimate zero vectors from slim snapshots instead of dropping them during reload
- Fall back to full checkpoints whenever sparse, edge, or multi-vector state is live so fast checkpoints do not discard non-dense data
- Rebuild WAL counters from the WAL body on reopen and roll back the in-memory truncation epoch if `.wal.meta` persistence fails
- Skip unknown future WAL entry types while scanning timestamps so newer WAL formats do not hard-fail older readers immediately

### Search and Indexing

- Reject invalid dense search requests earlier: `k=0`, non-finite query values, and zero-norm cosine queries now fail explicitly
- Add opt-in SQ8 construction mode for HNSW builds
- Align public multi-vector docs and types, including `poolFactor`

### Release and CI

- Refresh Rust, Python, and Node dependency locks plus fuzz targets
- Tighten benchmark/CI trust checks and align published benchmark docs with Linux as the authoritative baseline
- Keep `python/benchmark.py` compatible with Python 3.9 and clean up Python release packaging
- Harden crash-recovery CI by using spawned child processes for kill/reopen tests and stabilize the slow Node multi-vector lane
- Upgrade GitHub setup actions to current majors and use this changelog as the release-note source

## v0.0.31

### Features

- Add EdgeStore typed directed graph primitives for Rust, Python, and Node
- Add shortest path, subgraph, batch edge operations, and degree queries
- Expose `with_max_tokens` and move token limits into `MultiVectorConfig`

### Correctness and Robustness

- Fix EdgeStore WAL replay ordering so edge deletes are applied before inserts
- Deduplicate self-edges correctly and tighten EdgeStore error propagation
- Make WAL parsing more forward-compatible and clarify metadata/version constants
- Apply multi-vector token limits after pooling instead of before
- Improve EdgeStore coverage with WAL edge-case tests, property tests, and stress tests

## v0.0.27

### Bug Fixes

- **Metric handling**: Thread distance metric through brute-force search paths (cosine/IP now correct on small collections)
- **Metric persistence**: Metric survives flush+reopen via `set_metric()` on OmenFile
- **HNSW params on reopen**: m=8 no longer clamped to m=16 on reopen (fix `.max(DEFAULT)` logic)
- **update() re-indexing**: Vector updates now re-index in HNSW (delegates to `set()`)
- **FFI validation**: Non-numeric vector elements return explicit error instead of silent drop
- **Deserialization safety**: Cap allocation sizes to prevent corrupt files from exhausting memory
- **Runtime bounds checks**: Promote `debug_assert` to `assert` in node_ptr/node_ptr_mut
- **Path traversal**: `delete_collection()` validates names (alphanumeric + underscore)
- **Dimensions drift**: Reads actual dimensions from store on reopen, not user arg
- **Node.js scores**: Metric-aware score normalization (L2, cosine, inner product)

### Documentation

- Fix Node.js README: correct install command (`@omendb/omendb`), fix API docs
- Remove RaBitQ references from README, CLAUDE.md, examples, and benchmarks
- Remove internal docs from public repo

### API Changes

- Rename `text_search` → `search_text`, `hybrid_search` → `search_hybrid` (verb-first consistency)
- Remove `get_ef_search()` — use `ef_search` property instead (Python: `db.ef_search`, Node: `db.efSearch`)
- Unify distance metric enums: single `Metric` enum replaces `DistanceFunction` + `DistanceMetric`
- `set()` takes `&str` instead of `String` for document IDs
- `set_batch()` accepts generic `Into<String>` IDs (works with both `String` and `&str`)
- `delete_batch()` accepts `&[impl AsRef<str>]` (works with both `&[String]` and `&[&str]`)
- Rename `search_batch_with_metadata` → `search_batch`
- Remove `SearchParams` type alias (use `SearchOptions` directly)
- Remove `MuveraConfig` type alias (use `MultiVectorConfig` directly)
- Tighten visibility: `Record`, `RecordStore`, helpers, internal accessors now `pub(crate)`
- Remove dead code: `storage()`, `index_to_id_mapping()`, `id_to_index_mapping()`, unused `RecordStore` methods

### Performance

- Precompute query norm in parallel HNSW builder for cosine metric (eliminates redundant sqrt per distance call)

### FFI

- Add 12 new FFI functions: update, delete_by_filter, compact, optimize, exists, stats, enable_text_search, has_text_search, set_with_text, text_search, hybrid_search, flush (21 total)
- Regenerate C header (`include/omendb.h`) with all functions and correct field names
- Fix late null-pointer checks — validate all output pointers before work
- Use `store.dimensions()` in stats instead of cached copy

### Robustness

- Use `handle_alloc_error` instead of `expect` on allocation failure in node_storage
- Guard u64→u32 config casts with `try_from` fallbacks in file.rs
- Return error from SQ8 params invariant instead of panicking
- Replace `MuveraConfigTuple` (6-element tuple) with `PersistedMuveraConfig` named struct

### Cleanup

- Remove dead code: `SearchConfig`, duplicate `get_by_internal_index`, `_bench_*` functions
- Remove competitor packages from dependencies (chromadb, hnswlib, lancedb, qdrant-client)

## v0.0.26

### Performance

- Enable visited list prefetching in search

### Refactoring

- Move WAL entry parsing to wal.rs
- Split storage.rs into modular storage/ directory

### Bug Fixes

- Address code review findings in update() and set()

## v0.0.25

### Features

- Unified API for single and multi-vector stores
- Multi-vector (MUVERA/ColBERT) support: Python and Node.js bindings
- Token pooling for multi-vector storage reduction (k-means clustering)
- ACORN-1 filtered search in SegmentManager
- Background segment merging with IGTM algorithm
- Segment persistence with save/load for SegmentManager

### Performance

- Parallel HNSW construction (6.4x speedup)
- Thread-local VisitedList for parallel build
- Deferred pruning for 2.1x build throughput
- Replace naive distance with SIMD implementations
- MUVERA: auto-boost ef_search for FDE queries, d_proj dimension projection

### Refactoring

- Extract search.rs, helpers.rs, text_search.rs, multivec_ops.rs, persistence.rs from VectorStore
- Split segment_manager.rs and node_storage.rs into submodules
- Remove dual storage paths

## v0.0.24

### Features

- Node.js quickstart example and verify script
- Python/Node tests added to release workflow

### Bug Fixes

- Release workflow hardening (rerunnable publishes, skip already-published)
- Relax CI latency thresholds

### Documentation

- Add performance measurement dates
- Fix quickstart output

## v0.0.23

### BREAKING CHANGES

**Persistence format upgraded to v2 (postcard)**

- Existing `.omen` files from v0.0.21 or earlier will fail to load
- Error: `"Unsupported version: 1 (expected 2)"`
- **Migration**: Re-create databases by reinserting vectors
- Reason: bincode → postcard for better maintenance and smaller files

### Bug Fixes

- **ACORN-1 filtered search**: Fix sparse filter searches (<10% selectivity)
- **Binary quantization**: Use hamming distance for graph traversal
- **FFI**: Fix omendb_save mutability, filter parsing, config parsing
- **WAL**: Add 100MB size validation, skip corrupted entries on recovery
- **Metadata**: Fix f64::MIN→NEG_INFINITY, UTF-8 error handling
- **Persistence**: Add file locking (fs2), atomic checkpoint via temp-file-rename
- **Security**: Update lru to 0.16 for RUSTSEC-2026-0002

### Performance

- ACORN-1 filtered search: 6-13% faster via zero-copy neighbor access
- Early exit once M matching neighbors found

### Refactoring

- Simplify file.rs, store/mod.rs (~130 lines removed)
- Extract HNSW validation and construction helpers

### Node.js Bindings

- Add `close()` method for explicit file lock release
- Error on non-object metadata with document field
- Fix cache invalidation in update()

### Python Bindings

- Add `py.detach()` for GIL release in set_vectors, items, get_batch
- Remove unnecessary vector clones
