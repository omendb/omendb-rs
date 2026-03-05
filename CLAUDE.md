# OmenDB

Embedded vector database. Rust core with Python/Node bindings.

## 0.0.x Development Policy

**No backwards compatibility during 0.0.x.** Breaking changes are expected and encouraged.

- **No deprecated aliases** - Remove old code entirely, don't leave deprecated wrappers
- **No compatibility shims** - Users on 0.0.x accept breakage
- **Clean refactors only** - Update all code (tests, examples, bindings) to new APIs
- **Full validation required** - Run all tests, examples, benchmarks before release

## Quick Reference

```bash
# Rust
cargo test --lib
cargo clippy && cargo fmt --check

# Python (from python/)
uv sync && uv run maturin develop --release
uv run pytest tests/ -x --timeout=60
uv run ruff check . && uv run ruff format --check .

# Node (from node/)
bun install && bun run build && bun test
```

## Configuration Defaults

| Parameter       | Default               | Notes                                       |
| --------------- | --------------------- | ------------------------------------------- |
| m               | 16                    | HNSW neighbors per node (industry standard) |
| ef_construction | 100                   | Build quality                               |
| ef_search       | 100                   | Search quality                              |
| quantization    | off                   | `True` or `"sq8"`                           |
| rescore         | true (when quantized) | Rerank with exact L2                        |
| oversample      | 3.0                   | Fetch k×oversample candidates               |

**Quantization API:**

```python
# Enable quantization (SQ8 is default, fastest, best recall)
db = omendb.open("./db", dimensions=768, quantization=True)       # SQ8: 4x, ~99% recall
db = omendb.open("./db", dimensions=768, quantization="sq8")      # Same as True

# Tuning
db = omendb.open("./db", dimensions=768, quantization=True, rescore=False)  # Skip rescore
```

## Architecture

```
src/
├── vector/store/       # VectorStore API (CRUD, search, persistence)
├── vector/hnsw/        # HNSW index (graph, segments, node storage)
├── vector/hnsw_index.rs # High-level HNSW wrapper
├── vector/muvera/      # Multi-vector (ColBERT-style) encoder
├── compression/        # SQ8 quantization
├── distance/           # SIMD distance functions
├── text/               # BM25 hybrid search (tantivy)
└── omen/               # File format, WAL, persistence

python/                 # PyO3 bindings
node/                   # NAPI-RS bindings
omendb-ffi/             # C FFI bindings
```

## Key Modules

| Module                   | Purpose             | Hot Path |
| ------------------------ | ------------------- | -------- |
| `vector/store/mod.rs`    | Main API, batch ops | Yes      |
| `vector/store/search.rs` | Search algorithms   | Yes      |
| `vector/hnsw_index.rs`   | HNSW search wrapper | Yes      |
| `vector/hnsw/index/`     | Graph traversal     | Yes      |
| `distance/ops.rs`        | SIMD distance       | Yes      |

## Performance Notes

**Hot path optimizations applied:**

- `knn_search_ef()` (pub(crate)) avoids Option overhead (~40% faster)
- `search_batch()` pre-computes ef once
- Sequential HNSW insert (parallel degrades recall)

**Benchmarks:**

```bash
cd python && uv run python benchmark.py --quick   # Dev (~15s)
cd python && uv run python benchmark.py           # SIFT-100K (~60s)
```

Expected (SIFT-100K, 128D, M=16, ef=100, M3 Max):

- Build: ~60K vec/s
- Search: ~7,600 QPS single, ~65,000 QPS batch
- Recall@10: ~99.8%

## Testing

```bash
cargo test --lib                              # 525+ Rust tests
cd python && uv run pytest tests/ -x          # 322+ Python tests
cd node && bun test                           # 111+ Node tests
cd python && uv run pytest tests/test_recall.py  # Recall verification
```

**Recall thresholds:** 95%+ (small), 90%+ (medium), 85%+ (large)

## Pre-Release Checklist (MANDATORY)

Before ANY release, complete ALL of these checks:

### 1. Run ALL Test Suites

```bash
# Rust (must pass 525+ tests)
cargo test --lib
cargo clippy --lib -- -D warnings

# Python (must pass 322+ tests)
cd python && uv run pytest tests/ -x

# Node (must pass 111+ tests)
cd node && bun test
```

### 2. Run Performance Benchmarks

```bash
# Standard benchmark (SIFT-100K)
cd python && uv run python benchmark.py

# Full benchmark (multiple dimensions and scales)
cd python && uv run python benchmark.py --full
```

**Performance baselines (SIFT-100K, 128D, M=16, ef=100, M3 Max):**

| Metric    | fp32        | SQ8         |
| --------- | ----------- | ----------- |
| Build     | ~60K vec/s  | ~60K vec/s  |
| Single    | ~7,600 QPS  | ~15,400 QPS |
| Batch     | ~65,000 QPS | ~95,000 QPS |
| Recall@10 | ~99.8%      | ~99.8%      |

### 3. Verify SDK APIs Match

Ensure benchmark.py and README examples use correct method names:

- `search()`, `search_batch()`, `search_text()`, `search_hybrid()`
- NOT `text_search()` or `hybrid_search()`

### 4. Check Version Sync

```bash
./scripts/sync-version.sh --check
```

All version locations must match and be higher than published PyPI version.

## Release Process

```bash
./scripts/sync-version.sh 0.0.X   # Bump all version locations
git add -A && git commit -m "chore: Bump to 0.0.X"
git tag v0.0.X -m "v0.0.X"
git push && git push --tags
# Then manually trigger the release workflow — pushing tags does NOT auto-trigger it
gh workflow run release.yml
```

**IMPORTANT:** `release.yml` is `workflow_dispatch` only — it NEVER triggers automatically
on tag push. Pushing tags is safe. Always trigger the workflow explicitly after pushing.
Do NOT suggest "push tags to trigger release" — this is wrong.

## CI

| Workflow      | Trigger     | What                                        |
| ------------- | ----------- | ------------------------------------------- |
| `ci.yml`      | Push/PR     | fmt, clippy, test (Rust + Python + Node)    |
| `release.yml` | Manual only | Build wheels, publish to PyPI/crates.io/npm |

## Dependencies

- **tantivy**: BM25 text search

## Common Tasks

**Add a new Python API method:**

1. Add Rust method in `src/vector/store/mod.rs`
2. Expose in `python/src/`
3. Add test in `python/tests/`

**Optimize search path:**

1. Profile: `cargo build --release --example profile_search && samply record ./target/release/examples/profile_search`
2. Check for Option/closure overhead in hot loops
3. Pre-compute values outside parallel iterators

**Debug recall issues:**

1. Run `pytest tests/test_recall.py -v`
2. Check if batch_insert was used (should be sequential for new data)
3. Increase ef_search for better recall vs speed tradeoff
