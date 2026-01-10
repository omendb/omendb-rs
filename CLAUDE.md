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
| quantization    | off                   | `True`/`"sq8"` or `"rabitq"`                |
| rescore         | true (when quantized) | Rerank with exact L2                        |
| oversample      | 3.0                   | Fetch k×oversample candidates               |

**Quantization API:**

```python
# Enable quantization (SQ8 is default, fastest, best recall)
db = omendb.open("./db", dimensions=768, quantization=True)       # SQ8: 4x, ~99% recall
db = omendb.open("./db", dimensions=768, quantization="sq8")      # Same as True
db = omendb.open("./db", dimensions=768, quantization="rabitq")   # RaBitQ: 8x, ~98% recall

# Tuning
db = omendb.open("./db", dimensions=768, quantization=True, rescore=False)  # Skip rescore
```

| Mode       | Compression | Recall | Speed  | Use Case                    |
| ---------- | ----------- | ------ | ------ | --------------------------- |
| `True`     | 4x          | ~99%   | Fast   | Default, most users         |
| `"sq8"`    | 4x          | ~99%   | Fast   | Explicit scalar             |
| `"rabitq"` | 8x          | ~98%   | Slower | Large datasets needing more |

**Why only two modes?**

- `rabitq-8` (8-bit): Same 4x as SQ8 but slower → use SQ8 instead
- `rabitq-2` (2-bit): 93% recall too low → edge case not worth complexity

## Architecture

```
src/
├── vector/store/       # VectorStore API
├── vector/hnsw/        # HNSW index
├── vector/hnsw_index.rs # High-level HNSW wrapper
├── text/               # BM25 hybrid search
└── storage/            # SeerDB persistence

omendb-core/            # Extracted algorithms (published separately)
├── src/hnsw/           # Core HNSW implementation
├── src/compression/    # RaBitQ quantization
├── src/distance/       # SIMD distance functions
└── src/sampling/       # Sampling utilities

python/                 # PyO3 bindings
node/                   # NAPI-RS bindings
```

## Key Modules

| Module                        | Purpose             | Hot Path |
| ----------------------------- | ------------------- | -------- |
| `vector/store/mod.rs`         | Main API, batch ops | Yes      |
| `vector/hnsw_index.rs`        | HNSW search wrapper | Yes      |
| `omendb-core/src/distance/`   | SIMD distance       | Yes      |
| `omendb-core/src/hnsw/index/` | Graph traversal     | Yes      |

## Performance Notes

**Hot path optimizations applied:**

- `knn_search_ef()` avoids Option overhead (~40% faster)
- `search_batch()` pre-computes ef once
- Sequential HNSW insert (parallel degrades recall)

**Benchmarks:**

```bash
cd python && uv run python ../benchmarks/run.py --quick   # Dev (~15s)
cd python && uv run python ../benchmarks/run.py           # Full (~60s)
```

Expected (10K vectors, M3 Max):

- 128D: ~7,700 QPS single, ~50,000 QPS batch
- 768D: ~2,500 QPS single, ~12,600 QPS batch

## Testing

```bash
cargo test --lib                              # 248 Rust tests
cd python && uv run pytest tests/ -x          # 214 Python tests
cd python && uv run pytest tests/test_recall.py  # Recall verification
```

**Recall thresholds:** 95%+ (small), 90%+ (medium), 85%+ (large)

## Pre-Release Checklist (MANDATORY)

Before ANY release, complete ALL of these checks:

### 1. Run ALL Test Suites

```bash
# Rust (must pass 279+ tests)
cargo test --lib
cargo clippy --lib -- -D warnings

# Python (must pass 253+ tests)
cd python && uv run pytest tests/ -x

# Node (must pass 60+ tests)
cd node && bun test
```

### 2. Run Performance Benchmarks

```bash
# Standard benchmark (10k/128D) - compare against history.json
cd python && uv run python benchmark.py

# Full benchmark (multiple dimensions and scales)
cd python && uv run python benchmark.py --full

# Check for regressions against previous versions in python/benchmarks/history.json
```

**Performance baselines (10k vectors, M3 Max):**

| Dimension | Single QPS | Batch QPS | Recall |
| --------- | ---------- | --------- | ------ |
| 128D      | >9,000     | >80,000   | >89%   |
| 768D      | >3,000     | >15,000   | >84%   |

### 3. Verify SDK APIs Match

Ensure benchmark.py and README examples use correct method names:

- `search()`, `search_batch()`, `search_text()`, `search_hybrid()`
- NOT `text_search()` or `hybrid_search()`

### 4. Check Version Sync

```bash
./scripts/sync-version.sh --check
```

All 8 locations must match and be higher than published PyPI version.

## Release Process

```bash
./scripts/sync-version.sh 0.0.10   # Bump all 9 version locations
git add -A && git commit -m "chore: Bump to 0.0.10"
git push
gh workflow run release.yml
```

## CI

| Workflow      | Trigger | What                                        |
| ------------- | ------- | ------------------------------------------- |
| `ci.yml`      | Push/PR | fmt, clippy, test (Rust + Python + Node)    |
| `release.yml` | Manual  | Build wheels, publish to PyPI/crates.io/npm |

## Dependencies

- **seerdb**: Storage layer (separate crate)
- **omendb-core**: Algorithms (workspace member, published first)
- **tantivy**: BM25 text search

## Common Tasks

**Add a new Python API method:**

1. Add Rust method in `src/vector/store/mod.rs`
2. Expose in `python/src/lib.rs`
3. Add test in `python/tests/`

**Optimize search path:**

1. Profile: `cargo build --release --example profile_search && samply record ./target/release/examples/profile_search`
2. Check for Option/closure overhead in hot loops
3. Pre-compute values outside parallel iterators

**Debug recall issues:**

1. Run `pytest tests/test_recall.py -v`
2. Check if batch_insert was used (should be sequential for new data)
3. Increase ef_search for better recall vs speed tradeoff
