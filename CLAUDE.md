# OmenDB

Embedded vector database. Rust core with Python/Node bindings.

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
npm install && npm run build && npm test
```

## Configuration Defaults

| Parameter       | Default               | Notes                                       |
| --------------- | --------------------- | ------------------------------------------- |
| m               | 16                    | HNSW neighbors per node (industry standard) |
| ef_construction | 100                   | Build quality                               |
| ef_search       | 100                   | Search quality                              |
| quantization    | off                   | RaBitQ bits: 2, 4, or 8                     |
| rescore         | true (when quantized) | Rerank with exact L2                        |
| oversample      | 3.0                   | Fetch k×oversample candidates               |

**Quantization API:**

```python
db = omendb.open("./db", dimensions=128, quantization=4)  # rescore=True by default
db = omendb.open("./db", dimensions=128, quantization=4, rescore=False)  # max speed
```

**Insert performance** (10K vectors, 128D):

- Sequential (default): 1,430 vec/s, 100% recall
- With quantization: 3,400 vec/s, 95.7% recall

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
- `batch_search_parallel()` pre-computes ef once
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

## Release Process

See `RELEASING.md`. Quick version:

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
