# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmenDB is a fast embedded vector database written in Rust with Python and Node.js bindings. Features HNSW indexing, ACORN-1 filtered search, RaBitQ compression, and hybrid search (vector + BM25).

## Build Commands

### Rust

```bash
cargo build --release        # Build
cargo test --lib             # Test library
cargo fmt && cargo clippy    # Lint
```

### Python Bindings

```bash
cd python
uv sync
uv run maturin develop --release   # Build
uv run pytest tests/               # Test
uv run ruff check .                # Lint
uv run ruff format --check .       # Format check
```

### Node.js Bindings

```bash
cd node
npm install
npm run build    # Build
npm test         # Test
```

## Architecture

### Core Modules

| Path                        | Purpose                       |
| --------------------------- | ----------------------------- |
| `src/vector/store.rs`       | `VectorStore` - main API      |
| `src/vector/hnsw/`          | HNSW index with SIMD distance |
| `src/compression/rabitq.rs` | RaBitQ 2/4/8-bit quantization |
| `src/distance/distance.rs`  | SIMD-accelerated distance     |
| `src/storage/`              | Persistent storage (SeerDB)   |
| `python/`                   | PyO3 bindings (maturin)       |
| `node/`                     | NAPI-RS bindings              |

### Key Types

```rust
use omendb::{VectorStore, Vector, MetadataFilter};
use omendb::{VectorStoreOptions, StorageConfig, StorageTier, CompressionTier, DistanceMetric};
```

## CI/CD

- **CI**: Runs on push/PR - checks Rust (fmt, clippy, test), Python (ruff, pytest), Node (build, test)
- **Release**: Use `/omendb-release` command or see `RELEASING.md`. Quick: `./scripts/bump-version.sh && git commit && git push && gh workflow run Release`

## Testing & Benchmarks

### Running Tests

```bash
cargo test --lib                          # Rust unit tests
uv run pytest python/tests/               # Python tests (300+)
uv run pytest python/tests/ -m "not slow" # Skip slow tests
```

### Benchmarks

```bash
# Python benchmark (quick: 10K vectors, full: 10K-100K)
uv run python python/benchmark.py         # Quick (~30s)
uv run python python/benchmark.py --full  # Full suite (~5min)

# Rust criterion benchmarks
cargo bench --bench search_bench          # Search QPS
cargo bench --bench distance_bench        # Distance functions
```

### Expected Performance (10K vectors, 128D, M3 Max)

| Metric        | Target       |
| ------------- | ------------ |
| Single Search | ~5,400 QPS   |
| Batch Search  | ~52,000 QPS  |
| Build         | ~1,500 vec/s |

### Regression Detection

Performance regressions typically indicate:

1. **Search degradation**: Check HNSW graph quality (batch_insert vs sequential insert)
2. **Build slowdown**: Check if batch operations are being used
3. **Memory regression**: Profile with `cargo flamegraph`

## Notes

- Requires nightly Rust (`nightly-2025-12-04` pinned in `rust-toolchain.toml`)
- Python 3.9+, Node.js 18+
- Uses `parking_lot`, `rayon`, `multiversion` for runtime SIMD detection
- Slow tests marked with `@pytest.mark.slow` (skipped in CI)

## External Context

Design docs and roadmap: `../cloud/ai/omendb/`
