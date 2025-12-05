# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmenDB is a fast embedded vector database written in Rust with Python and Node.js bindings. It features HNSW indexing, ACORN-1 filtered search, and RaBitQ compression.

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
- **Release**: Manual workflow - builds all platforms, publishes to crates.io, PyPI, npm
- All registries use OIDC trusted publishing

## Notes

- Requires nightly Rust (`nightly-2025-12-04` pinned in `rust-toolchain.toml`)
- Python 3.9+, Node.js 18+
- Uses `parking_lot`, `rayon`, `multiversion` for runtime SIMD detection
- Slow tests marked with `@pytest.mark.slow` (skipped in CI)

## External Context

Design docs and roadmap: `../cloud/ai/omendb/`
