# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmenDB is a fast embedded vector database written in Rust with Python bindings. It features HNSW indexing, ACORN-1 filtered search, and RaBitQ compression.

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
```

## Architecture

### Core Modules

| Path | Purpose |
|------|---------|
| `src/vector/store.rs` | `VectorStore` - main API |
| `src/vector/hnsw/` | HNSW index with SIMD distance |
| `src/compression/rabitq.rs` | RaBitQ 2/4/8-bit quantization |
| `src/simd/distance.rs` | SIMD-accelerated distance |
| `src/storage/` | seerdb persistent storage |
| `python/` | PyO3 bindings (maturin) |

### Key Types

```rust
use omendb::{VectorStore, Vector, MetadataFilter};
use omendb::{SeerdbVectorConfig, StorageTier, CompressionTier, DistanceMetric};
```

## Notes

- Requires nightly Rust (`#![feature(portable_simd)]`)
- Python 3.9+
- Uses `parking_lot`, `rayon`, `seerdb`
