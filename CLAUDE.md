# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmenDB is a fast embedded vector database written in Rust with Python bindings. It features HNSW indexing, ACORN-1 filtered search, and RaBitQ compression.

## Build Commands

### Rust

```bash
# Build
cargo build --release

# Test (library tests only)
cargo test --lib

# Lint
cargo fmt && cargo clippy

# Build with FFI bindings
cargo build --release --features ffi
```

### Python Bindings

```bash
cd python
uv sync

# Build native module (development)
uv run maturin develop --release

# Run tests
uv run pytest tests/

# Run single test
uv run pytest tests/test_basic.py::test_set_and_search -v
```

## Architecture

### Crate Structure

- **omendb** (root) - Main Rust library with vector store and HNSW implementation
- **seerdb-vector** - Vector-optimized storage layer built on seerdb LSM-tree
- **python/** - PyO3 bindings (separate workspace, uses maturin)

### Core Modules (`src/vector/`)

| Module | Purpose |
|--------|---------|
| `store.rs` | `VectorStore` - main API with HNSW + filtered search |
| `hnsw_index.rs` | HNSW index wrapper |
| `custom_hnsw/` | Custom HNSW implementation with SIMD distance |
| `extended_rabitq.rs` | RaBitQ 2/4/8-bit quantization |
| `storage.rs` | seerdb persistent storage backend |
| `types.rs` | `Vector` type and metadata |

### Storage Tiers (seerdb-vector)

- `edge_storage.rs` - HNSW graph edge storage
- `node_metadata.rs` - Per-node metadata
- `compression/rabitq.rs` - Vector compression
- `simd_distance.rs` - SIMD-accelerated distance calculations

### Python Bindings

- `python/src/lib.rs` - PyO3 module exposing `VectorDatabase`
- `python/omendb/__init__.py` - Python entry point
- `python/omendb/langchain.py` - LangChain integration
- `python/omendb/llamaindex.py` - LlamaIndex integration

## Key Types

```rust
// Main entry points
use omendb::{VectorStore, Vector, MetadataFilter};
use seerdb_vector::{SeerdbVectorConfig, StorageTier, CompressionTier, DistanceMetric};
```

## Feature Flags

- `ffi` - Enable C FFI bindings (`src/ffi.rs`)
- `profile_search` - Enable search profiling

## Notes

- Requires nightly Rust (`#![feature(portable_simd)]`)
- Python requires Python 3.9+
- Uses `parking_lot` for locking, `rayon` for parallelism
