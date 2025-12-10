# omendb-core

Core algorithms for [OmenDB](https://github.com/omendb/omendb):

- **HNSW** - Hierarchical Navigable Small World graphs for approximate nearest neighbor search
- **RaBitQ** - Randomized Bit Quantization for memory-efficient vector compression
- **SIMD distance** - Optimized distance functions with runtime CPU detection

## Usage

This crate is primarily used internally by `omendb`. For most users, use the main `omendb` crate instead.

```rust
use omendb_core::hnsw::HNSWIndex;
use omendb_core::compression::RaBitQ;
use omendb_core::distance;
```

## License

Apache-2.0
