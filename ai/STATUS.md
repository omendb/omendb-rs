# OmenDB Status

**Updated**: 2025-12-04

## Current State

OmenDB is a fast embedded vector database in Rust with Python and Node.js bindings.

### Recent Changes (Uncommitted)

1. **VByte Compression for Disk Storage (FORMAT_VERSION 2)**
   - Absolute vbyte encoding for neighbor IDs (preserves HNSW distance order)
   - ~50% space savings for typical graph sizes
   - Backward compatible with v1 format
   - Files: `src/vector/hnsw/disk_storage.rs`

2. **ADC (Asymmetric Distance Computation) Integration**
   - Precomputed lookup tables for fast quantized search
   - Helper function with fallback logging
   - Files: `src/vector/hnsw/index/mod.rs`, `src/vector/hnsw/storage.rs`

3. **Node.js Filter Support**
   - MongoDB-style filter parsing ($gt, $gte, $lt, $lte, $in, $contains, $and, $or)
   - Direct equality shorthand
   - Files: `node/src/lib.rs`, `node/__test__/index.spec.ts`

4. **Code Review Fixes**
   - Removed neighbor sorting (was breaking HNSW order)
   - Added bounds checks to vbyte_decode
   - Extracted shared write_neighbors_vbyte helpers
   - Improved ADC scale comment
   - Reverted wasteful 4KB metadata padding

### Test Status

- Rust: 336 passed
- Node.js: 30 passed

### Benchmark (Mac M3 Max, 10K vectors, 128D)

| Metric   | Result       |
| -------- | ------------ |
| Build    | 17,994 vec/s |
| Search   | 3,152 QPS    |
| Filtered | 315 QPS      |
| Batch    | 34,992 QPS   |

## What's Done

- [x] ADC integration for fast quantized search
- [x] Node.js metadata filter support
- [x] VByte disk compression (v2 format)
- [x] Code review fixes

## Next Steps

- [ ] Commit and push changes
- [ ] Final benchmarks on Fedora
- [ ] Consider release preparation
