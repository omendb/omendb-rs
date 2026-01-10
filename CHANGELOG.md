# Changelog

## v0.0.22 (Unreleased)

### BREAKING CHANGES

**Persistence format upgraded to v2 (postcard)**
- Existing `.omen` files from v0.0.21 or earlier will fail to load
- Error: `"Unsupported version: 1 (expected 2)"`
- **Migration**: Re-create databases by reinserting vectors
- Reason: bincode → postcard for better maintenance and smaller files

### Bug Fixes

- **ACORN-1 filtered search**: Fix sparse filter searches (<10% selectivity)
  - Entry points that don't match filter now have neighbors explored
  - Correctly implements 2-hop expansion per ACORN-1 paper
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
