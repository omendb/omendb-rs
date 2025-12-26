# Changelog

All notable changes to OmenDB are documented here.

## [0.0.19] - 2025-12-26

### Added

- CHANGELOG.md for tracking releases
- Pre-release checklist in RELEASING.md

### Developer

- Fixed all clippy warnings across lib, Python, and Node bindings
- Updated RELEASING.md with git tagging workflow

## [0.0.18] - 2025-12-25

### Changed

- **Stability**: Eliminated all `unwrap()` calls in core modules - errors now return `Result`
- **Stability**: Replaced `panic!` with `Option` in metadata index functions
- **Stability**: Replaced `assert!` with `Result` in compression training
- Consolidated duplicate type definitions

### Fixed

- Bounds check for `new_idx` in `optimize()`
- Log errors instead of silently ignoring in `delete_batch`

### Added

- 13 tests for error path coverage
- Benchmark script now stores latency metrics (`s_ms`, `b_ms`)

### Developer

- Fixed all clippy warnings (lib, Python, Node bindings)
- Clean up repo artifacts

## [0.0.17] - 2025-12-24

### Changed

- **License**: Changed from AGPL-3.0 to Elastic License 2.0

### Fixed

- Handle tantivy cleanup race in test fixtures
- Use `contextlib.suppress` for ruff SIM105
- Add Elastic-2.0 to allowed licenses in deny.toml
- Use license text instead of file reference for maturin sdist

## [0.0.16] - 2025-12-20

### Added

- `max_distance` search parameter - filter results by maximum distance threshold
- Builder methods for HNSWParams (`with_m`, `with_ef_construction`)

### Changed

- **License**: Changed from Apache-2.0 to AGPL-3.0 (later changed to Elastic-2.0 in 0.0.17)
- Merged omendb-core into main omendb crate (simpler structure)
- Consolidated to single HNSW implementation

### Fixed

- VectorStore mappings after `optimize()`
- npm platform package require paths
- Missing k and ef validation in Node binding
- Decouple publish jobs so crates.io failure doesn't block npm/PyPI

### Removed

- `valid_at` parameter (removed before release)
- omendb-core crate (merged into main)

## [0.0.15] - 2025-12-15

Previous stable release.
