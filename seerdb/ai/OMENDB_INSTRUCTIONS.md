# OMENDB INSTRUCTIONS

**Goal**: Instructions for working in the `../omendb` private repository.

## Context
*   **`seerdb`**: Public Open Source engine (this repo).
*   **`omendb`**: Private Product repo (sibling directory).
*   **Relationship**: `omendb` depends on `seerdb` (via path dependency).

## Workflows

### 1. Switching Context
To work on `omendb`, you must access the sibling directory `../omendb`.
*   **Permission**: If `opencode` asks for permission, grant it.
*   **Path**: Always use absolute paths or relative `../omendb` paths.

### 2. Integration Points
*   **Merge Operator**: `seerdb` defines `trait MergeOperator`. `omendb` implements it for `GraphEdge` merging.
*   **Vector Index**: `omendb` implements the vector index (HNSW/IVF) which stores data in `seerdb`.
*   **Prefix Bloom**: `omendb` configures `seerdb` to use specific prefix extractors.

### 3. "seerdb-vector"
*   This is a workspace crate inside `omendb`.
*   It contains the vector-specific optimizations that are proprietary.

## Agent Protocol
When working on `omendb` tasks:
1.  Read `../omendb/ai/STATUS.md` first.
2.  Check `../omendb/Cargo.toml` to ensure it points to local `seerdb`.
3.  Do not commit proprietary code to `seerdb` repo.
