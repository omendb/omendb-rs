# seerdb - Research-Grade Storage Engine

**Repository**: seerdb (Storage Engine with Learned Data Structures)
**Last Updated**: November 20, 2025
**License**: Apache-2.0
**Status**: Production-ready (stability complete)

---

## Product Overview

**seerdb**: Modern embedded storage engine implementing 2018-2024 research

**What It Is**:
- LSM-tree based storage engine (like RocksDB)
- Learned data structures (ALEX) replace traditional components
- WiscKey separation for minimized write amplification
- Optimized for graph workloads and massive scale

**Positioning**: "RocksDB but with 2020s research - 4.82x better write amplification (validated)"

**Key Features**:
- ✅ **Performance**: 878K writes/sec (2.47x RocksDB), 4.7M reads/sec
- ✅ **Snapshots**: Point-in-time consistent views
- ✅ **Range Iterators**: `range()`, `prefix()`
- ✅ **Merge Operators**: O(1) blind writes for graphs
- ✅ **Zero Data Loss**: WAL + fsync on shutdown

---

## Quick Start for AI Agents

**→ First time?** Load these in order:
1. This file (AGENTS.md / CLAUDE.md symlink) - Project overview
2. `ai/STATUS.md` - Current state (read FIRST)
3. `ai/TODO.md` - Active tasks and priorities
4. `ai/PLAN.md` - Strategic roadmap (Massive scale)
5. `ai/design/seerdb_core_architecture.md` - Core Architecture

**→ Deep Context?** See `ai/research/PROJECT_CONTEXT.md` for competitive analysis and workload details.

**→ API research?** See `ai/design/API_COMPARISON_TABLE.md` for gap analysis

**→ Private Repo?** See `ai/OMENDB_INSTRUCTIONS.md` for `omendb` context.

**→ Full documentation guide**: See `ai/README.md` for all available docs

---

## Environment & Benchmarking

### Development Environments

- **Mac (M3 Max, 128GB)**: Primary development, large-scale tests
  - **Backend**: `tokio` + `LocalFileSystem`
  - **Use for**: Development, functional tests, large-scale integration tests

- **Fedora (i9-13900KF, 32GB, RTX 4090)**: Performance benchmarks
  - **Backend**: `io_uring` (Linux-specific optimizations)
  - **Use for**: SOTA performance validation, Linux-specific optimizations

### When to Benchmark Where

**Run on Mac**:
- ✅ Quick iteration benchmarks during development
- ✅ Functional correctness validation
- ✅ Large-scale tests (Mac has 128GB RAM)
- ✅ Cross-platform compatibility checks

**Run on Fedora**:
- ✅ **SOTA performance claims** (required for publication/docs)
- ✅ Linux-specific optimizations (io_uring, kernel features)
- ✅ Final pre-release validation
- ✅ Competitive benchmarks vs RocksDB, LevelDB
- ✅ CPU-intensive workloads (i9-13900KF has higher single-thread perf)

**Rule of Thumb**: Develop on Mac, validate SOTA on Fedora. Always benchmark on Fedora before claiming performance numbers in docs/commits.

## Workflow Rules
- NO AI attribution in commits/PRs (strip manually)
- Ask before: PRs, publishing packages, force ops, resource deletion
- Commit frequently, push regularly (no ask needed)
- Never force push to main/master
- Delete files directly (no archiving)
- `/tmp`: ephemeral test artifacts only. `ai/`: context/state across sessions
  - Delete temp artifacts after use, never commit

### Git Worktree (Required)

**Use `../seerdb-dev/` worktree for all development** - the main directory (`../seerdb/`) is used by omendb and must stay stable.

```bash
# Development workflow:
cd ../seerdb-dev/                    # Work in the dev worktree
git checkout -b feature-name         # Create feature branch
# ... make changes, test, commit ...
git push -u origin feature-name      # Push branch

# When ready to merge:
cd ../seerdb/                        # Switch to main worktree
git merge feature-name               # Merge completed work
git push                             # Push to main
```

**Worktree locations:**
- `../seerdb/` - main branch, stable (used by omendb)
- `../seerdb-dev/` - development worktree for feature work

### ai/ Directory
**AI session context** - workspace for tracking project state across sessions. Read first, update on exit.

| File | Purpose |
|------|---------|
| `AGENTS.md` | Project overview (symlink: `CLAUDE.md` → `AGENTS.md`) |
| `ai/STATUS.md` | Current state, blockers (read FIRST) |
| `ai/TODO.md` | Active tasks only |
| `ai/PLAN.md` | Strategic roadmap |
| `ai/research/PROJECT_CONTEXT.md` | Detailed context, research, & competitive analysis |
| `ai/design/` | Design specifications |

**Format:** All ai/ files - tables/lists/structured (not prose). Answer first, evidence second.

---

## Repository Structure

```
seerdb/
├── AGENTS.md              # Primary AI agent entry point
├── CLAUDE.md → AGENTS.md  # Symlink for Claude Code compatibility
├── README.md              # Public documentation
├── LICENSE                # Elastic License 2.0
├── Cargo.toml             # Rust package manifest
├── src/
│   ├── lib.rs             # Public API
│   ├── wal/               # Write-ahead log
│   ├── memtable/          # In-memory buffer (partitioned skiplist)
│   ├── sstable/           # SSTable format
│   ├── compaction/        # Compaction strategies
│   ├── vlog/              # Value log (WiscKey)
│   ├── cache/             # Block cache (quick_cache)
│   └── simd/              # SIMD optimizations
├── examples/              # Usage examples + benchmarks
├── benches/               # Performance benchmarks
├── tests/                 # Integration tests
└── ai/                    # AI session context
```
