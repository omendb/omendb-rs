# STATUS - seerdb

**Last Updated**: November 26, 2025
**Version**: 0.0.1-alpha
**Status**: Pre-release hardening

---

## Current State

| Metric | Value |
|--------|-------|
| **Tests** | 227 tests passing (220 lib + 7 stress) |
| **Compilation** | Clean (no errors, no warnings) |
| **Lines of Code** | ~30K Rust |
| **Scale Verified** | 100M keys (Fedora: 930K writes/sec) |

---

## Active Work: Pre-Release Hardening

Goal: Complete all seerdb items before 0.1.0 release for stability.

### Hardening (Code Changes Required)
| Item | Status | Notes |
|------|--------|-------|
| WAL per-record checksums | ✅ Done | V2 format with CRC32C |
| `db.verify()` API | ✅ Done | Full database integrity check |
| Recovery mode option | ⬜ Pending | `Strict` vs `BestEffort` (~1hr) |

### Validation (No Code Changes)
| Item | Status | Notes |
|------|--------|-------|
| 100M scale test | ✅ Done | Fedora: 930K ops/sec |
| 500M scale test | ✅ Done | Mac: Completed |
| SSTable benchmarks (Fedora) | ✅ Done | See below |
| Extended fuzz | ⬜ Pending | 1hr runs |

### Documentation
| Item | Status |
|------|--------|
| README perf numbers | ⬜ Pending |
| API docs review | ⬜ Partial |

---

## Fedora SSTable Benchmark Results (i9-13900KF)

| Benchmark | 1K entries | 10K entries | 100K entries |
|-----------|-----------|-------------|--------------|
| existing_keys | 3.27 µs | 39.16 µs | 550 µs |
| missing_keys | 7.82 µs | 9.43 µs | 14.93 µs |
| full_scan | 127.5 µs | 1.21 ms | - |

**Bloom filter benefit**: Missing keys ~37x faster than existing at 100K (bloom filter rejects quickly).

---

## Checksum Coverage

| Component | Checksum | Notes |
|-----------|----------|-------|
| SSTable blocks | ✅ CRC32C | Per-block checksums |
| vLog entries | ✅ CRC32C | Per-entry checksums |
| WAL header | ✅ Magic | Validates 0x574C4F47 |
| WAL records | ✅ CRC32C | V2 format, hardware-accelerated |

---

## Recent Fixes (Nov 2025)

| Fix | Commit | Notes |
|-----|--------|-------|
| MVCC prefix key bug | `092f3ec` | `Block::find_mvcc()` - InternalKey encoding fix |
| WAL corruption test | `6e6290e` | Header magic validation |
| Scale test example | - | `billion_key_scale.rs` verified 100M keys |

---

## Architecture

```
Write Path:  put() → WAL (seq#) → Memtable (InternalKey) → [flush] → SSTable
Read Path:   get() → Memtable → Immutable Memtables → L0..L6 SSTables
Snapshot:    Captures current seq# → reads filter by seq ≤ snapshot_seq
```

### Module Map

```
src/
├── types.rs          # InternalKey, ValueType (MVCC core)
├── db.rs             # Main DB interface
├── memtable/         # Partitioned concurrent skiplist
├── wal/              # Write-ahead log with seq numbers
├── sstable/          # SSTable format with bloom + ALEX
├── compaction/       # Leveled compaction + filters
├── vlog/             # WiscKey value separation
├── snapshot.rs       # Point-in-time reads
├── transaction.rs    # OCC transactions
├── batch.rs          # Atomic batch writes
├── bloom/            # Traditional + learned bloom
├── alex/             # ALEX learned index
└── buffer/           # Buffer pool management
```
