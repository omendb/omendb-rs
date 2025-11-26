# TODO - seerdb

**Last Updated**: November 26, 2025
**Status**: Pre-release hardening

---

## Pre-Release Checklist (0.1.0)

Goal: Complete all items before release for stability.

### Hardening (Code Changes)
- [x] WAL per-record checksums ✅
  - V2 format: `[crc32c:4 LE][len:4 BE][record_data:len]`
  - CRC32C validated on read, backward compatible with V1
  - Hardware-accelerated via crc32c crate
- [x] `db.verify()` API ✅
  - Full database integrity check
  - Validates all SSTable block checksums, vLog record checksums
- [ ] Recovery mode option (~1 hr)
  - `RecoveryMode::Strict` - fail on any corruption
  - `RecoveryMode::BestEffort` - skip corrupted records
- [x] Fix flaky test: `test_delete_during_read` ✅
  - Used AtomicBool flag for deterministic synchronization
  - Commit: `d4730f7`

### Validation (No Code Changes)
- [x] 100M scale test (Fedora) - ✅ 930K ops/sec
- [x] 500M scale test (Mac) - ✅ Completed
- [x] SSTable benchmarks (Fedora) - ✅ See STATUS.md
- [ ] Extended fuzz testing - 1hr runs

### Documentation
- [ ] Update README with Fedora perf numbers
- [ ] API docs review

---

## Completed (Nov 2025)

| Feature | Notes |
|---------|-------|
| db.verify() API | `DB::verify()` validates all SSTable/vLog checksums |
| MVCC prefix key bug | `Block::find_mvcc()` - InternalKey encoding fix |
| Scale test example | `billion_key_scale.rs` - 100M verified |
| Fuzzing setup | 4 fuzz targets working |
| Crash recovery tests | 7 stress tests passing |
| Examples cleanup | 60→10 essential |
| Config profiles | `embedded()`, `high_throughput()`, `large_scale()` |
| Skip WAL | `DBOptions.skip_wal` |
| Direct WAL | `DBOptions.use_direct_wal` |
| ZSTD compression | `CompressionType::Zstd` |
| Tiered storage | `cold_tier_level`, `cold_storage` |
| Bulk load API | `db.bulk_load()` |
| Transaction API | OCC with snapshot isolation |

---

## Not Planned

| Feature | Reason |
|---------|--------|
| io_uring | Security CVEs, cloud providers disable |
| SSI (serializable) | OCC+SI sufficient |
| Lock-free WAL | Batch API is the right pattern |
| Column families | Use key prefixes |
| MANIFEST file | Not needed - rebuild from SSTables |
