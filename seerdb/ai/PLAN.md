# PLAN - seerdb

**Goal**: SOTA storage engine optimized for vector/graph workloads
**Status**: Stable, feature-complete for current needs
**Last Updated**: November 25, 2025

---

## Target Workloads

### Embedded Vector DB (oadb)
- Single machine, millions of vectors
- Low latency point lookups (<1ms)
- Memory efficient (64-256MB footprint)

### Large-Scale Vector DB (OmenDB)
- Distributed, cloud-native
- High throughput batch writes
- Tiered storage (hot/cold to S3/GCS)

---

## Completed Features (Nov 2025)

| Feature | Status | Notes |
|---------|--------|-------|
| MVCC core | ✅ | InternalKey, Memtable, WAL, SSTable |
| Snapshot API | ✅ | Point-in-time reads |
| Transaction API | ✅ | OCC with snapshot isolation |
| Batch API | ✅ | 10x faster than single writes |
| Config profiles | ✅ | `embedded()`, `high_throughput()`, `large_scale()` |
| Skip WAL | ✅ | For rebuildable index data |
| Direct WAL | ✅ | 6.9x faster single writes |
| ZSTD compression | ✅ | `CompressionType::Zstd` |
| Tiered storage | ✅ | `cold_tier_level`, `cold_storage` |
| Bulk load API | ✅ | Bypass WAL for initial load |

---

## Deferred (Not Planned)

| Feature | Reason |
|---------|--------|
| io_uring async I/O | Security CVEs, cloud providers disable it |
| SSI (serializable) | OCC+SI sufficient, SSI adds complexity |
| Lock-free WAL | Marginal gain, batch API is the right pattern |
| Column families | Use key prefixes instead |

---

## Architecture

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MVCC | Native InternalKey | RocksDB-style, no external deps |
| Compaction | Leveled | Bounded read amp for graph queries |
| Value separation | WiscKey vLog | 4.82x better write amp |
| Learned index | ALEX | Faster SSTable block lookups |
| Block cache | quick_cache | Lock-free, high concurrency |

---

## Performance (v0.0.1-alpha, M3 Max)

| Workload | Throughput | vs RocksDB |
|----------|------------|------------|
| Skip WAL writes | 2.94M ops/sec | ~8x faster |
| Direct WAL writes | 2.15M ops/sec | 6x faster |
| Batch writes | 1.9M ops/sec | 5.5x faster |
| Point reads | 4.1M ops/sec | 3.8x faster |
| Write amplification | 1.01x | 4.82x better |
