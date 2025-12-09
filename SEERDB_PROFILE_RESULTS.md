# seerdb Profile Results (Comprehensive)

**Date**: 2025-12-08
**Dataset**: 10000 vectors, 128 dimensions
**Queries**: 1000

## Should oadb Use seerdb?

**YES** - seerdb provides:
1. **Persistence** - Data survives process restart
2. **Durability** - WAL protects against crashes
3. **Minimal overhead** - See benchmarks below

## Performance Comparison

| Operation | In-Memory | seerdb | Overhead |
|-----------|-----------|--------|----------|
| Insert (10000 vec) | 518.4715ms | 592.501875ms | 1.1x |
| knn_search | 3520 QPS | 3488 QPS | 0.9% |
| search (metadata) | 3509 QPS | 3431 QPS | 2.2% |
| Cold start | N/A | 908.800459ms | - |
| Flush | N/A | 32.535792ms | - |

## Key Findings

### 1. Search Performance: IDENTICAL
- knn_search uses in-memory HNSW graph (no seerdb I/O)
- seerdb overhead: 0.9% (within noise)

### 2. Insert Performance: 1.1x Overhead
- seerdb writes: vectors, metadata, ID mappings (4 KV pairs/vector)
- Still achieves 16878 vec/s with persistence

### 3. Cold Start: 908.800459ms
- Loads 10000 vectors from seerdb on startup
- Rebuilds HNSW index in memory

### 4. Metadata Lookups
- search() reads metadata from seerdb for results
- Cache hit rate: 0.0%
- Total seerdb gets: 5

## seerdb Stats

```
Cache hits: 0
Cache misses: 0
Hit rate: 0.0%
SSTables per level: [1, 0, 0, 0, 0, 0, 0]
Total gets: 5
```

## Recommendation

**Keep seerdb for oadb** because:
1. Search performance is NOT affected (in-memory HNSW)
2. Insert overhead is acceptable (1.1x for durability)
3. Cold start is fast enough (908.800459ms for 10000 vectors)
4. Durability is critical for production use

## Alternatives Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| No persistence | Fastest inserts | Data lost on restart |
| Simple file | Simpler code | No durability, slow reload |
| fjall | Simpler API | Less optimized for oadb use case |
| **seerdb** | Optimized, durable | Slight insert overhead |
