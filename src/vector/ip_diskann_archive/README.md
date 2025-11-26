# IP-DiskANN Implementation (Archived)

**Status**: Archived November 8, 2025
**Reason**: Too slow for embedded use case (4 vec/sec vs HNSW 7K vec/sec)

---

## What is this?

Pure Rust implementation of IP-DiskANN algorithm based on:
- Microsoft Research paper (arXiv 2502.13826, February 2025)
- microsoft/DiskANN reference implementation (MIT licensed)

**Implementation**: 1,650+ lines
- `types.rs` - Core types (Neighbor, NodeId, config)
- `graph.rs` - Bi-directional graph structure
- `prune.rs` - RobustPrune algorithm
- `search.rs` - Greedy search
- `index.rs` - Main IPDiskANNIndex API
- `integration_tests.rs` - 100/1K/10K scale tests
- `debug_test.rs` - Graph visualization

**Tests**: 29 passing
- ✅ Correctness: 100% recall @ 1K vectors
- ❌ Performance: 4 vec/sec build speed

---

## Why Archived?

### Performance Issues

**Build Speed**:
- IP-DiskANN: 4 vec/sec @ 1K (223 seconds)
- HNSW: 7,000 vec/sec @ 10K
- **68x slower** than production HNSW

**Search Speed**:
- IP-DiskANN: 128 QPS @ 1K, 100% recall
- HNSW: 7,094 QPS @ 10K, 100% recall
- **55x slower** for queries

### Wrong Algorithm for Embedded

**IP-DiskANN is designed for**:
- Billion-scale datasets (1B+ vectors)
- SSD-backed storage
- Streaming updates (thousands per second)
- Cloud deployments

**omendb embedded needs**:
- 10K-1M vectors (fits in RAM)
- Static or infrequent updates
- Simple API (pip install)
- Fast build + query

**Conclusion**: HNSW is perfect for embedded, IP-DiskANN is overkill

---

## What We Learned

### 1. Paper Performance ≠ Implementation Performance

**Paper claims** (Microsoft DiskANN):
- 5,000 QPS @ 1B vectors
- <3ms latency
- Real-time updates

**Our implementation**:
- 4 vec/sec build (1,250x slower than paper claims)
- Would need C++ FFI to microsoft/DiskANN for production

### 2. Algorithm Choice is Scale-Dependent

| Scale | Best Algorithm |
|-------|---------------|
| 10K-1M | HNSW |
| 1M-10M | HNSW or IP-DiskANN |
| 10M-1B | IP-DiskANN (via FFI) |

### 3. Don't Over-Optimize for Scale We Don't Have

Mistake: Chose IP-DiskANN before validating embedded user needs
Reality: Embedded users need simple, fast, RAM-based solution
Learning: Ship what works today, not what scales to tomorrow

---

## Alternative Explored

**SPFresh** (SOSP 2023):
- Clustering-based (not graph)
- 1% DRAM usage
- LIRE incremental rebalancing
- Verdict: Different paradigm, no clear Rust implementation

**LSM-Vec** (2025):
- LSM-tree + graph hybrid
- 66% memory reduction
- High update throughput
- Verdict: Complex, no reference implementation

**HNSW variations**:
- Graph merging (IGTM/CGTM): 70% less compute for updates
- ACORN-1: 5x faster filtered search
- Dual-branch: Addresses local optima (too complex)

---

## Future Re-evaluation

**When to reconsider IP-DiskANN**:
1. Cloud deployment phase (Q3 2026+)
2. Need billion-scale support (>10M vectors)
3. Streaming updates required
4. Use FFI to microsoft/DiskANN (not pure Rust)

**Criteria for adoption**:
- <10ms @ 10M vectors
- Real-time inserts (thousands per second)
- Lower memory than HNSW at scale

---

## Current Plan

**Q1 2026**: Ship HNSW + Extended RaBitQ
- Python bindings (PyO3)
- ChromaDB performance parity
- 10K-1M vector target

**Q2 2026**: Add ACORN-1 filtered search
- Hybrid queries (vector + metadata)
- 5x speedup for filters
- Production-proven (Lucene/Elasticsearch)

**Q3 2026**: Evaluate incremental updates
- Option A: Graph merging (IGTM/CGTM)
- Option B: IP-DiskANN via FFI (if billion-scale needed)

---

## Research Artifacts

**Comprehensive analysis**:
- `ai/research/algorithm_comparison_nov2025.md`
- `ai/research/hnsw_variations_2024_2025.md`
- `ai/research/session_summary_nov8.md`
- `ai/DECISIONS.md` (2025-11-08 entry)

**Papers reviewed**:
- IP-DiskANN (arXiv 2502.13826)
- SPFresh (SOSP 2023)
- LSM-Vec (arXiv 2505.17152)
- ACORN-1 (arXiv 2403.04871)
- Graph Merging (arXiv 2505.16064)

---

## Code Quality

Despite being archived, this code is:
- ✅ Well-structured (modular design)
- ✅ Tested (29 passing tests)
- ✅ Correct (100% recall validated)
- ✅ Documented (inline comments)

**Not production-ready because**:
- ❌ Too slow (4 vec/sec)
- ❌ Wrong scale (designed for billions, not thousands)
- ❌ Missing optimizations (SIMD, parallel build, batching)

---

**Archived**: November 8, 2025
**Decision**: Use HNSW for embedded, defer IP-DiskANN to cloud phase
**Reference**: ai/DECISIONS.md (2025-11-08)
