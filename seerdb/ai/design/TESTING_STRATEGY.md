# Testing Strategy & Specifications

**Goal**: 80%+ code coverage and production-grade stability.
**Scope**: Unit tests, Integration tests, Crash Recovery, Fuzzing, and Soak Testing.

---

## 5. System-Level Benchmarking Strategy

**Goal**: Validate concurrency model, lock contention, and macro-level performance.

### Benchmarks
1.  **Mixed Workload (`mixed_workload`)**:
    *   **Pattern**: 50% Get, 40% Put, 10% Scan (YCSB-like).
    *   **Concurrency**: 1, 2, 4, 8, 16 threads.
    *   **Metric**: Ops/sec, P99 Latency, Lock Contention.
    *   **Focus**: Validate `ArcSwap` and `PipelinedWAL` under contention.

2.  **Write Amplification (`write_amplification`)**:
    *   **Pattern**: sustained random overwrites.
    *   **Metric**: (Physical Bytes Written to Disk) / (Logical Bytes Written by User).
    *   **Target**: < 1.0x (with VLog), < 10x (pure LSM).

3.  **Recovery Scale (`recovery_scale`)**:
    *   **Pattern**: Replay 1GB+ WAL.
    *   **Metric**: MB/sec replay speed.
    *   **Target**: > 500 MB/sec.

---

## 1. Test Inventory & Gaps

### Current State
| Focus Area | Status |
|------------|--------|
| Batch Atomicity | ✅ Complete |
| Compaction Correctness | ✅ Complete |
| Concurrent Operations | ✅ Good |
| Configuration | ✅ Good |
| Leak Detection | ✅ Good |
| Snapshot Consistency | ✅ Good |
| Stress Testing | ✅ Good |

### Critical Gaps (High Priority)
1.  **ALEX Learned Index**: Node split/merge logic, bulk loading.
2.  **VLog**: Corruption detection, truncation handling.
3.  **SSTable**: Block corruption, varint decoding errors.
4.  **Compaction**: Multi-level cascading, size ratio enforcement.
5.  **WAL Recovery**: Partial records, header corruption.

---

## 2. Crash Recovery Strategy

**Approach**: Simulated crashes (fast, deterministic) rather than process killing.

### Test Cases
1.  **WAL Recovery**: Write without flush -> Close -> Reopen.
2.  **Flush Crash**: Simulate crash during flush (incomplete SSTable).
3.  **Compaction Crash**: Simulate crash mid-compaction (partial output).
4.  **Corruption**: Flip bits in SSTable/WAL/vLog -> Verify checksum error.
5.  **Truncation**: Truncate files -> Verify graceful recovery.

### SyncPolicy Tests
- `SyncAll`: Verify every write durable.
- `SyncData`: Verify data persisted.
- `SyncNone`: Verify recovery without recent data.

---

## 3. Fuzzing Strategy

**Tool**: `cargo-fuzz` (libfuzzer).

### Fuzz Targets
1.  **`sstable_parse`**: Random bytes -> `SSTable::open()`. Tests header/index/bloom parsing.
2.  **`wal_parse`**: Random bytes -> WAL iterator. Tests record validation.
3.  **`vlog_parse`**: Random bytes -> vLog reader. Tests value extraction.
4.  **`db_operations`**: Random sequence of put/get/delete/scan/flush. Tests API robustness.

### Execution
- **CI**: Short runs (60s) to catch obvious regressions.
- **Nightly**: Long runs (24h+) on dedicated infrastructure.

---

## 4. Soak Testing Strategy

**Purpose**: Validate long-term stability and memory usage (leak detection).

### Tests
1.  **24-Hour Soak (`test_24hour_soak`)**:
    *   Mixed read/write workload (70/30).
    *   Monitor memory (must stay <3x initial).
    *   Verify throughput stability.
2.  **100GB Dataset (`test_large_dataset_100gb`)**:
    *   Write 100M keys.
    *   Verify LSM compaction at scale.
    *   Ensure memory bounded during massive writes.

### Success Criteria
- **No Crashes**: Process must survive 24h.
- **Bounded Memory**: No unbounded growth.
- **Data Integrity**: All reads must succeed.
