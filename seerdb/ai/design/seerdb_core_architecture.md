# seerdb Core Architecture

**Status**: Alpha (Implementation Active)
**Last Updated**: November 18, 2025
**License**: Apache 2.0

## Overview

`seerdb` is a high-performance, embedded LSM-tree storage engine written in Rust. It is designed as the storage foundation for `omendb` (a vector database) but remains a general-purpose key-value store.

**Key Differentiators**:
1.  **Learned Indexing**: Uses ALEX (Adaptive Learned Index) for SSTable blocks to improve read performance.
2.  **WiscKey Separation**: Separates large values (vLog) to minimize write amplification (validated 1.01x vs RocksDB's ~5x).
3.  **Workload Awareness**: Optimized for high-throughput prefix scans (graph traversals).
4.  **Cloud Native (Planned)**: Architecture supports hybrid storage (Local Memtable + S3 SSTables).

---

## System Architecture

```mermaid
graph TD
    Client[Client API] --> API[DB Interface]
    API --> Batch[Write Batch]
    API --> Read[Read Path]
    
    subgraph "Write Path (Durability)"
        Batch --> WAL[Write Ahead Log]
        Batch --> Memtable[Partitioned Memtable]
        WAL --> DiskWAL[(Local Disk WAL)]
    end
    
    subgraph "Read Path (Latency)"
        Read --> Memtable
        Read --> BlockCache[Block Cache]
        BlockCache --> SSTables[SSTable Manager]
    end
    
    subgraph "Background Operations"
        Flush[Flush Worker] --> |Immutable Memtable| Writer[SSTable Writer]
        Writer --> SSTables
        Compaction[Compaction Worker] --> SSTables
    end
    
    subgraph "Storage Layer (Pluggable)"
        SSTables --> |Read/Write| StorageTrait[Storage Trait]
        StorageTrait --> Local[Local Filesystem]
        StorageTrait --> ObjectStore[Object Store (S3/GCS)]
    end
```

## Core Components & SOTA Alignment

### 1. Buffer Management (Target: LeanStore)
*   **Current**: `quick_cache` (Sharded LRU) + OS Page Cache.
    *   Effective for current scale but incurs overhead from double caching (OS + Application).
*   **Target (SOTA)**: **LeanStore Buffer Manager**.
    *   **Concept**: Pointer swizzling to avoid hash table lookups for in-memory pages.
    *   **Plan**: Implement for 0.1.0+ to reduce CPU overhead and fully exploit RAM.
    *   *Note*: LeanStore replaces standard block caching; implementation is a significant architectural shift.

### 2. Concurrency & I/O Model
*   **Current**: Threaded Synchronous I/O.
    *   Reads: blocking `std::fs` calls.
    *   Writes: `parking_lot` mutexes.
    *   **Status**: Optimal for macOS/cross-platform compatibility and current scale.
*   **Target (SOTA)**: **Async I/O (io_uring)**.
    *   **Concept**: Submit batch I/O requests to kernel, wake up only on completion.
    *   **Plan**: Future optimization for Linux production builds.
    *   **Constraint**: Must maintain synchronous fallback for macOS/Windows testing.

### 3. Memtable (Write Buffer)
*   **Structure**: Partitioned SkipList (Crossbeam-Skiplist).
*   **Partitioning**: 16 shards to reduce contention (validated performance gain).
*   **Snapshot Isolation**:
    *   **Current**: **Memtable Switching (Copy-On-Write)**. Atomically swaps active memtables. Simple, correct, but forces flushes.
    *   **Target**: **MVCC**. Full versioning in memtable to allow snapshots without flush. (Future optimization).

### 4. Storage Format (SSTable)
*   **Layout**:
    *   `[Data Blocks]`: LZ4 compressed key-value pairs.
    *   `[Filter Block]`: Bloom Filter (10 bits/key).
    *   `[Index Block]`: ALEX (Adaptive Learned Index) for block offsets.
*   **Key-Value Separation (WiscKey)**:
    *   **Status**: **Implemented & SOTA**.
    *   **Mechanism**: Large values (> threshold) written to `vLog`; SSTable stores pointer.
    *   **Impact**: Critical for 100B vector scale to minimize compaction I/O.

### 5. Compaction Engine
*   **Strategy**: Leveling (L0 -> L1 -> ... -> L6).
*   **Critical Gap**: **Custom Compaction Filters**.
    *   *Problem*: Vector graph indexes (HNSW) need intelligent merging, not just byte-wise compaction.
    *   *Solution*: Expose `CompactionFilter` trait to consumers.

---

## Roadmap to Massive Scale

To support massive vector scale, `seerdb` must evolve:

1.  **Object Store Integration (Priority 1 - Critical)**:
    *   Implement `Storage` trait using `object_store` (S3/GCS).
    *   Enable "Hybrid Mode": Local WAL/Memtable + Cloud SSTables.
    *   *Requirement*: Stick to `tokio` for now; `io_uring` is not a blocker.

2.  **Compaction Filters (Priority 2 - Critical)**:
    *   Allow consumers to inject logic during compaction (e.g., `merge_graphs`).
    *   Essential for correctness and performance of Graph-based LSM.

3.  **LeanStore Integration (Priority 3 - Optimization)**:
    *   Implement pointer swizzling buffer manager.
    *   *Goal*: Maximize single-node performance before scaling out.

4.  **Async API (Priority 4 - Feature)**:
    *   Expose `async fn` to clients for high-concurrency cloud workloads.
