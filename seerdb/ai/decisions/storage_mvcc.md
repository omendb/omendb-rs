# Decision: Native MVCC with Internal Keys

**Date**: November 24, 2025
**Status**: Accepted
**Context**: Omendb requires Snapshot Isolation for consistent graph traversals and atomic transactions.

## Problem
The previous storage engine (v0.0.1) was a simple Key-Value store.
- Overwrites replaced data immediately (Last-Write-Wins).
- No read consistency across multiple keys (dirty reads possible during concurrent writes).
- Graph traversals could see inconsistent state if edges were updated during traversal.

## Solution: Native MVCC (Multi-Version Concurrency Control)

We are adopting a RocksDB/LevelDB-style MVCC implementation where versioning is baked into the storage key.

### 1. Internal Key Format
All keys stored in Memtable and SSTables are now `InternalKey`s:
```rust
struct InternalKey {
    user_key: Bytes,
    seq: u64,
    kind: ValueType,
}
```

**Encoding**: `[ User Key ] [ 8 bytes: (SeqNum << 8) | ValueType ]`

**Sorting**:
1.  User Key (Ascending)
2.  Sequence Number (Descending) - *Inverted in encoding*

This ensures that for any `User Key`, the latest version appears first in the iterator.

### 2. Sequence Numbers
- **Global Counter**: `DB.next_seq` (AtomicU64) increments on every write.
- **WAL**: Sequence numbers are now stored in the WAL `Record`. This ensures that upon recovery, we restore the exact version history.
- **Memtable**: Stores `InternalKey` directly.

### 3. Read Path (Snapshot Isolation)
- `get(key)` captures `seq = DB.next_seq` (or uses a specific snapshot seq).
- Iterators seek to `InternalKey(key, snapshot_seq)`.
- Since sorting is Seq Descending, the iterator naturally lands on the newest version `<= snapshot_seq`.

### 4. Compaction
- Compaction now drops versions of keys that are:
  1.  Overwritten by a newer version *and*
  2.  Older than the oldest active snapshot.
- Tombstones are effectively `ValueType::Deletion` records.

## Alternatives Considered

### A. External MVCC Layer
Implement MVCC in `omendb` (application layer) by appending versions to keys manually.
- **Pros**: Keeps `seerdb` simple.
- **Cons**: Massive read amplification. `scan(prefix)` becomes very hard (need to skip versions manually). Performance penalty.

### B. Append-Only Log (Bitcask style)
- **Pros**: Simple.
- **Cons**: High read amplification for scans. Not suitable for graph workloads which require efficient range scans.

## Consequences
- **Breaking Change**: Storage format changes. Old DBs incompatible.
- **WAL Format**: Changed to include seq.
- **Memtable**: Changed to `SkipMap<InternalKey, Bytes>`.
- **Performance**: Slight overhead (8 bytes per key). Compression usually hides this.
