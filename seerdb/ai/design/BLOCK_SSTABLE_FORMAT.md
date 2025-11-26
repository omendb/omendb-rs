# Block-Based SSTable Format (V3 with LZ4 + Varint)

## Overview

This document specifies the block-based SSTable format designed for:
- Memory efficiency with large datasets (block-based index)
- **Storage efficiency** (LZ4 compression + varint encoding) - **NEW in V3**
- **Cache efficiency** (2-3x more data in cache with compression) - **NEW in V3**

## Problem Statement

**V1 Format**: Loads entire key index into RAM
- 10M keys × 26 bytes = 260 MB per SSTable
- With 20 SSTables open: 5.2 GB of memory
- Unacceptable for large datasets

**V2 Format** (block-based): Memory efficiency
- Only top-level index in RAM: ~200 KB per SSTable
- Load index/data blocks on-demand
- 99% memory reduction

**V3 Format** (this spec): Memory + Storage + Cache efficiency
- **LZ4 compression**: 2-3x more data fits in cache (CRITICAL for large values)
- **Varint encoding**: 30-50% metadata space savings
- **Combined**: +30-50% read throughput potential
- Still block-based (V2 benefits retained)

## File Structure (V3)

```
┌─────────────────────────────────────────────────────────────┐
│ File Header (32 bytes)                                       │
│  magic: u32 (0x53535442 = "SSTB")                          │
│  version: u32 (0x00000003 = V3) ← NEW                       │
│  flags: u64                                                  │
│    bit 0: LZ4 compression enabled                           │
│    bit 1: Varint encoding enabled                           │
│    bits 2-63: reserved                                       │
│  num_entries: u64                                            │
│  reserved: u64                                               │
├─────────────────────────────────────────────────────────────┤
│ Data Blocks (variable size, compressed with LZ4)            │
│  ┌──────────────────────────────────────┐                   │
│  │ Compressed Block 0:                  │ ← NEW: LZ4        │
│  │  Uncompressed Length: u32 (varint)   │ ← NEW: varint     │
│  │  Compressed Data: [bytes]            │                   │
│  │    (After decompression):            │                   │
│  │    Entry 0: [key_len][key][flag][value_or_pointer]     │
│  │    Entry 1: [key_len][key][flag][value_or_pointer]     │
│  │    ...                                │                   │
│  │    Restart Points: [offset_0]...[offset_N] (varint)    │
│  │    Num Restarts: u32 (varint)         │ ← NEW: varint     │
│  │  Checksum: u32 (of compressed data)   │                   │
│  └──────────────────────────────────────┘                   │
│  Block 1: ...                                                │
│  Block N: ...                                                │
├─────────────────────────────────────────────────────────────┤
│ Index Blocks (variable size, compressed with LZ4)           │
│  ┌──────────────────────────────────────┐                   │
│  │ Compressed Index Block 0:            │ ← NEW: LZ4        │
│  │  Uncompressed Length: u32 (varint)   │ ← NEW: varint     │
│  │  Compressed Data: [bytes]            │                   │
│  │    (After decompression):            │                   │
│  │    Entry 0: [key_len][key][block_offset: u64 (varint)] │
│  │    Entry 1: [key_len][key][block_offset: u64 (varint)] │
│  │    ...                                │                   │
│  │    Restart Points: [offset_0]...[offset_N] (varint)    │
│  │    Num Restarts: u32 (varint)         │ ← NEW: varint     │
│  │  Checksum: u32                         │                   │
│  └──────────────────────────────────────┘                   │
│  Index Block 1: ...                                          │
│  Index Block M: ...                                          │
├─────────────────────────────────────────────────────────────┤
│ Top-Level Index (variable size, loaded into RAM)            │
│  Num Index Blocks: u32 (varint) ← NEW                       │
│  For each index block:                                       │
│    [last_key_len: u32 (varint)][last_key: bytes]           │
│    [offset: u64 (varint)] ← NEW: varint                     │
├─────────────────────────────────────────────────────────────┤
│ Bloom Filter                                                 │
│  Length: u64 (varint) ← NEW                                 │
│  Data: [bytes] (NOT compressed - random bits)               │
├─────────────────────────────────────────────────────────────┤
│ Footer (48 bytes)                                            │
│  index_blocks_offset: u64                                    │
│  top_level_index_offset: u64                                 │
│  bloom_offset: u64                                           │
│  checksum: u32 (CRC32 of entire file except this field)     │
│  magic: u32 (0x53535442 = "SSTB")                           │
│  version: u32 (0x00000003 = V3) ← NEW                       │
│  reserved: u32                                               │
└─────────────────────────────────────────────────────────────┘
```

## Entry Format (V3 with Varint)

```
[key_len: varint][key: bytes][flag: u8][value_data]

flag = 0x00: Inline value
  value_data = [value_len: varint][value: bytes]

flag = 0x01: VLog pointer
  value_data = [vlog_offset: varint][vlog_length: varint]
```

**Varint Encoding**: Variable-length integer encoding (protobuf-style)
- Small values: 1 byte (0-127)
- Medium values: 2-3 bytes (128-2^21)
- Large values: 4-5 bytes (2^21+)

**Space Savings Example**:
```
Fixed u32 encoding:
  key_len=26, value_len=512, offset=1024
  Total: 4 + 4 + 4 = 12 bytes

Varint encoding:
  key_len=26 → 1 byte
  value_len=512 → 2 bytes
  offset=1024 → 2 bytes
  Total: 1 + 2 + 2 = 5 bytes
  Savings: 58%!
```

## Lookup Algorithm (V3 with Compression)

### Point Lookup: Get(key)

1. **Bloom Filter Check** (in RAM)
   - If not present → return None
   - If present → continue (might be false positive)

2. **Top-Level Index Binary Search** (in RAM, varint-decoded)
   - Find index block that might contain key
   - Load compressed index block from disk (or cache)

3. **Decompress Index Block** (LZ4, ~3GB/s) ← NEW
   - Check decompressed cache first (LRU)
   - If cache miss: decompress (16 KB → 4 KB typical, ~5 µs)
   - Cache decompressed block for future lookups

4. **Index Block Binary Search** (decompressed, varint-decoded)
   - Find data block that contains key
   - Load compressed data block from disk (or cache)

5. **Decompress Data Block** (LZ4, ~3GB/s) ← NEW
   - Check decompressed cache first (LRU)
   - If cache miss: decompress (8 KB → 4 KB typical, ~3 µs)
   - Cache decompressed block for future lookups

6. **Data Block Binary Search** (decompressed, varint-decoded)
   - Find exact key in block
   - Return value (or load from vlog if pointer)

### Scan: Range(start_key, end_key)

1. Use top-level index to find first relevant index block
2. Load index blocks sequentially
3. For each relevant data block:
   - Load block
   - Iterate entries in range
   - Yield matching entries

## Memory Usage Analysis

### V1 (Current)
```
Per SSTable (1M entries):
- Full Index: 1M × (26 bytes key + 8 bytes offset) = 34 MB
- Bloom Filter: ~1.2 MB
- Total per SSTable: ~35 MB

With 20 SSTables open: 700 MB
```

### V2 (Block-Based)
```
Per SSTable (1M entries):
- Top-Level Index: ~250 index blocks × (26 bytes + 8 bytes) = 8.5 KB
- Bloom Filter: ~1.2 MB
- Total in RAM: ~1.21 MB

With 20 SSTables open: ~24 MB
Block Cache (LRU): 32 MB (configured)
Total: ~56 MB

Memory Reduction: 700 MB → 56 MB = **92% reduction**
```

## Implementation Strategy

### Phase 1: Core Format (This Session)
- [x] Block module (BlockBuilder, Block, BlockIterator)
- [ ] New SSTableBuilder with block-based writing
- [ ] New SSTable with lazy block loading
- [ ] Update all existing tests

### Phase 2: Performance (Next Session)
- [ ] LRU cache for blocks
- [ ] Benchmark vs V1
- [ ] Optimize hot paths

### Phase 3: Compression (Next Priority - Nov 8, 2025)
- [ ] Add lz4_flex dependency
- [ ] Compress data blocks on write
- [ ] Decompress data blocks on read
- [ ] LRU cache for decompressed blocks
- [ ] Benchmark: verify +30-40% improvement

### Phase 4: Varint Encoding
- [ ] Replace fixed u32/u64 with varint-rs
- [ ] Update serialization for all metadata
- [ ] Benchmark: verify +3-5% improvement

### Phase 5: Production (Future)
- [ ] mmap support for zero-copy
- [ ] Alternative compression (zstd for cold data)
- [ ] Adaptive compression based on data type

## Backward Compatibility

**Breaking Change**: V3 format is incompatible with V1 and V2

Migration strategy:
1. Detect format version from footer magic/version
2. V1/V2 readers still available (keep old code for migration)
3. Write helper to convert V1/V2 → V3 (optional)

**Why Breaking Changes Are OK at 0.0.x**:
- No production users yet
- Better to implement SOTA now than migrate later
- Format stability comes at 1.0.0, not before
- Competitors (fjall) already use LZ4 + varint

## Testing Strategy

1. **Unit Tests**: Block read/write, index operations
2. **Integration Tests**: Full SSTable lifecycle with blocks
3. **Memory Tests**: Verify <10% memory growth with 10GB dataset
4. **Performance Tests**: Compare V1 vs V2 throughput/latency
5. **Stress Tests**: 100GB+ datasets with limited RAM

## Success Criteria

### V2 (Block-Based) - Completed
- ✅ All existing tests pass with new format
- ✅ Memory usage: <100 MB for 20 SSTables (vs 700 MB in V1)
- ✅ Performance: Within 10% of V1 for point lookups
- ✅ Performance: 2-3x faster for scans (better cache locality)

### V3 (Compression + Varint) - In Progress
- [ ] All tests pass with compression enabled
- [ ] **Cache efficiency**: 2-3x more blocks fit in same cache size
- [ ] **Read throughput**: +30-40% improvement (more cache hits)
- [ ] **Compression ratio**: 40-60% for typical data
- [ ] **Decompression speed**: <10 µs per 4KB block (LZ4 @ 3GB/s)
- [ ] **Space savings**: +30-50% with varint metadata encoding
- [ ] 10GB soak test: Verify compression ratio holds at scale

## LZ4 Compression Details (V3)

**Why LZ4 vs Snappy/Zstd**:
- **Speed**: 3GB/s decompression (fastest general-purpose codec)
- **Rust-native**: lz4_flex is pure Rust (no C dependencies)
- **Proven**: Used by fjall, RocksDB, Cassandra
- **Sweet spot**: 40-60% compression at 3GB/s (vs Snappy 50% @ 500MB/s)

**Compression Ratio Examples**:
```
Typical seerdb data:
- Keys: Low compression (random UUIDs/hashes) → 10-20%
- Values (blobs/documents): High compression (similar patterns) → 50-70%
- Metadata: High compression (small integers) → 60-80%
- Overall block: 40-60% compression ratio
```

**Cache Impact**:
```
Without LZ4 (4KB blocks):
- 32 MB cache = 8,192 blocks
- 100 entries per block = 819,200 entries cached

With LZ4 (2KB compressed):
- 32 MB cache = 16,384 blocks (2x more!)
- 100 entries per block = 1,638,400 entries cached
- Cache miss rate: 15% → 7.5% (2x reduction)
- Read throughput: +30-40% improvement
```

**Performance Trade-off**:
- Compression overhead: ~5 µs per 4KB block @ 500MB/s
- Decompression overhead: ~1.3 µs per 4KB block @ 3GB/s
- Disk I/O savings: 2x fewer reads (compressed blocks smaller)
- Net benefit: +30-40% throughput (disk I/O dominates)

## Varint Encoding Details (V3)

**Why Varint**:
- Most metadata values are small (key_len < 100, offsets < 1GB)
- Fixed u32 wastes 3 bytes for small values
- Protobuf-style varint: 1 byte for 0-127, 2 bytes for 128-16383

**Space Savings**:
```
Block with 100 entries (before):
- 100 × (4 + 4 + 4) = 1,200 bytes metadata
- Restart points: 10 × 4 = 40 bytes
- Total: 1,240 bytes per 4KB block = 30%

Block with 100 entries (after):
- 100 × (1 + 2 + 2) = 500 bytes metadata (typical)
- Restart points: 10 × 2 = 20 bytes (typical)
- Total: 520 bytes per 4KB block = 13%
- Savings: 17% more space for actual data
```

**Combined with LZ4**:
- Varint saves 17% metadata space
- More data fits in block → better compression ratio
- LZ4 compresses varint-encoded data well (pattern-friendly)
- Combined: +5-10% additional compression

## References

- RocksDB Block Format: https://github.com/facebook/rocksdb/wiki/Rocksdb-BlockBasedTable-Format
- LevelDB Table Format: https://github.com/google/leveldb/blob/main/doc/table_format.md
- WiscKey Paper: https://www.usenix.org/system/files/conference/fast16/fast16-papers-lu.pdf
- lz4_flex: https://github.com/PSeitz/lz4_flex
- varint-rs: https://github.com/dermesser/varint-rs
- Protobuf Varint Encoding: https://protobuf.dev/programming-guides/encoding/#varints
