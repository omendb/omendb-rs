# seerdb

Research-grade LSM storage engine with learned data structures.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **Experimental**: Not recommended for production use.

Modern embedded storage engine integrating learned indexes (ALEX), key-value separation (WiscKey), and workload-aware compaction (Dostoevsky) from recent systems research.

## Features

- **Learned indexes** (ALEX) for faster lookups
- **Key-value separation** (WiscKey vLog) for lower write amplification
- **OCC transactions** with snapshot isolation
- **Point-in-time snapshots** for consistent reads
- **Range queries** with k-way merge iterator
- **Bulk load API** for fast initial data loading
- **Tiered storage** with S3/GCS cold tier support
- **ZSTD/LZ4 compression**, SIMD, lock-free structures

## Quick Start

```rust
use seerdb::{DB, DBOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = DB::open(DBOptions::default())?;

    // Basic operations
    db.put(b"key1", b"value1")?;
    let val = db.get(b"key1")?;
    db.delete(b"key1")?;

    // Batch writes (atomic)
    let mut batch = db.batch();
    batch.put(b"user:1", b"alice");
    batch.put(b"user:2", b"bob");
    batch.commit()?;

    // Range queries
    for result in db.range(b"user:", Some(b"user:~"))? {
        let (key, value) = result?;
        println!("{:?} = {:?}", key, value);
    }

    // Prefix scans
    for result in db.prefix(b"user:")? {
        let (key, value) = result?;
        println!("{:?} = {:?}", key, value);
    }

    // Point-in-time snapshots
    let snapshot = db.snapshot();
    db.put(b"key1", b"new_value")?;
    // Snapshot still sees old state
    let old_val = snapshot.get(b"key1")?;

    // Full table iteration
    for result in db.iter()? {
        let (key, value) = result?;
        println!("{:?} = {:?}", key, value);
    }

    Ok(())
}
```

## Performance

**Benchmark vs RocksDB** (100K ops, 1KB values, M3 Max):

| Workload | seerdb | RocksDB | Speedup |
|----------|--------|---------|---------|
| **Writes** | 2.15M ops/sec | 356K ops/sec | **6x** |
| **Reads** | 4.1M ops/sec | 1,065K ops/sec | **3.8x** |
| **Batch writes** | 1.9M ops/sec | 345K ops/sec | **5.5x** |

**Write Amplification**: 1.01x (4.82x better than traditional LSM at 4.88x)

Platform: M3 Max (ARM64). See [ai/STATUS.md](ai/STATUS.md) for detailed analysis.

## Getting Started

```bash
# Requires nightly Rust (for std::simd)
rustup override set nightly

# Run all tests
cargo test

# Run baseline benchmark (vs RocksDB)
cargo run --release --features baseline-benchmarks --example baseline_benchmark

# Measure write amplification
cargo run --release --example write_amplification
```

## Testing

- 230 tests (221 lib + 9 integration)
- Memory safety validated (ASAN clean)
- Thread safety validated (50+ concurrent tests)

## Architecture

LSM tree with 7 levels, partitioned skiplist memtables (16 partitions), write-ahead log for durability, SSTable format with ALEX learned indexes, WiscKey vLog for key-value separation, lock-free WAL and cache structures, SIMD key comparison.

See [ai/DECISIONS.md](ai/DECISIONS.md) for design rationale.

## References

- "ALEX: An Updatable Adaptive Learned Index" (Ding et al., 2020)
- "WiscKey: Separating Keys from Values" (Lu et al., 2016)
- "Dostoevsky: Better LSM-Tree Trade-Offs" (Dayan et al., 2018)
- "The Case for Learned Index Structures" (Kraska et al., 2018)

See [ai/research/](ai/research/) for paper summaries and [ai/STATUS.md](ai/STATUS.md) for benchmarks.

## License

[Apache License 2.0](LICENSE)
