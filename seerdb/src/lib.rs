#![cfg_attr(feature = "simd", feature(portable_simd))] // SIMD optimizations (nightly-only)
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

//! seerdb - Research-grade embedded storage engine
//!
//! A modern LSM-tree based key-value storage engine implementing 2018-2024 research
//! on learned data structures, workload-aware optimization, and efficient key-value separation.
//!
//! # Features
//!
//! - **LSM-tree architecture**: Write-optimized with efficient compaction
//! - **Durability**: Write-ahead logging with configurable sync policies
//! - **Concurrency**: Lock-free reads with concurrent writes
//! - **Observability**: Built-in metrics, health checks, and structured logging
//! - **Key-Value Separation**: WiscKey-style vLog for large values (reduces write amplification)
//! - **Background Compaction**: Non-blocking async compaction for better write throughput
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use seerdb::{DB, DBOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open database with default options
//! let db = DB::open(DBOptions::default())?;
//!
//! // Write data
//! db.put(b"hello", b"world")?;
//!
//! // Read data
//! let value = db.get(b"hello")?;
//! assert_eq!(value, Some(bytes::Bytes::from("world")));
//!
//! // Delete data
//! db.delete(b"hello")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Advanced Configuration
//!
//! ```rust,no_run
//! use seerdb::{DB, DBOptions, SyncPolicy};
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let opts = DBOptions {
//!     data_dir: PathBuf::from("./my_database"),
//!     memtable_capacity: 64 * 1024 * 1024,  // 64MB memtable
//!     wal_sync_policy: SyncPolicy::SyncData, // fsync data on each write
//!     background_compaction: true,            // Enable async compaction
//!     vlog_threshold: Some(4096),            // Use vLog for values >4KB
//!     ..Default::default()
//! };
//!
//! let db = DB::open(opts)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! seerdb uses an LSM-tree architecture with the following components:
//!
//! - **Memtable**: In-memory buffer using concurrent skiplist
//! - **WAL**: Write-ahead log for durability
//! - **SSTable**: Sorted string tables on disk with bloom filters
//! - **LSM Levels**: 7 levels with exponential sizing (10x ratio)
//! - **VLog**: Optional value log for key-value separation (large values)
//! - **Compaction**: Background merge of SSTables to reduce read amplification
//!
//! # Performance Characteristics
//!
//! - **Writes**: O(log n) in-memory + O(1) WAL append
//! - **Reads**: O(log n) skiplist + O(levels) SSTable lookups with bloom filter optimization
//! - **Scans**: Efficient via merge iteration over memtable + SSTables
//! - **Space Amplification**: ~2x (typical LSM-tree)
//! - **Write Amplification**: 10-30x (reduced with vLog for large values)
//!
//! # Durability Guarantees
//!
//! seerdb provides configurable durability via [`SyncPolicy`]:
//!
//! - `SyncAll`: fsync both data and metadata (slowest, strongest)
//! - `SyncData`: fsync data only (fast, strong)
//! - `None`: No fsync (fastest, data loss possible on crash)
//!
//! # Observability
//!
//! Built-in metrics and health checks for production deployment:
//!
//! ```rust,ignore
//! # use seerdb::{DB, DBOptions};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let db = DB::open(DBOptions::default())?;
//! // Get current database statistics
//! let stats = db.stats();
//! println!("Operations: {} reads, {} writes", stats.total_reads, stats.total_writes);
//!
//! // Check database health
//! let health = db.health();
//! println!("Health: {:?}", health);
//! # Ok(())
//! # }
//! ```

// Use jemalloc as the global allocator for better multi-threaded performance
// Tested jemalloc vs mimalloc: jemalloc wins 3/4 workloads (+17-21% improvement)
// Disabled when using dhat profiler (conflicts with #[global_allocator])
#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

pub mod alex;
mod background_workers;
pub mod batch;
pub mod bloom;
pub mod buffer;
pub mod compaction;
pub mod db;
mod db_helpers;
pub mod health;
pub mod memtable;
pub mod merge_operator;
pub mod metrics;
pub mod range;
pub mod range_merge;
#[cfg(feature = "simd")]
pub mod simd;
pub mod snapshot;
pub mod sstable;
pub mod storage;
pub mod transaction;
pub mod types;
pub mod vlog;
pub mod wal;

// Re-export main types for convenient access
pub use alex::AlexTree;
pub use batch::Batch;
pub use bloom::{BlockedBloomFilter, BloomFilter, LearnedBloomFilter};
#[cfg(feature = "object-store")]
pub use db::StorageConfig;
pub use db::{BulkLoadOptions, BulkLoadStats, DBError, DBOptions, VerifyResult, DB};
pub use health::{CheckStatus, HealthCheck, HealthStatus};
pub use memtable::Memtable;
pub use merge_operator::{MergeOperator, StringAppendOperator};
pub use metrics::DBStats;
pub use snapshot::Snapshot;
pub use sstable::{CompressionType, SSTable, SSTableBuilder};
pub use storage::LocalStorage;
#[cfg(feature = "object-store")]
pub use storage::{ObjectStoreBackend, RetryConfig, Storage};
pub use transaction::{Transaction, TransactionConflict};
pub use types::{InternalKey, ValueType};
pub use vlog::{VLog, ValuePointer};
pub use wal::{Record, RecoveryMode, SyncPolicy, WAL};

#[cfg(test)]
mod tests {
    #[test]
    fn basic_test() {
        assert_eq!(2 + 2, 4);
    }
}
