// Custom HNSW implementation for OmenDB
//
// Design goals:
// - Cache-optimized (64-byte aligned hot data)
// - Memory-efficient (flattened index with u32 node IDs)
// - SIMD-ready (AVX2/AVX512 distance calculations)
// - SOTA features support (Extended RaBitQ, delta encoding)

mod types;
mod storage;
mod node_storage;
mod disk_storage;
mod cached_storage;
mod storage_tiering;
mod layered_storage;
mod graph_storage;
mod storage_integration_tests;
mod index;
mod simd_distance;
mod error;
mod query_buffers;
mod merge;

// Public API exports
pub use types::{
    HNSWParams, HNSWNode, DistanceFunction, Candidate, SearchResult,
};

// Re-export SIMD-enabled distance functions
pub use simd_distance::{l2_distance, cosine_distance, dot_product};

pub use storage::{NeighborLists, VectorStorage};

pub use node_storage::{NodeStorage, MemoryStorage, NodeId, Level};

pub use disk_storage::{DiskStorage, WritableDiskStorage};

pub use cached_storage::CachedStorage;

pub use storage_tiering::{StorageMode, TieringConfig};

pub use layered_storage::LayeredStorage;

pub use graph_storage::{GraphStorage, DiskConfig};

pub use index::{HNSWIndex, IndexStats};

// Re-export error types
pub use error::{HNSWError, Result};

// Re-export graph merging
pub use merge::{GraphMerger, MergeConfig, MergeStats};
