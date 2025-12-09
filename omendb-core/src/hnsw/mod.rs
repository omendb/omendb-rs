// Custom HNSW implementation for OmenDB
//
// Design goals:
// - Cache-optimized (64-byte aligned hot data)
// - Memory-efficient (flattened index with u32 node IDs)
// - SIMD-ready (AVX2/AVX512 distance calculations)
// - SOTA features support (Extended RaBitQ, delta encoding)

mod cached_storage;
mod disk_storage;
mod error;
mod graph_storage;
mod index;
mod layered_storage;
mod merge;
mod node_storage;
mod query_buffers;
mod storage;
mod storage_integration_tests;
mod storage_tiering;
mod types;

// Public API exports
pub use types::{Candidate, DistanceFunction, HNSWNode, HNSWParams, SearchResult};

// Re-export SIMD-enabled distance functions (single source of truth)
pub use crate::distance::{cosine_distance, dot_product, l2_distance};

pub use storage::{NeighborLists, VectorStorage};

pub use node_storage::{Level, MemoryStorage, NodeId, NodeStorage};

pub use disk_storage::{DiskStorage, WritableDiskStorage};

pub use cached_storage::CachedStorage;

pub use storage_tiering::{StorageMode, TieringConfig};

pub use layered_storage::LayeredStorage;

pub use graph_storage::{DiskConfig, GraphStorage};

pub use index::{HNSWIndex, IndexStats};

// Re-export error types
pub use error::{HNSWError, Result};

// Re-export graph merging
pub use merge::{GraphMerger, MergeConfig, MergeStats};
