// Custom HNSW implementation for OmenDB
//
// Design goals:
// - Cache-optimized (64-byte aligned hot data)
// - Memory-efficient (flattened index with u32 node IDs)
// - SIMD-ready (AVX2/AVX512 distance calculations)
// - SOTA features support (Extended RaBitQ, delta encoding)

mod error;
mod graph_storage;
mod index;
mod merge;
mod prefetch;
mod query_buffers;
mod storage;
mod types;

// Public API exports
pub use types::{Candidate, DistanceFunction, HNSWNode, HNSWParams, SearchResult};

// Export trait-based distance types for monomorphization
pub use types::{Cosine, Distance, NegDot, L2};

// Re-export SIMD-enabled distance functions (single source of truth)
pub use crate::distance::{cosine_distance, dot_product, l2_distance};

pub use storage::{NeighborLists, VectorStorage};

pub use graph_storage::GraphStorage;

pub use index::{HNSWIndex, IndexStats};

// Re-export error types
pub use error::{HNSWError, Result};

// Re-export graph merging
pub use merge::{GraphMerger, MergeConfig, MergeStats};
