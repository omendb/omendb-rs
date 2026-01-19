// Custom HNSW implementation for OmenDB
//
// Design goals:
// - Cache-optimized (64-byte aligned hot data)
// - Memory-efficient (flattened index with u32 node IDs)
// - SIMD-ready (AVX2/AVX512 distance calculations)
// - SQ8 scalar quantization support (4x compression)
// - Parallel construction with fine-grained locking

mod atomic_bitvec;
mod batch_builder;
mod error;
mod graph_storage;
mod index;
mod merge;
mod node_storage;
mod prefetch;
mod query_buffers;
mod segment;
mod segment_manager;
mod segment_persistence;
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

pub use index::{HNSWIndex, IndexStats, ParallelBuilder};

// Re-export error types
pub use error::{HNSWError, Result};

// Re-export graph merging
pub use merge::{GraphMerger, MergeConfig, MergeStats};

// V1 architecture - unified node storage and segments
pub use batch_builder::{BatchBuilder, Cluster};
pub use node_storage::NodeStorage;
pub use segment::{FrozenSegment, MutableSegment, SegmentSearchResult};
pub use segment_manager::{MergePolicy, SegmentConfig, SegmentManager};
