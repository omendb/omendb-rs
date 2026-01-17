//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod hnsw_index;
pub mod store;
pub mod types;

// Re-export main types
pub use hnsw_index::{HNSWIndex, HNSWIndexBuilder, HNSWQuantization};
pub use store::{
    MetadataFilter, SearchResult, ThreadSafeVectorStore, VectorStore, VectorStoreOptions,
};
pub use types::Vector;

/// Quantization mode for vector storage.
///
/// Controls how vectors are compressed for memory/disk efficiency.
#[derive(Debug, Clone)]
pub enum QuantizationMode {
    /// Scalar Quantization (SQ8): f32 -> u8
    /// - 4x compression
    /// - ~2x faster than f32
    /// - ~99% recall
    SQ8,
}

impl QuantizationMode {
    /// SQ8 quantization (4x compression)
    #[must_use]
    pub fn sq8() -> Self {
        Self::SQ8
    }

    /// Check if SQ8 mode
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        matches!(self, Self::SQ8)
    }
}
