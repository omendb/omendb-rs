//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod hnsw_index;
pub mod store;
pub mod types;

// Re-export main types
pub use crate::compression::{CodeMetadata, QueryLUT, RaBitQIndex};
pub use crate::compression::{QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams};
pub use hnsw_index::{HNSWIndex, HNSWIndexBuilder, HNSWQuantization};
pub use store::{MetadataFilter, VectorStore, VectorStoreOptions};
pub use types::Vector;

/// Quantization mode for vector storage
///
/// Controls how vectors are compressed for memory/disk efficiency.
#[derive(Debug, Clone)]
pub enum QuantizationMode {
    /// Binary Quantization (BBQ): f32 → 1 bit (DEPRECATED, use RaBitQ)
    /// - 32x compression
    /// - 2-4x faster than SQ8 (SIMD Hamming)
    /// - ~85% raw recall, ~95-98% with rescore
    #[deprecated(since = "0.0.24", note = "Use RaBitQ for better accuracy")]
    Binary,

    /// Scalar Quantization (SQ8): f32 → u8
    /// - 4x compression
    /// - ~2x faster than f32 (direct SIMD)
    /// - ~99% recall with rescore
    SQ8,

    /// Legacy multi-bit scalar quantization: f32 → 2-8 bits (DEPRECATED)
    /// - 4-16x compression
    /// - ~0.5x slower than f32 (ADC lookup tables)
    /// - 93-99% recall depending on bits
    #[deprecated(
        since = "0.0.24",
        note = "Use SQ8 for 4x compression or RaBitQ for 32x"
    )]
    LegacyMultiBit(RaBitQParams),

    /// RaBitQ: f32 → 1 bit with FFHT rotation
    /// - 32x compression (28x with metadata)
    /// - ~95% recall with rescore
    /// - Best for memory-constrained, >100K vectors
    RaBitQ,
}

impl QuantizationMode {
    /// Create Binary quantization mode (32x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use rabitq() instead")]
    #[allow(deprecated)]
    pub fn binary() -> Self {
        Self::Binary
    }

    /// Create SQ8 quantization mode (4x compression, fastest)
    #[must_use]
    pub fn sq8() -> Self {
        Self::SQ8
    }

    /// Create RaBitQ mode (32x compression with FFHT rotation)
    #[must_use]
    pub fn rabitq() -> Self {
        Self::RaBitQ
    }

    /// Alias for rabitq() - kept for transition
    #[must_use]
    pub fn true_rabitq() -> Self {
        Self::RaBitQ
    }

    /// Create legacy multi-bit with 4-bit quantization (8x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use sq8() or rabitq() instead")]
    #[allow(deprecated)]
    pub fn legacy_multibit_4() -> Self {
        Self::LegacyMultiBit(RaBitQParams::bits4())
    }

    /// Create legacy multi-bit with 2-bit quantization (16x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use rabitq() instead")]
    #[allow(deprecated)]
    pub fn legacy_multibit_2() -> Self {
        Self::LegacyMultiBit(RaBitQParams::bits2())
    }

    /// Create legacy multi-bit with 8-bit quantization (4x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use sq8() instead")]
    #[allow(deprecated)]
    pub fn legacy_multibit_8() -> Self {
        Self::LegacyMultiBit(RaBitQParams::bits8())
    }

    /// Check if this is Binary mode
    #[must_use]
    #[allow(deprecated)]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary)
    }

    /// Check if this is SQ8 mode
    #[must_use]
    pub fn is_sq8(&self) -> bool {
        matches!(self, Self::SQ8)
    }

    /// Check if this is legacy multi-bit mode
    #[must_use]
    #[allow(deprecated)]
    pub fn is_legacy_multibit(&self) -> bool {
        matches!(self, Self::LegacyMultiBit(_))
    }

    /// Check if this is RaBitQ mode (FFHT rotation)
    #[must_use]
    pub fn is_rabitq(&self) -> bool {
        matches!(self, Self::RaBitQ)
    }
}
