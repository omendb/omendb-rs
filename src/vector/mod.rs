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
    /// Binary Quantization (BBQ): f32 → 1 bit (DEPRECATED, use TrueRaBitQ)
    /// - 32x compression
    /// - 2-4x faster than SQ8 (SIMD Hamming)
    /// - ~85% raw recall, ~95-98% with rescore
    #[deprecated(since = "0.0.24", note = "Use TrueRaBitQ for better accuracy")]
    Binary,

    /// Scalar Quantization (SQ8): f32 → u8
    /// - 4x compression
    /// - ~2x faster than f32 (direct SIMD)
    /// - ~99% recall with rescore
    SQ8,

    /// Multi-bit scalar quantization: f32 → 2-8 bits (DEPRECATED, use SQ8 or TrueRaBitQ)
    /// - 4-16x compression
    /// - ~0.5x slower than f32 (ADC lookup tables)
    /// - 93-99% recall depending on bits
    #[deprecated(
        since = "0.0.24",
        note = "Use SQ8 for 4x compression or TrueRaBitQ for 32x"
    )]
    RaBitQ(RaBitQParams),

    /// True RaBitQ: f32 → 1 bit with random rotation
    /// - 32x compression (28x with metadata)
    /// - ~95% recall with rescore
    /// - Best for memory-constrained, >100K vectors
    TrueRaBitQ,
}

impl QuantizationMode {
    /// Create Binary quantization mode (32x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use true_rabitq() instead")]
    #[allow(deprecated)]
    pub fn binary() -> Self {
        Self::Binary
    }

    /// Create SQ8 quantization mode (4x compression, fastest)
    #[must_use]
    pub fn sq8() -> Self {
        Self::SQ8
    }

    /// Create multi-bit RaBitQ with 4-bit quantization (8x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use sq8() or true_rabitq() instead")]
    #[allow(deprecated)]
    pub fn rabitq() -> Self {
        Self::RaBitQ(RaBitQParams::bits4())
    }

    /// Create multi-bit RaBitQ with 2-bit quantization (16x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use true_rabitq() instead")]
    #[allow(deprecated)]
    pub fn rabitq_2bit() -> Self {
        Self::RaBitQ(RaBitQParams::bits2())
    }

    /// Create multi-bit RaBitQ with 8-bit quantization (4x compression)
    #[must_use]
    #[deprecated(since = "0.0.24", note = "Use sq8() instead")]
    #[allow(deprecated)]
    pub fn rabitq_8bit() -> Self {
        Self::RaBitQ(RaBitQParams::bits8())
    }

    /// Create True RaBitQ mode (32x compression with rotation)
    #[must_use]
    pub fn true_rabitq() -> Self {
        Self::TrueRaBitQ
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

    /// Check if this is legacy multi-bit RaBitQ mode
    #[must_use]
    #[allow(deprecated)]
    pub fn is_rabitq(&self) -> bool {
        matches!(self, Self::RaBitQ(_))
    }

    /// Check if this is True RaBitQ mode
    #[must_use]
    pub fn is_true_rabitq(&self) -> bool {
        matches!(self, Self::TrueRaBitQ)
    }
}
