//! Core types for `OmenDB` storage layer

use crate::distance::{
    cosine_distance, cosine_distance_precomputed, dot_product, l2_distance, l2_distance_squared,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Vector ID type (globally unique identifier)
pub type VectorID = u64;

/// Distance metric for similarity search.
///
/// Canonical enum used throughout the codebase for distance computation,
/// file format headers, persistence (postcard), and public API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Metric {
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
}

impl Metric {
    const L2_ROUNDOFF_EPSILON: f32 = 1.0e-5;

    /// Parse from string
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" | "cos" => Ok(Self::Cosine),
            "ip" | "inner_product" | "dot" => Ok(Self::InnerProduct),
            _ => Err(format!("Unknown metric: {s}")),
        }
    }

    /// Canonical public API name for this metric.
    #[must_use]
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "dot",
        }
    }

    /// Compute distance between two vectors
    #[must_use]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::InnerProduct => -dot_product(a, b),
        }
    }

    /// Compute distance for internal comparisons (optimized, may not be actual distance)
    ///
    /// For L2: returns squared distance (skips sqrt) since relative ordering is preserved.
    /// For Cosine/InnerProduct: same as `distance()`.
    #[inline(always)]
    #[must_use]
    pub fn distance_for_comparison(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => l2_distance_squared(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::InnerProduct => -dot_product(a, b),
        }
    }

    /// Compute distance for comparisons with precomputed query norm
    ///
    /// Like `distance_for_comparison` but avoids redundant query norm computation
    /// for cosine distance. L2 and InnerProduct ignore the norm parameter.
    #[inline(always)]
    #[must_use]
    pub fn distance_for_comparison_precomputed(&self, a: &[f32], b: &[f32], a_norm: f32) -> f32 {
        match self {
            Self::L2 => l2_distance_squared(a, b),
            Self::Cosine => cosine_distance_precomputed(a, b, a_norm),
            Self::InnerProduct => -dot_product(a, b),
        }
    }

    /// Convert comparison distance back to actual distance
    ///
    /// For L2: applies sqrt. For others: identity.
    #[inline]
    #[must_use]
    pub fn comparison_to_actual(&self, d: f32) -> f32 {
        match self {
            // Decomposed L2 can drift slightly around zero from floating-point error.
            // Clamp tiny values so user-facing distances stay finite and exact-match
            // results remain stable at 0.0.
            Self::L2 => {
                if d <= Self::L2_ROUNDOFF_EPSILON {
                    0.0
                } else {
                    d.sqrt()
                }
            }
            _ => d,
        }
    }
}

impl From<u8> for Metric {
    fn from(v: u8) -> Self {
        match v {
            1 => Self::Cosine,
            2 => Self::InnerProduct,
            _ => Self::L2,
        }
    }
}

/// Statistics for compaction operation
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionStats {
    /// Number of segments before compaction
    pub segments_before: usize,
    /// Number of segments after compaction
    pub segments_after: usize,
    /// Number of vectors compacted
    pub vectors_compacted: usize,
    /// Number of edges written
    pub edges_written: usize,
    /// Duration of compaction in seconds
    pub duration_secs: f64,
}

/// `OmenDB` error types
#[derive(Debug, Error)]
pub enum OmenDBError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid dimension
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Vector not found
    #[error("Vector not found: {0}")]
    VectorNotFound(VectorID),

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Storage backend error
    #[error("Storage backend error: {0}")]
    Backend(String),

    /// Invalid data format
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Result type for `OmenDB` operations
pub type Result<T> = std::result::Result<T, OmenDBError>;

#[cfg(test)]
mod tests {
    use super::Metric;

    #[test]
    fn l2_comparison_to_actual_clamps_negative_roundoff() {
        let metric = Metric::L2;
        assert_eq!(metric.comparison_to_actual(-1.0e-6), 0.0);
        assert_eq!(metric.comparison_to_actual(1.0e-6), 0.0);
        assert_eq!(metric.comparison_to_actual(4.0), 2.0);
    }
}
