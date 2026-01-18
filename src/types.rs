//! Core types for `OmenDB` storage layer

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Vector ID type (globally unique identifier)
pub type VectorID = u64;

/// Distance metric for vector comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance
    L2,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Inner product (dot product)
    InnerProduct,
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
