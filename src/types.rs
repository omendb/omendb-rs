//! Core types for `OmenDB` storage layer

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Vector ID type (globally unique identifier)
pub type VectorID = u64;

/// Compression tier for vector storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionTier {
    /// Full precision (f32) - L0-L2 hot tier
    Full,
    /// `RaBitQ` 4-bit (8× compression) - L3-L4 warm tier
    RaBitQ4Bit,
    /// `RaBitQ` 2-bit (16× compression) - L5-L6 cold tier
    RaBitQ2Bit,
}

impl CompressionTier {
    /// Get compression ratio vs f32
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        match self {
            CompressionTier::Full => 1.0,
            CompressionTier::RaBitQ4Bit => 8.0,
            CompressionTier::RaBitQ2Bit => 16.0,
        }
    }

    /// Expected recall for this tier
    #[must_use]
    pub fn expected_recall(&self) -> f32 {
        match self {
            CompressionTier::Full => 1.0,
            CompressionTier::RaBitQ4Bit => 0.98,
            CompressionTier::RaBitQ2Bit => 0.95,
        }
    }

    /// Get bits per dimension
    #[must_use]
    pub fn bits_per_dim(&self) -> u8 {
        match self {
            CompressionTier::Full => 32,
            CompressionTier::RaBitQ4Bit => 4,
            CompressionTier::RaBitQ2Bit => 2,
        }
    }
}

/// Storage tier (corresponds to LSM levels)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StorageTier {
    /// L0: In-memory HNSW (RAM)
    L0,
    /// L1: Hot SSD (full precision)
    L1,
    /// L2: Warm SSD (full precision)
    L2,
    /// L3: Warm SSD (4-bit `RaBitQ`)
    L3,
    /// L4: Cold SSD (4-bit `RaBitQ`)
    L4,
    /// L5: Cold S3 (2-bit `RaBitQ`)
    L5,
    /// L6: Archive S3 (2-bit `RaBitQ`)
    L6,
}

impl StorageTier {
    /// Get compression tier for this storage tier
    #[must_use]
    pub fn compression(&self) -> CompressionTier {
        match self {
            StorageTier::L0 | StorageTier::L1 | StorageTier::L2 => CompressionTier::Full,
            StorageTier::L3 | StorageTier::L4 => CompressionTier::RaBitQ4Bit,
            StorageTier::L5 | StorageTier::L6 => CompressionTier::RaBitQ2Bit,
        }
    }

    /// Is this tier in memory (L0)?
    #[must_use]
    pub fn is_memory(&self) -> bool {
        matches!(self, StorageTier::L0)
    }

    /// Is this tier on SSD?
    #[must_use]
    pub fn is_ssd(&self) -> bool {
        matches!(
            self,
            StorageTier::L1 | StorageTier::L2 | StorageTier::L3 | StorageTier::L4
        )
    }

    /// Is this tier on S3?
    #[must_use]
    pub fn is_s3(&self) -> bool {
        matches!(self, StorageTier::L5 | StorageTier::L6)
    }
}

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

/// Search result from vector database
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Vector ID
    pub id: VectorID,
    /// Distance to query vector
    pub distance: f32,
    /// Optional metadata
    pub metadata: Option<Vec<u8>>,
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
