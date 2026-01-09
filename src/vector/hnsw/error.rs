//! Error types for HNSW operations

use thiserror::Error;

/// Errors that can occur during HNSW operations
#[derive(Debug, Error)]
pub enum HNSWError {
    /// Vector dimension doesn't match index dimensions
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Vector not found in storage
    #[error("Vector not found: id {0}")]
    VectorNotFound(u32),

    /// Node not found in storage
    #[error("Node not found: id {0}")]
    NodeNotFound(u32),

    /// Invalid level (exceeds `max_levels`)
    #[error("Invalid level: {level} exceeds max_levels {max_levels}")]
    InvalidLevel { level: usize, max_levels: usize },

    /// Index is empty (no vectors inserted yet)
    #[error("Index is empty (no entry point)")]
    EmptyIndex,

    /// Invalid search parameters
    #[error("Invalid search parameters: k={k}, ef={ef}. Requirements: k > 0, ef >= k")]
    InvalidSearchParams { k: usize, ef: usize },

    /// Vector contains invalid values (NaN or Infinity)
    #[error("Vector contains invalid values (NaN or Infinity)")]
    InvalidVector,

    /// Batch size too large or invalid
    #[error("Invalid batch size: {0}. Must be > 0 and < max_elements")]
    InvalidBatchSize(usize),

    /// Storage operation failed
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization failed
    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    /// IO error during save/load
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Parameter validation failed
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    /// Internal consistency error (should never happen)
    #[error("Internal error: {0}. This is a bug, please report it.")]
    Internal(String),
}

/// Result type alias for HNSW operations
pub type Result<T> = std::result::Result<T, HNSWError>;

impl HNSWError {
    /// Create an internal error (for unexpected conditions)
    ///
    /// Marked cold since internal errors should never happen in production.
    #[cold]
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Create a storage error
    #[cold]
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Check if this is a recoverable error
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::DimensionMismatch { .. }
                | Self::InvalidSearchParams { .. }
                | Self::InvalidVector
                | Self::InvalidBatchSize(_)
                | Self::InvalidParams(_)
        )
    }

    /// Check if this indicates a bug in the implementation
    #[must_use]
    pub fn is_internal_bug(&self) -> bool {
        matches!(self, Self::Internal(_) | Self::VectorNotFound(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = HNSWError::DimensionMismatch {
            expected: 128,
            actual: 256,
        };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 128, got 256");

        let err = HNSWError::VectorNotFound(42);
        assert_eq!(err.to_string(), "Vector not found: id 42");

        let err = HNSWError::InvalidSearchParams { k: 0, ef: 100 };
        assert!(err.to_string().contains("k=0, ef=100"));
    }

    #[test]
    fn test_error_classification() {
        let recoverable = HNSWError::DimensionMismatch {
            expected: 128,
            actual: 256,
        };
        assert!(recoverable.is_recoverable());
        assert!(!recoverable.is_internal_bug());

        let bug = HNSWError::Internal("something went wrong".to_string());
        assert!(!bug.is_recoverable());
        assert!(bug.is_internal_bug());
    }
}
