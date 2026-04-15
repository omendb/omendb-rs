use thiserror::Error;

pub type Result<T> = std::result::Result<T, HNSWError>;

#[derive(Error, Debug)]
pub enum HNSWError {
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid vector: contains NaN or Inf")]
    InvalidVector,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("Storage error: {0}")]
    Storage(String),
}

impl HNSWError {
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }
}

impl From<postcard::Error> for HNSWError {
    fn from(err: postcard::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}
