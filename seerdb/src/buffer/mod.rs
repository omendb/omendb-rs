pub mod eviction;
pub mod manager;

pub use eviction::{EvictionPolicy, LruPolicy};
pub use manager::{BufferPool, BufferPoolOptions, FrameRef, PageId};

use thiserror::Error;

/// BufferPool error types
#[derive(Debug, Error)]
pub enum BufferPoolError {
    #[error("Buffer pool is full (all frames pinned or in use)")]
    PoolFull,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
