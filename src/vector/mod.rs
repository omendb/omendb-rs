//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod hnsw_index;
pub mod storage;
pub mod store;
pub mod types;

// Re-export main types
pub use hnsw_index::HNSWIndex;
pub use omendb_core::compression::{QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams};
pub use store::{MetadataFilter, VectorStore, VectorStoreOptions};
pub use types::Vector;
