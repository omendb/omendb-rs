//! Vector storage with HNSW indexing for approximate nearest neighbor search.

pub mod hnsw;
pub mod hnsw_index;
pub mod metadata;
pub mod muvera;
pub mod store;
pub mod types;

// Re-export main types
pub use hnsw_index::{HNSWIndex, HNSWIndexBuilder, HNSWQuantization};
pub use metadata::{FieldIndex, Filter, FilterValue, MetadataIndex};
pub use store::{
    MetadataFilter, SearchResult, ThreadSafeVectorStore, VectorStore, VectorStoreOptions,
};
pub use types::Vector;
