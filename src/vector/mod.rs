//! Vector database module
//!
//! Provides vector storage with HNSW (Hierarchical Navigable Small World) indexing
//! for approximate nearest neighbor search.
//!
//! Week 2 Goal: Production-ready HNSW implementation
//! - Recall@10: >95%
//! - Latency: <10ms p95
//! - Memory: <200 bytes/vector overhead

pub mod types;
pub mod store;
pub mod storage; // seerdb persistent storage backend
pub mod hnsw_index;
pub mod vector_value;
pub mod custom_hnsw; // Internal implementation
pub mod extended_rabitq; // Extended RaBitQ quantization (SIGMOD 2025)

// Re-export main types
pub use types::Vector;
pub use store::{VectorStore, MetadataFilter};
pub use hnsw_index::HNSWIndex;
pub use vector_value::VectorValue;
pub use extended_rabitq::{ExtendedRaBitQ, ExtendedRaBitQParams, QuantizationBits, QuantizedVector};
