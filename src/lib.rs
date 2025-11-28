#![feature(portable_simd)]

//! Embedded vector database with HNSW indexing.
//!
//! # Example
//!
//! ```rust,no_run
//! use omendb::vector::{Vector, VectorStore};
//!
//! let mut store = VectorStore::new(128); // 128 dimensions
//! store.insert(Vector::new(vec![0.1; 128])).unwrap();
//!
//! let results = store.knn_search(&Vector::new(vec![0.1; 128]), 10).unwrap();
//! ```

pub mod vector;

#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export core types
pub use vector::{MetadataFilter, Vector, VectorStore};

// Re-export seerdb-vector types
pub use seerdb_vector::{config::SeerdbVectorConfig, CompressionTier, DistanceMetric, StorageTier};
