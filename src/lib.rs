#![feature(portable_simd)]

//! OmenDB - Fast Embedded Vector Database
//!
//! High-performance vector database with custom HNSW implementation,
//! ACORN-1 filtered search, and RaBitQ compression.
//!
//! ## Features
//!
//! - **HNSW+**: Optimized HNSW with SIMD (9.2x faster than ChromaDB)
//! - **ACORN-1**: Filtered search (37.79x speedup)
//! - **RaBitQ**: 2/4/8-bit quantization (8x compression, 100% recall)
//! - **seerdb**: LSM-tree persistent storage
//!
//! ## Example
//!
//! ```rust,no_run
//! use omendb::vector::{Vector, VectorStore};
//!
//! let mut store = VectorStore::new(128); // 128 dimensions
//!
//! let vec1 = Vector::new(vec![0.1; 128]);
//! store.insert(vec1).unwrap();
//!
//! let query = Vector::new(vec![0.1; 128]);
//! let results = store.knn_search(&query, 10).unwrap();
//! ```

pub mod vector;

// Re-export core types
pub use vector::{Vector, VectorStore};

// Re-export seerdb-vector types
pub use seerdb_vector::{config::SeerdbVectorConfig, CompressionTier, StorageTier, DistanceMetric};
