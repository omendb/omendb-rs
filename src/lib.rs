#![feature(portable_simd)]
#![warn(clippy::pedantic)]
#![allow(
    // Naming
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::many_single_char_names, // FHT algorithm uses standard math notation (n, h, i, j, a, b)
    // Casts - numeric conversions are validated at API boundaries
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    // Documentation - errors/panics are clear from context
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Design choices
    clippy::unsafe_derive_deserialize, // Serde derive is safe, unsafe methods are for SIMD/RNG
    clippy::too_many_lines,            // Complex functions (batch_insert, load_from_disk) are well-structured
    clippy::needless_pass_by_value     // Public API takes owned values for clarity and storage
)]

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

// Core modules
pub mod compression;
pub mod config;
pub mod sampling;
pub mod simd;
pub mod storage;
pub mod types;
pub mod vector;

#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export core types
pub use vector::{MetadataFilter, Vector, VectorStore};

// Re-export storage types (formerly seerdb-vector)
pub use config::SeerdbVectorConfig;
pub use types::{CompressionTier, DistanceMetric, Result, SeerdbVectorError, StorageTier};
