#![feature(portable_simd)]
// Allow retpoline cfg values from multiversion crate's target feature detection
#![allow(unexpected_cfgs)]
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
    clippy::doc_markdown,              // Math notation in docs doesn't need backticks
    // Design choices
    clippy::unsafe_derive_deserialize, // Serde derive is safe, unsafe methods are for SIMD/RNG
    clippy::too_many_lines,            // Complex functions (batch_insert, load_from_disk) are well-structured
    clippy::needless_pass_by_value,    // Public API takes owned values for clarity and storage
    clippy::inline_always,             // Hot path functions are intentionally force-inlined
    clippy::items_after_statements,    // Local items near usage improve readability
    clippy::ptr_as_ptr,                // Raw pointer casts in SIMD code are intentional
    clippy::borrow_as_ptr,             // Borrowing as raw pointer is intentional in prefetch code
    clippy::manual_let_else,           // Match pattern is clearer in some contexts
    clippy::ref_as_ptr,                // Reference to raw pointer is intentional in prefetch code
    clippy::needless_borrow            // Explicit borrows clarify ownership in some contexts
)]

//! Fast embedded vector database with HNSW indexing.
//!
//! # Quick Start
//!
//! ```rust
//! use omendb::{Vector, VectorStore};
//! use serde_json::json;
//!
//! // Create store (128-dimensional vectors)
//! let mut store = VectorStore::new(128);
//!
//! // Insert vectors with metadata
//! store.set("doc1".into(), Vector::new(vec![1.0; 128]), json!({"type": "article"})).unwrap();
//! store.set("doc2".into(), Vector::new(vec![0.9; 128]), json!({"type": "note"})).unwrap();
//!
//! // Search
//! let query = Vector::new(vec![1.0; 128]);
//! let results = store.knn_search(&query, 2).unwrap();
//! // results: [(0, 0.0), (1, 1.13)] - (index, distance)
//!
//! // Get by ID
//! let (vec, metadata) = store.get("doc1").unwrap();
//! ```
//!
//! # Filtered Search (ACORN-1)
//!
//! ```rust
//! use omendb::{MetadataFilter, Vector, VectorStore};
//! use serde_json::json;
//!
//! let mut store = VectorStore::new(64);
//! store.set("a".into(), Vector::new(vec![0.1; 64]), json!({"year": 2024})).unwrap();
//! store.set("b".into(), Vector::new(vec![0.2; 64]), json!({"year": 2023})).unwrap();
//!
//! let query = Vector::new(vec![0.1; 64]);
//! let filter = MetadataFilter::Gte("year".into(), 2024.0);
//! let results = store.knn_search_with_filter(&query, 10, &filter).unwrap();
//! // Only returns vectors where year >= 2024
//! ```
//!
//! # Persistence
//!
//! ```rust,no_run
//! use omendb::VectorStore;
//!
//! // Open or create persistent store
//! let mut store = VectorStore::open_with_dimensions("./vectors", 128).unwrap();
//!
//! // ... insert vectors ...
//!
//! // Save (also auto-saves on drop)
//! store.flush().unwrap();
//!
//! // Reopen later
//! let store = VectorStore::open("./vectors").unwrap();
//! ```

// Core modules
pub mod compression;
pub mod config;
pub mod distance;
pub mod omen;
pub mod text;
pub mod types;
pub mod vector;

#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export core types
pub use compression::{ADCTable, QuantizedVector, QueryPrep, RaBitQ, RaBitQParams, ScalarParams};
pub use distance::{cosine_distance, dot_product, l2_distance, l2_distance_squared};
pub use types::{
    CompactionStats, CompressionTier, DistanceMetric, OmenDBError, Result, SearchResult,
    StorageTier, VectorID,
};
pub use vector::{MetadataFilter, Vector, VectorStore, VectorStoreOptions};

// Re-export storage types
pub use config::StorageConfig;
