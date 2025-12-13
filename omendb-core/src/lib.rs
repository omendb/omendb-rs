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
    clippy::manual_let_else            // Match pattern is clearer in some contexts
)]

//! Core algorithms for `OmenDB`: HNSW, `RaBitQ` compression, SIMD distance functions.
//!
//! This crate contains the pure algorithmic components of `OmenDB`, without any
//! storage or I/O dependencies. It can be used independently for:
//!
//! - Vector quantization (`RaBitQ`)
//! - SIMD-accelerated distance computation
//! - HNSW graph construction and search
//!
//! # Architecture
//!
//! ```text
//! omendb-core (this crate)     omendb (full database)
//! ├── compression/             ├── depends on omendb-core
//! ├── distance/                ├── omen/ (.omen storage)
//! ├── hnsw/                    ├── text/ (tantivy)
//! └── types.rs                 └── vector/store/
//! ```

// Core modules
pub mod compression;
pub mod distance;
pub mod hnsw;
pub mod types;

// Re-export core types
pub use types::{
    CompactionStats, CompressionTier, DistanceMetric, OmenDBError, Result, SearchResult,
    StorageTier, VectorID,
};

// Re-export compression types
pub use compression::{ADCTable, QuantizedVector, RaBitQ, RaBitQParams};

// Re-export distance functions
pub use distance::{cosine_distance, dot_product, l2_distance, l2_distance_squared};

// Re-export HNSW types
pub use hnsw::{HNSWIndex, HNSWParams};
