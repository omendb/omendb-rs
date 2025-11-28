#![feature(portable_simd)] // Required for SIMD distance calculations

//! seerdb-vector: Vector-optimized storage layer
//!
//! This crate provides a vector-optimized storage layer built on top of seerdb.
//! It is the core storage engine for OmenDB, providing:
//!
//! - **Tiered vector storage**: L0-L6 with RaBitQ compression
//! - **Graph edge persistence**: HNSW/Vamana graph edges
//! - **SIMD distance calculations**: AVX-512/AVX2 optimization
//! - **Crash recovery**: WAL-based durability guarantees
//!
//! # Architecture
//!
//! ```text
//! OmenDB Vector Store API
//!        ↓
//! HNSW Index + ACORN-1 Filtering
//!        ↓
//! seerdb-vector Storage Layer (THIS CRATE)
//!        ↓
//! seerdb Core LSM Engine
//! ```
//!
//! # Compression Tiers
//!
//! ```text
//! L0-L2: Full precision (f32) - Hot tier
//! L3-L4: RaBitQ 4-bit (8× compression, 98% recall) - Warm tier
//! L5-L6: RaBitQ 2-bit (16× compression, 95% recall) - Cold tier
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use seerdb_vector::{VectorConfig, StorageTier};
//!
//! // Create vector database configuration
//! let config = VectorConfig::default()
//!     .with_dimensions(128)
//!     .with_compression_tiers(true);
//! ```

pub mod compression; // RaBitQ compression (Phase 2)
pub mod config;
pub mod edge_storage;
pub mod node_metadata;
pub mod sampling;
pub mod simd_distance; // SIMD-accelerated distance calculations (Phase 1)
pub mod types;
pub mod vector_metadata; // Sampling-based search (Phase 2b)

// Future modules (to be implemented):
// pub mod vector_storage;  // Vector data with SIMD alignment

pub use compression::{QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams};
pub use config::*;
pub use edge_storage::EdgeStorage;
pub use node_metadata::NodeMetadataStorage;
pub use types::*;
pub use vector_metadata::VectorMetadataStorage;

// Re-export sampling functions
pub use sampling::{
    compute_hash, count_collisions, default_threshold, init_random_projections, should_read_vector,
    HASH_BITS, HASH_BYTES,
};
