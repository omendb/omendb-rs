//! MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings)
//!
//! Transforms variable-length multi-vector sets into fixed-dimensional encodings (FDEs).
//! The inner product of two FDEs approximates Chamfer/MaxSim similarity.
//!
//! # Algorithm
//!
//! FDE dimension = r_reps × 2^k_sim × token_dim
//!
//! - `k_sim`: Number of SimHash hyperplanes (creates 2^k_sim buckets)
//! - `r_reps`: Independent repetitions concatenated
//! - `token_dim`: Original token embedding dimension
//!
//! # Asymmetric Encoding
//!
//! Queries use SUM aggregation, documents use AVERAGE aggregation.
//! This asymmetry preserves Chamfer similarity semantics.
//!
//! # Example
//!
//! ```ignore
//! use omendb::vector::muvera::{MuveraConfig, MuveraEncoder};
//!
//! let config = MuveraConfig::default();  // k_sim=4, r_reps=8
//! let encoder = MuveraEncoder::new(128, config);  // 128D tokens -> 16,384D FDE
//!
//! let doc_fde = encoder.encode_document(&doc_tokens);
//! let query_fde = encoder.encode_query(&query_tokens);
//!
//! let similarity = dot(&query_fde, &doc_fde);  // Approximates MaxSim
//! ```
//!
//! # References
//!
//! - [MUVERA Paper](https://arxiv.org/abs/2405.19504)
//! - [Weaviate Implementation](https://weaviate.io/blog/muvera)
//! - [Qdrant MUVERA](https://qdrant.tech/articles/muvera-embeddings/)

mod config;
mod encoder;
mod storage;

pub use config::{MultiVectorConfig, MuveraConfig}; // MuveraConfig is alias for backwards compat
pub use encoder::{maxsim, maxsim_batch, maxsim_batch_par, AggMode, MuveraEncoder};
pub use storage::MultiVecStorage;
