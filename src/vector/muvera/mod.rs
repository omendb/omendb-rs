//! MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings)
//!
//! Transforms variable-length multi-vector sets into fixed-dimensional encodings (FDEs).
//! The inner product of two FDEs approximates Chamfer/MaxSim similarity.
//!
//! # Algorithm
//!
//! FDE dimension = r_reps × 2^k_sim × d_proj
//!
//! - `k_sim`: Number of SimHash hyperplanes (creates 2^k_sim buckets)
//! - `r_reps`: Independent repetitions concatenated
//! - `d_proj`: Projection dimension (default: 16, matching Weaviate/Qdrant)
//!
//! Tokens are projected from token_dim → d_proj before encoding, reducing FDE size
//! by 8x (e.g., 16,384D → 2,048D for 128D tokens with d_proj=16).
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
//! let config = MuveraConfig::default();  // k_sim=4, r_reps=8, d_proj=16
//! let encoder = MuveraEncoder::new(128, config);  // 128D tokens -> 2,048D FDE
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
mod pooling;
mod storage;

pub use config::{MultiVectorConfig, MuveraConfig}; // MuveraConfig is alias for backwards compat
pub use encoder::{maxsim, maxsim_batch, maxsim_batch_par, AggMode, MuveraEncoder};
pub use pooling::pool_tokens;
pub use storage::MultiVecStorage;
