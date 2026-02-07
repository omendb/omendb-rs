//! Vector compression for `OmenDB` storage
//!
//! Quantization modes:
//! - SQ8: 4x compression, ~99% recall (default when quantization enabled)
//! - PQ: 16-64x compression for 768D+ vectors, ~95% recall with rescore
//! - RaBitQ: 32x compression with unbiased distance estimation, >95% recall with rescore

pub mod product;
pub mod rabitq;
pub mod scalar;

pub use product::{PQParams, PQQueryPrep};
pub use rabitq::{RaBitQParams, RaBitQQueryPrep};
pub use scalar::{symmetric_l2_squared_u8, QueryPrep, ScalarParams};
