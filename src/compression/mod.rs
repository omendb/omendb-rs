//! Vector compression for `OmenDB` storage
//!
//! Quantization modes:
//! - SQ8: 4x compression, ~99% recall (default when quantization enabled)
//! - RaBitQ: 32x compression with unbiased distance estimation, >95% recall with rescore

pub mod rabitq;
pub mod scalar;

pub use rabitq::{RaBitQParams, RaBitQQueryPrep};
pub use scalar::{symmetric_l2_squared_u8, QueryPrep, ScalarParams};
