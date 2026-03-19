//! Vector compression for `OmenDB` storage
//!
//! Quantization modes:
//! - SQ8: 4x compression, ~99% recall (default when quantization enabled)

pub mod scalar;

pub use scalar::{QueryPrep, ScalarParams, symmetric_l2_squared_u8};
