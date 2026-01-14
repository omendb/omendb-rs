//! Vector compression for `OmenDB` storage
//!
//! Production quantization mode:
//! - SQ8: 4x compression, ~99% recall (default when quantization enabled)

pub mod scalar;

pub use scalar::{symmetric_l2_squared_u8, QueryPrep, ScalarParams};
