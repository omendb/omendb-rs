//! Vector compression for `OmenDB` storage
//!
//! Provides multiple compression methods:
//! - Binary (BBQ): 32x compression, ~85% raw recall (~95% with rescore)
//! - Scalar (SQ8): 4x compression, ~99% recall
//! - RaBitQ: 8x compression, ~98% recall

pub mod binary;
pub mod rabitq;
pub mod scalar;

pub use binary::{hamming_distance, BinaryParams};
pub use rabitq::{
    ADCTable, QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams, TrainedParams,
};
pub use scalar::{SQ8ADCTable, ScalarParams};
