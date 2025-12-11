//! Vector compression for `OmenDB` storage
//!
//! Provides multiple compression methods:
//! - Scalar (SQ8): 4x compression, ~98% recall
//! - Binary (RaBitQ): 32x compression, ~95% recall
//! - Product (PQ): 64x compression, ~90% recall

pub mod rabitq;
pub mod scalar;

pub use rabitq::{
    ADCTable, QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams, TrainedParams,
};
pub use scalar::{SQ8ADCTable, ScalarParams};
