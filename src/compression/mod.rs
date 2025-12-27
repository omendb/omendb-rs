//! Vector compression for `OmenDB` storage
//!
//! Provides multiple compression methods:
//! - Binary (BBQ): 32x compression, ~85% raw recall (~95% with rescore)
//! - Scalar (SQ8): 4x compression, ~99% recall
//! - RaBitQ: 8x compression, ~98% recall
//! - FastScan: SIMD-accelerated batched distance computation (5x speedup)

pub mod binary;
pub mod fastscan;
pub mod rabitq;
pub mod scalar;

pub use binary::{hamming_distance, BinaryParams};
pub use fastscan::{fastscan_batch, FastScanLUT, BATCH_SIZE as FASTSCAN_BATCH_SIZE};
pub use rabitq::{
    ADCTable, QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams, TrainedParams,
};
pub use scalar::{SQ8ADCTable, ScalarParams};
