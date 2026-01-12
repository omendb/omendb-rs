//! Vector compression for `OmenDB` storage
//!
//! Two production quantization modes:
//! - SQ8: 4x compression, ~99% recall (default when quantization enabled)
//! - RaBitQ: 32x compression, ~95% recall with rescore (true 1-bit RaBitQ)
//!
//! Legacy modules (deprecated, will be removed):
//! - sq_multi: Multi-bit scalar quantization (use SQ8 instead)
//! - binary: Basic binary quantization (use RaBitQ instead)
//! - fastscan: SIMD batched ADC (for sq_multi, deprecated)

// Production modules
pub mod rabitq;
pub mod scalar;

// Legacy modules (deprecated)
pub mod binary;
pub mod fastscan;
pub mod sq_multi;

// Production exports
pub use rabitq::{estimate_distance, CodeMetadata, QueryLUT, RaBitQIndex};
pub use scalar::{symmetric_l2_squared_u8, QueryPrep, ScalarParams};

// Legacy exports (deprecated, for backward compatibility)
// These will be removed in a future version
pub use binary::{hamming_distance, BinaryParams};
pub use fastscan::{
    fastscan_batch, fastscan_batch_with_lut, FastScanLUT, BATCH_SIZE as FASTSCAN_BATCH_SIZE,
};
pub use sq_multi::{
    ADCTable, QuantizationBits, QuantizedVector, RaBitQ, RaBitQParams, TrainedParams,
};
