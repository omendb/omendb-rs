//! True RaBitQ: Randomly transformed Binary Quantization
//!
//! Implements the RaBitQ algorithm from arXiv:2405.12497 for 32x compression
//! with theoretical error bounds and unbiased distance estimation.
//!
//! # Algorithm Overview
//!
//! 1. **Training**: Compute centroid from sample vectors, generate random rotation
//! 2. **Quantization**: Subtract centroid → Rotate (FFHT) → Sign extraction (1-bit)
//! 3. **Distance**: Binary inner product + corrective factors for unbiased estimation
//!
//! # Why Rotation Matters
//!
//! Random orthogonal rotation (via FFHT + Kac's Walk) spreads information across
//! all dimensions, making sign-based quantization much more informative than
//! naive binary quantization.
//!
//! # Reference
//!
//! Based on [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) (Apache-2.0)

mod distance;
mod quantize;
mod rotate;
mod simd;

pub use distance::{estimate_distance, QueryLUT};
pub use quantize::{CodeMetadata, RaBitQIndex};
