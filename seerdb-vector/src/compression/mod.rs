//! Vector compression for seerdb-vector storage
//!
//! Provides RaBitQ compression for tiered storage:
//! - L0-L2: Full precision (f32)
//! - L3-L4: RaBitQ 4-bit (8× compression, 98% recall)
//! - L5-L6: RaBitQ 2-bit (16× compression, 95% recall)

pub mod rabitq;

pub use rabitq::{RaBitQ, RaBitQParams, QuantizedVector, QuantizationBits};
