//! Pure helper functions for VectorStore operations.
//!
//! These functions have no VectorStore dependency and can be tested in isolation.

use crate::vector::QuantizationMode;
use serde_json::Value as JsonValue;

/// Compute effective ef_search value.
///
/// Ensures ef >= k (HNSW requirement) and falls back to default if not specified.
#[inline]
pub fn compute_effective_ef(ef: Option<usize>, stored_ef: usize, k: usize) -> usize {
    ef.unwrap_or(stored_ef).max(k)
}

/// Compute optimal oversample factor based on quantization mode.
///
/// Different quantization modes have different baseline recall:
/// - SQ8: ~99% accurate, needs minimal oversampling (2.0x)
/// - No quantization: 1.0 (rescore disabled)
pub fn default_oversample_for_quantization(mode: Option<&QuantizationMode>) -> f32 {
    match mode {
        None => 1.0,
        Some(QuantizationMode::SQ8) => 2.0,
    }
}

/// Convert stored quantization mode ID to QuantizationMode.
///
/// Mode IDs: 0=none, 1=sq8
pub fn quantization_mode_from_id(mode_id: u64) -> Option<QuantizationMode> {
    match mode_id {
        1 => Some(QuantizationMode::SQ8),
        _ => None,
    }
}

/// Convert QuantizationMode to storage mode ID.
pub fn quantization_mode_to_id(mode: &QuantizationMode) -> u64 {
    match mode {
        QuantizationMode::SQ8 => 1,
    }
}

/// Default empty JSON object for missing metadata.
#[inline]
pub fn default_metadata() -> JsonValue {
    serde_json::json!({})
}
