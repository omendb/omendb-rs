//! Pure helper functions for VectorStore operations.
//!
//! These functions have no VectorStore dependency and can be tested in isolation.

use serde_json::Value as JsonValue;

/// Compute effective ef_search value.
///
/// Ensures ef >= k (HNSW requirement) and falls back to default if not specified.
#[inline]
pub fn compute_effective_ef(ef: Option<usize>, stored_ef: usize, k: usize) -> usize {
    ef.unwrap_or(stored_ef).max(k)
}

/// Convert stored quantization mode ID to bool.
///
/// Mode IDs: 0=none, 1=sq8
pub fn quantization_from_id(mode_id: u64) -> bool {
    mode_id == 1
}

/// Convert quantization bool to storage mode ID.
pub fn quantization_to_id(enabled: bool) -> u64 {
    u64::from(enabled)
}

/// Default empty JSON object for missing metadata.
#[inline]
pub fn default_metadata() -> JsonValue {
    serde_json::json!({})
}

/// Static default metadata for borrowing without allocation.
pub static DEFAULT_METADATA: std::sync::LazyLock<JsonValue> =
    std::sync::LazyLock::new(|| serde_json::json!({}));
