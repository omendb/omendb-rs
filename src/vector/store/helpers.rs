//! Pure helper functions for VectorStore operations.
//!
//! These functions have no VectorStore dependency and can be tested in isolation.

use crate::distance::norm_squared;
use crate::omen::Metric;
use crate::vector::types::Vector;
use anyhow::{bail, Result};
use serde_json::Value as JsonValue;

/// Compute effective ef_search value.
///
/// Ensures ef >= k (HNSW requirement) and falls back to default if not specified.
#[inline]
pub(crate) fn compute_effective_ef(ef: Option<usize>, stored_ef: usize, k: usize) -> usize {
    ef.unwrap_or(stored_ef).max(k)
}

/// Validate a public dense-search query before dispatching to brute-force or HNSW.
pub(crate) fn validate_search_query(
    metric: Metric,
    query: &Vector,
    expected_dim: usize,
    k: usize,
) -> Result<()> {
    if k == 0 {
        bail!("Invalid search parameters: k=0. Requirement: k > 0");
    }

    if query.dim() != expected_dim {
        bail!(
            "Query dimension mismatch: expected {}, got {}",
            expected_dim,
            query.dim()
        );
    }

    if query.data.iter().any(|x| !x.is_finite()) {
        bail!("Query vector contains invalid values (NaN or Infinity)");
    }

    if matches!(metric, Metric::Cosine) && norm_squared(&query.data) == 0.0 {
        bail!("Cannot search cosine index with zero vector query");
    }

    Ok(())
}

/// Convert stored quantization mode ID to bool.
///
/// Mode IDs: 0=none, 1=sq8
pub(crate) fn quantization_from_id(mode_id: u64) -> bool {
    mode_id == 1
}

/// Convert quantization bool to storage mode ID.
pub(crate) fn quantization_to_id(enabled: bool) -> u64 {
    u64::from(enabled)
}

/// Default empty JSON object for missing metadata.
#[inline]
pub(crate) fn default_metadata() -> JsonValue {
    serde_json::json!({})
}

/// Static default metadata for borrowing without allocation.
pub(crate) static DEFAULT_METADATA: std::sync::LazyLock<JsonValue> =
    std::sync::LazyLock::new(|| serde_json::json!({}));
