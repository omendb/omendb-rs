//! Pure helper functions for VectorStore operations.
//!
//! These functions have no VectorStore dependency and can be tested in isolation.

use super::record_store::RecordStore;
use super::HNSWIndex;
use super::{DEFAULT_HNSW_EF_CONSTRUCTION, DEFAULT_HNSW_EF_SEARCH, DEFAULT_HNSW_M};
use crate::omen::Metric;
use crate::vector::hnsw::HNSWParams;
use crate::vector::QuantizationMode;
use anyhow::Result;
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

/// Create HNSW index with quantization mode.
pub fn create_hnsw_index(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quantization_mode: Option<&QuantizationMode>,
    training_vectors: &[Vec<f32>],
) -> Result<HNSWIndex> {
    use crate::vector::hnsw_index::HNSWQuantization;

    let m = hnsw_m.max(DEFAULT_HNSW_M);
    let ef_construction = hnsw_ef_construction.max(DEFAULT_HNSW_EF_CONSTRUCTION);
    let ef_search = hnsw_ef_search.max(DEFAULT_HNSW_EF_SEARCH);

    let quantization = match quantization_mode {
        Some(QuantizationMode::SQ8) => HNSWQuantization::SQ8,
        None => HNSWQuantization::None,
    };

    HNSWIndex::builder()
        .dimensions(dimensions)
        .max_elements(training_vectors.len().max(10_000))
        .m(m)
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .metric(distance_metric.into())
        .quantization(quantization)
        .build_with_training(training_vectors)
}

/// Rebuild HNSW index maintaining slot-index correspondence
///
/// Inserts vectors in slot order so HNSW indices match RecordStore slots.
/// For deleted slots, inserts zero vectors and marks them deleted.
#[allow(clippy::too_many_arguments)]
pub fn rebuild_hnsw_with_slots(
    records: &RecordStore,
    deleted: &roaring::RoaringBitmap,
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quantization_mode: Option<&QuantizationMode>,
) -> Result<HNSWIndex> {
    // Collect live vectors for training (PQ/SQ codebooks)
    let training_vectors: Vec<Vec<f32>> = records.collect_vectors();

    let mut index = create_hnsw_index(
        dimensions,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        distance_metric,
        quantization_mode,
        &training_vectors,
    )?;

    // Insert vectors in slot order to maintain index == slot correspondence
    let zero_vector = vec![0.0f32; dimensions];
    let mut deleted_slots = Vec::new();

    for slot in 0..records.slot_count() {
        if deleted.contains(slot) {
            // Insert placeholder for deleted slot, mark deleted after
            index.insert(&zero_vector)?;
            deleted_slots.push(slot);
        } else if let Some(record) = records.get_by_slot(slot) {
            index.insert(&record.vector)?;
        } else {
            // Empty slot without delete marker - shouldn't happen but handle it
            index.insert(&zero_vector)?;
            deleted_slots.push(slot);
        }
    }

    // Mark all deleted slots in HNSW
    if !deleted_slots.is_empty() {
        index.mark_deleted_batch(&deleted_slots)?;
    }

    Ok(index)
}

/// Initialize HNSW index from quantization mode.
#[allow(dead_code)]
pub fn initialize_quantized_hnsw(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    quant_mode: QuantizationMode,
    _training_vectors: &[Vec<f32>],
) -> Result<HNSWIndex> {
    // Note: ef_search is a runtime parameter passed to search(), not stored in HNSWParams
    let _ = hnsw_ef_search; // Silence unused warning - caller passes it to search() at runtime
    let hnsw_params = HNSWParams::default()
        .with_m(hnsw_m)
        .with_ef_construction(hnsw_ef_construction);

    match quant_mode {
        QuantizationMode::SQ8 => {
            HNSWIndex::new_with_sq8(dimensions, hnsw_params, distance_metric.into())
        }
    }
}

/// Initialize standard (non-quantized) HNSW index.
#[allow(dead_code)]
pub fn initialize_standard_hnsw(
    dimensions: usize,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    distance_metric: Metric,
    capacity: usize,
) -> Result<HNSWIndex> {
    HNSWIndex::new_with_params(
        capacity,
        dimensions,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        distance_metric.into(),
    )
}

/// Default empty JSON object for missing metadata.
#[inline]
pub fn default_metadata() -> JsonValue {
    serde_json::json!({})
}
