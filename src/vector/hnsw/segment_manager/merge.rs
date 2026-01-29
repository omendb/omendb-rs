//! Segment merge operations
//!
//! Provides automatic and manual merging of frozen segments using
//! the IGTM (Iterative Greedy Tree Merging) algorithm.

use super::SegmentManager;
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::merge::{MergeConfig, MergeStats};
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment};
use std::sync::Arc;
use tracing::{debug, info};

/// Policy for automatic segment merging
///
/// Controls when and how frozen segments are merged together.
/// Merging reduces the number of segments to search and improves
/// cache locality, but requires CPU time.
#[derive(Clone, Debug)]
pub struct MergePolicy {
    /// Minimum number of frozen segments before considering merge
    /// Default: 2 (merge when at least 2 frozen segments exist)
    pub min_segments: usize,

    /// Maximum number of frozen segments before forcing merge
    /// Default: 8 (always merge when this many segments exist)
    pub max_segments: usize,

    /// Minimum total vectors in frozen segments before merge
    /// Default: 1000 (don't merge tiny segments)
    pub min_vectors: usize,

    /// Size ratio threshold: merge if largest / smallest > ratio
    /// Default: 4.0 (merge if segments are very unbalanced)
    pub size_ratio_threshold: f32,

    /// IGTM merge configuration
    pub merge_config: MergeConfig,

    /// Whether automatic merging is enabled
    pub enabled: bool,
}

impl Default for MergePolicy {
    fn default() -> Self {
        Self {
            min_segments: 2,
            max_segments: 8,
            min_vectors: 1000,
            size_ratio_threshold: 4.0,
            merge_config: MergeConfig::default(),
            enabled: true,
        }
    }
}

impl MergePolicy {
    /// Create a disabled merge policy (no automatic merging)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create an aggressive merge policy (merge frequently)
    pub fn aggressive() -> Self {
        Self {
            min_segments: 2,
            max_segments: 4,
            min_vectors: 100,
            size_ratio_threshold: 2.0,
            merge_config: MergeConfig::default(),
            enabled: true,
        }
    }

    /// Create a conservative merge policy (merge rarely)
    pub fn conservative() -> Self {
        Self {
            min_segments: 4,
            max_segments: 16,
            min_vectors: 10_000,
            size_ratio_threshold: 8.0,
            merge_config: MergeConfig::default(),
            enabled: true,
        }
    }

    /// Set minimum segments threshold
    #[must_use]
    pub fn with_min_segments(mut self, min: usize) -> Self {
        self.min_segments = min;
        self
    }

    /// Set maximum segments threshold
    #[must_use]
    pub fn with_max_segments(mut self, max: usize) -> Self {
        self.max_segments = max;
        self
    }

    /// Enable or disable automatic merging
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl SegmentManager {
    /// Check if merge should be triggered based on current policy
    ///
    /// Returns true if:
    /// - Policy is enabled AND
    /// - (frozen segments >= max_segments OR
    ///   (frozen segments >= min_segments AND
    ///   (total frozen vectors >= min_vectors OR size ratio exceeded)))
    pub fn should_merge(&self) -> bool {
        if !self.merge_policy.enabled {
            return false;
        }

        let num_frozen = self.frozen.len();

        // Always merge if we hit max segments
        if num_frozen >= self.merge_policy.max_segments {
            return true;
        }

        // Need at least min_segments to consider merging
        if num_frozen < self.merge_policy.min_segments {
            return false;
        }

        // Check total vectors threshold
        let total_frozen_vectors: usize = self.frozen.iter().map(|s| s.len()).sum();
        if total_frozen_vectors >= self.merge_policy.min_vectors {
            return true;
        }

        // Check size ratio (merge unbalanced segments)
        if num_frozen >= 2 {
            let sizes: Vec<usize> = self.frozen.iter().map(|s| s.len()).collect();
            let max_size = *sizes.iter().max().unwrap_or(&0);
            let min_size = *sizes.iter().min().unwrap_or(&1).max(&1);
            let ratio = max_size as f32 / min_size as f32;

            if ratio > self.merge_policy.size_ratio_threshold {
                return true;
            }
        }

        false
    }

    /// Collect all (vector, slot) pairs from frozen segments
    pub(super) fn collect_vectors_and_slots(
        segments: &[Arc<FrozenSegment>],
    ) -> Vec<(Vec<f32>, u32)> {
        let total_len: usize = segments.iter().map(|s| s.len()).sum();
        let mut all_vectors = Vec::with_capacity(total_len);

        for frozen_arc in segments {
            let frozen = frozen_arc.as_ref();
            if frozen.is_empty() {
                continue;
            }

            let storage = frozen.storage();
            for id in 0..frozen.len() as u32 {
                let vector = storage.vector(id).to_vec();
                let slot = storage.slot(id);
                all_vectors.push((vector, slot));
            }
        }

        all_vectors
    }

    /// Insert vectors into index, tracking slots. Returns (slots, duration) on success.
    pub(super) fn insert_vectors_with_slots(
        index: &mut HNSWIndex,
        vectors: &[(Vec<f32>, u32)],
    ) -> Result<(Vec<u32>, std::time::Duration)> {
        let mut collected_slots = Vec::with_capacity(vectors.len());
        let insert_start = std::time::Instant::now();

        for (vector, slot) in vectors {
            index.insert(vector)?;
            collected_slots.push(*slot);
        }

        Ok((collected_slots, insert_start.elapsed()))
    }

    /// Create a frozen segment from merged index with slots
    pub(super) fn create_merged_segment(
        &mut self,
        index: HNSWIndex,
        slots: &[u32],
    ) -> Arc<FrozenSegment> {
        let mut mutable = MutableSegment::from_index(index, slots);
        mutable.set_id(self.next_segment_id);
        self.next_segment_id += 1;
        Arc::new(mutable.freeze())
    }

    /// Build MergeStats from merge operation
    pub(super) fn build_merge_stats(
        vectors_merged: usize,
        insert_duration: std::time::Duration,
    ) -> MergeStats {
        MergeStats {
            vectors_merged,
            join_set_size: 0,
            join_set_duration: std::time::Duration::ZERO,
            join_set_insert_duration: insert_duration,
            remaining_insert_duration: std::time::Duration::ZERO,
            total_duration: insert_duration,
            fast_path_inserts: vectors_merged,
            fallback_inserts: 0,
        }
    }

    /// Merge all frozen segments into a single new frozen segment
    ///
    /// The result is a single frozen segment replacing all previous frozen segments.
    /// Returns merge statistics if any segments were merged.
    pub fn merge_all_frozen(&mut self) -> Result<Option<MergeStats>> {
        if self.frozen.len() < 2 {
            return Ok(None);
        }

        info!(
            frozen_count = self.frozen.len(),
            frozen_vectors = self.frozen.iter().map(|s| s.len()).sum::<usize>(),
            "Starting segment merge"
        );

        // Take ownership of segments (will restore on failure)
        let segments_to_merge = std::mem::take(&mut self.frozen);

        // Collect vectors and slots from all segments
        let all_vectors = Self::collect_vectors_and_slots(&segments_to_merge);
        if all_vectors.is_empty() {
            return Ok(None);
        }

        // Build merged index
        let mut merged_index = HNSWIndex::new(
            self.config.dimensions,
            self.config.params,
            self.config.distance_fn,
            self.config.use_quantization,
        )?;

        // Insert all vectors with slot tracking
        let (collected_slots, insert_duration) =
            match Self::insert_vectors_with_slots(&mut merged_index, &all_vectors) {
                Ok(result) => result,
                Err(e) => {
                    self.frozen = segments_to_merge;
                    return Err(e);
                }
            };

        let vectors_merged = all_vectors.len();
        debug!(
            vectors_merged,
            duration_ms = insert_duration.as_millis(),
            "Merged frozen segments"
        );

        // Create merged segment and stats
        if !merged_index.is_empty() {
            let frozen = self.create_merged_segment(merged_index, &collected_slots);
            self.frozen.push(frozen);
        }

        let stats = Self::build_merge_stats(vectors_merged, insert_duration);
        info!(
            total_vectors = stats.vectors_merged,
            total_duration_ms = stats.total_duration.as_millis(),
            "Segment merge complete"
        );

        self.last_merge_stats = Some(stats.clone());
        Ok(Some(stats))
    }

    /// Check and merge if policy conditions are met
    ///
    /// Call this periodically (e.g., after each freeze) to trigger
    /// automatic merging when the policy thresholds are reached.
    ///
    /// Returns merge statistics if a merge was performed.
    pub fn check_and_merge(&mut self) -> Result<Option<MergeStats>> {
        if self.should_merge() {
            self.merge_all_frozen()
        } else {
            Ok(None)
        }
    }

    /// Merge specific frozen segments by index
    ///
    /// Merges the specified segments into a new frozen segment,
    /// removing the originals. Useful for targeted merging.
    ///
    /// # Arguments
    /// * `indices` - Indices of frozen segments to merge (must be sorted ascending, unique)
    pub fn merge_segments(&mut self, indices: &[usize]) -> Result<Option<MergeStats>> {
        if indices.is_empty() || indices.len() == 1 {
            return Ok(None);
        }

        // Validate indices are sorted ascending and unique
        for i in 1..indices.len() {
            if indices[i] <= indices[i - 1] {
                return Err(crate::vector::hnsw::error::HNSWError::internal(
                    "Segment indices must be sorted ascending with no duplicates".to_string(),
                ));
            }
        }

        // Validate indices in range
        for &idx in indices {
            if idx >= self.frozen.len() {
                return Err(crate::vector::hnsw::error::HNSWError::internal(format!(
                    "Segment index {} out of range (have {})",
                    idx,
                    self.frozen.len()
                )));
            }
        }

        // Extract segments to merge (in reverse order to preserve indices)
        let mut segments_to_merge: Vec<Arc<FrozenSegment>> = Vec::with_capacity(indices.len());
        for &idx in indices.iter().rev() {
            segments_to_merge.push(self.frozen.remove(idx));
        }
        segments_to_merge.reverse();

        // Collect vectors and slots from selected segments
        let all_vectors = Self::collect_vectors_and_slots(&segments_to_merge);
        if all_vectors.is_empty() {
            return Ok(None);
        }

        // Build merged index
        let mut merged_index = HNSWIndex::new(
            self.config.dimensions,
            self.config.params,
            self.config.distance_fn,
            self.config.use_quantization,
        )?;

        // Insert all vectors with slot tracking
        let (collected_slots, insert_duration) =
            match Self::insert_vectors_with_slots(&mut merged_index, &all_vectors) {
                Ok(result) => result,
                Err(e) => {
                    // Restore segments on failure (best-effort)
                    for (i, seg) in segments_to_merge.into_iter().enumerate() {
                        let insert_idx = indices[i].min(self.frozen.len());
                        self.frozen.insert(insert_idx, seg);
                    }
                    return Err(e);
                }
            };

        // Create merged segment and stats
        if !merged_index.is_empty() {
            let frozen = self.create_merged_segment(merged_index, &collected_slots);
            self.frozen.push(frozen);
        }

        let stats = Self::build_merge_stats(all_vectors.len(), insert_duration);
        self.last_merge_stats = Some(stats.clone());
        Ok(Some(stats))
    }
}
