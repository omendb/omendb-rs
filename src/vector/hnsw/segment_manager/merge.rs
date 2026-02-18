//! Segment merge operations
//!
//! Provides automatic and manual merging of frozen segments.
//! Merges use parallel HNSW construction (`build_parallel`) for
//! the merged index.

use super::{SegmentConfig, SegmentManager};
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::merge::{MergeConfig, MergeStats};
use crate::vector::hnsw::segment::FrozenSegment;
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

    /// IGTM merge configuration (reserved for future asymmetric merge
    /// where one segment is >>10x larger — avoids full rebuild).
    /// Currently unused: parallel build is fast enough for symmetric merge.
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

    /// Collect vectors and slots from frozen segments into separate vecs
    fn collect_from_segments(segments: &[Arc<FrozenSegment>]) -> (Vec<Vec<f32>>, Vec<u32>) {
        let total_len: usize = segments.iter().map(|s| s.len()).sum();
        let mut vectors = Vec::with_capacity(total_len);
        let mut slots = Vec::with_capacity(total_len);

        for frozen_arc in segments {
            let frozen = frozen_arc.as_ref();
            if frozen.is_empty() {
                continue;
            }

            let storage = frozen.storage();
            for id in 0..frozen.len() as u32 {
                vectors.push(storage.get_vector_ref(id).to_vec());
                slots.push(storage.slot(id));
            }
        }

        (vectors, slots)
    }

    /// Build a merged HNSWIndex via parallel construction, then remap slots
    fn build_merged_index(
        config: &SegmentConfig,
        vectors: Vec<Vec<f32>>,
        slots: &[u32],
    ) -> Result<(HNSWIndex, std::time::Duration)> {
        let start = std::time::Instant::now();

        let mut index = HNSWIndex::build_parallel(
            config.dimensions,
            config.params,
            config.distance_fn,
            config.quantization,
            vectors,
        )?;
        index.remap_slots(slots);

        Ok((index, start.elapsed()))
    }

    /// Create a frozen segment directly from an HNSWIndex (no MutableSegment roundtrip)
    pub(super) fn create_merged_segment(&mut self, index: HNSWIndex) -> Arc<FrozenSegment> {
        let segment = FrozenSegment::from_parts(
            self.next_segment_id,
            index.entry_point,
            *index.params(),
            index.distance_fn,
            index.storage,
        );
        self.next_segment_id += 1;
        Arc::new(segment)
    }

    /// Finish a merge: add merged segment, record stats, log
    fn finish_merge(
        &mut self,
        index: HNSWIndex,
        vectors_merged: usize,
        build_duration: std::time::Duration,
    ) -> MergeStats {
        if !index.is_empty() {
            let frozen = self.create_merged_segment(index);
            self.frozen.push(frozen);
        }

        let stats = MergeStats {
            vectors_merged,
            join_set_size: 0,
            join_set_duration: std::time::Duration::ZERO,
            join_set_insert_duration: build_duration,
            remaining_insert_duration: std::time::Duration::ZERO,
            total_duration: build_duration,
            fast_path_inserts: vectors_merged,
            fallback_inserts: 0,
        };

        info!(
            total_vectors = stats.vectors_merged,
            total_duration_ms = stats.total_duration.as_millis(),
            "Segment merge complete"
        );

        self.last_merge_stats = Some(stats.clone());
        stats
    }

    /// Merge all frozen segments into a single new frozen segment
    ///
    /// Uses parallel HNSW construction for the merged index.
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

        let segments_to_merge = std::mem::take(&mut self.frozen);
        let (vectors, slots) = Self::collect_from_segments(&segments_to_merge);
        if vectors.is_empty() {
            self.frozen = segments_to_merge;
            return Ok(None);
        }

        let vectors_merged = vectors.len();
        let (index, build_duration) = match Self::build_merged_index(&self.config, vectors, &slots)
        {
            Ok(result) => result,
            Err(e) => {
                self.frozen = segments_to_merge;
                return Err(e);
            }
        };

        debug!(
            vectors_merged,
            duration_ms = build_duration.as_millis(),
            "Merged frozen segments"
        );

        let stats = self.finish_merge(index, vectors_merged, build_duration);
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

        let (vectors, slots) = Self::collect_from_segments(&segments_to_merge);
        if vectors.is_empty() {
            return Ok(None);
        }

        let vectors_merged = vectors.len();
        let (index, build_duration) = match Self::build_merged_index(&self.config, vectors, &slots)
        {
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

        let stats = self.finish_merge(index, vectors_merged, build_duration);
        Ok(Some(stats))
    }
}
