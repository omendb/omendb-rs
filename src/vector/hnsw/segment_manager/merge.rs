//! Segment merge operations
//!
//! Provides automatic and manual merging of frozen segments.
//! Merges use parallel HNSW construction (`build_parallel`) for
//! the merged index.

use super::{PendingMergeState, SegmentConfig, SegmentManager};
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::merge::{MergeConfig, MergeStats};
use crate::vector::hnsw::segment::FrozenSegment;
use std::sync::Arc;
use tracing::{debug, info};

enum PendingMergeCompletion {
    NotReady,
    Finished(Option<usize>),
}

struct FrozenMergeInput {
    source_segment_ids: Box<[u64]>,
    segments: Vec<Arc<FrozenSegment>>,
    total_vectors: usize,
}

impl FrozenMergeInput {
    fn from_segments(segments: Vec<Arc<FrozenSegment>>) -> Option<Self> {
        if segments.len() < 2 {
            return None;
        }

        let total_vectors = segments.iter().map(|segment| segment.len()).sum();
        let source_segment_ids = segments
            .iter()
            .map(|segment| segment.id())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Some(Self {
            source_segment_ids,
            segments,
            total_vectors,
        })
    }

    fn source_count(&self) -> usize {
        self.source_segment_ids.len()
    }

    fn total_vectors(&self) -> usize {
        self.total_vectors
    }

    fn size_bounds(&self) -> Option<(usize, usize)> {
        let mut sizes = self.segments.iter().map(|segment| segment.len());
        let first = sizes.next()?;
        let mut min_size = first;
        let mut max_size = first;
        for size in sizes {
            min_size = min_size.min(size);
            max_size = max_size.max(size);
        }
        Some((min_size, max_size))
    }
}

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
    fn current_frozen_merge_input(&self) -> Option<FrozenMergeInput> {
        FrozenMergeInput::from_segments(self.published.cloned_frozen_segments())
    }

    fn selected_frozen_merge_input(&self, indices: &[usize]) -> Result<Option<FrozenMergeInput>> {
        if indices.is_empty() || indices.len() == 1 {
            return Ok(None);
        }

        for i in 1..indices.len() {
            if indices[i] <= indices[i - 1] {
                return Err(crate::vector::hnsw::error::HNSWError::internal(
                    "Segment indices must be sorted ascending with no duplicates".to_string(),
                ));
            }
        }

        FrozenMergeInput::from_segments(self.published.cloned_frozen_indices(indices)?)
            .map_or(Ok(None), |input| Ok(Some(input)))
    }

    fn should_merge_input(&self, input: &FrozenMergeInput) -> bool {
        let num_frozen = input.source_count();

        if num_frozen >= self.merge_policy.max_segments {
            return true;
        }

        if num_frozen < self.merge_policy.min_segments {
            return false;
        }

        if input.total_vectors() >= self.merge_policy.min_vectors {
            return true;
        }

        let Some((min_size, max_size)) = input.size_bounds() else {
            return false;
        };
        let ratio = max_size as f32 / min_size.max(1) as f32;
        ratio > self.merge_policy.size_ratio_threshold
    }

    fn complete_pending_merge(&mut self, wait: bool) -> Option<PendingMergeCompletion> {
        let pending = self.pending_merge.take()?;
        if !wait && !pending.is_finished() {
            self.pending_merge = Some(pending);
            return Some(PendingMergeCompletion::NotReady);
        }

        let (handle, source_segment_ids) = pending.into_parts();
        match handle.join() {
            Ok(Ok(merged)) => Some(PendingMergeCompletion::Finished(
                self.publish_completed_pending_merge(merged, source_segment_ids.as_ref()),
            )),
            Ok(Err(e)) => {
                tracing::warn!("Background merge failed: {e}");
                Some(PendingMergeCompletion::Finished(None))
            }
            Err(_) => {
                tracing::warn!("Background merge thread panicked");
                Some(PendingMergeCompletion::Finished(None))
            }
        }
    }

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

        self.current_frozen_merge_input()
            .is_some_and(|input| self.should_merge_input(&input))
    }

    /// Collect borrowed vectors and slots from frozen segments.
    pub(super) fn collect_from_segments(
        segments: &[Arc<FrozenSegment>],
    ) -> (Vec<&[f32]>, Vec<u32>) {
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
                vectors.push(storage.get_vector_ref(id));
                slots.push(storage.slot(id));
            }
        }

        (vectors, slots)
    }

    /// Build a merged HNSWIndex via parallel construction, then remap slots
    pub(super) fn build_merged_index(
        config: &SegmentConfig,
        vectors: Vec<&[f32]>,
        slots: &[u32],
    ) -> Result<(HNSWIndex, std::time::Duration)> {
        let start = std::time::Instant::now();

        let mut index = HNSWIndex::build_parallel_from_refs(
            config.dimensions,
            config.params,
            config.distance_fn,
            config.quantization,
            vectors,
        )?;
        index.remap_slots(slots);

        // Optimize merged index for cache locality
        if !index.is_empty() {
            let _ = index.optimize_cache_locality();
        }

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
        segments_merged: usize,
        build_duration: std::time::Duration,
    ) -> MergeStats {
        if !index.is_empty() {
            let frozen = self.create_merged_segment(index);
            self.publish_frozen_segment(frozen);
        }

        let stats = MergeStats {
            vectors_merged,
            segments_merged,
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
        // Wait for any in-progress background merge to finish before starting
        // an explicit merge — prevents redundant concurrent builds of the same segments.
        self.drain_pending_merge();
        let Some(merge_input) = self.current_frozen_merge_input() else {
            return Ok(None);
        };

        let segments_merged = merge_input.source_count();
        info!(
            frozen_count = segments_merged,
            frozen_vectors = merge_input.total_vectors(),
            "Starting segment merge"
        );

        self.generation += 1;
        let segments_to_merge = self.published.take_all_frozen(self.generation);
        let (vectors, slots) = Self::collect_from_segments(&segments_to_merge);
        if vectors.is_empty() {
            self.generation += 1;
            self.published
                .restore_all_frozen(segments_to_merge, self.generation);
            return Ok(None);
        }

        let vectors_merged = vectors.len();
        let (index, build_duration) = match Self::build_merged_index(&self.config, vectors, &slots)
        {
            Ok(result) => result,
            Err(e) => {
                self.generation += 1;
                self.published
                    .restore_all_frozen(segments_to_merge, self.generation);
                return Err(e);
            }
        };

        debug!(
            vectors_merged,
            duration_ms = build_duration.as_millis(),
            "Merged frozen segments"
        );

        let stats = self.finish_merge(index, vectors_merged, segments_merged, build_duration);
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
        // A background merge in flight has snapshotted the current frozen prefix by segment ID.
        // Mutating the published frozen topology before that merge is applied would invalidate
        // that publication path.
        // Complete any pending merge first to get a stable baseline.
        self.drain_pending_merge();

        let Some(_merge_input) = self.selected_frozen_merge_input(indices)? else {
            return Ok(None);
        };

        let segments_merged = indices.len();
        self.generation += 1;
        let segments_to_merge = self.published.take_frozen_indices(indices, self.generation);

        let (vectors, slots) = Self::collect_from_segments(&segments_to_merge);
        if vectors.is_empty() {
            self.generation += 1;
            self.published
                .restore_frozen_indices(indices, segments_to_merge, self.generation);
            return Ok(None);
        }

        let vectors_merged = vectors.len();
        let (index, build_duration) = match Self::build_merged_index(&self.config, vectors, &slots)
        {
            Ok(result) => result,
            Err(e) => {
                // Restore segments on failure (best-effort)
                self.generation += 1;
                self.published
                    .restore_frozen_indices(indices, segments_to_merge, self.generation);
                return Err(e);
            }
        };

        let stats = self.finish_merge(index, vectors_merged, segments_merged, build_duration);
        Ok(Some(stats))
    }

    /// Start a background merge if conditions are met and no merge is already running.
    ///
    /// Clones the Arc refs to frozen segments (cheap) and spawns a thread to build
    /// the merged index. The original frozen segments stay published and remain
    /// searchable while the merge runs. When the merge completes, call
    /// `apply_pending_merge_if_ready()` to atomically swap in the merged segment.
    pub fn try_start_background_merge(&mut self) {
        if self.pending_merge.is_some() {
            return; // Already merging
        }
        let Some(merge_input) = self.current_frozen_merge_input() else {
            return;
        };
        if !self.should_merge_input(&merge_input) {
            return;
        }

        // Clone Arcs (cheap) — original segments stay published and remain searchable.
        let FrozenMergeInput {
            source_segment_ids,
            segments,
            total_vectors,
        } = merge_input;
        let config = self.config.clone();
        // Pre-assign segment ID so the background thread can build the FrozenSegment directly
        let segment_id = self.next_segment_id;
        self.next_segment_id += 1;

        let segments_dir = self.pending_merge_dir.clone();
        let source_ids = source_segment_ids.to_vec();
        let source_ids_for_meta = source_ids.clone();

        tracing::info!(
            frozen_count = source_ids.len(),
            frozen_vectors = total_vectors,
            "Starting background segment merge"
        );

        let handle = std::thread::spawn(move || {
            let (vectors, slots) = SegmentManager::collect_from_segments(&segments);
            if vectors.is_empty() {
                return Err(crate::vector::hnsw::error::HNSWError::internal(
                    "No vectors to merge".to_string(),
                ));
            }

            let (index, elapsed) = SegmentManager::build_merged_index(&config, vectors, &slots)?;

            tracing::debug!(
                vectors = slots.len(),
                duration_ms = elapsed.as_millis(),
                "Background merge build complete"
            );

            let frozen = FrozenSegment::from_parts(
                segment_id,
                index.entry_point,
                *index.params(),
                index.distance_fn,
                index.storage,
            );

            // Persist the merged segment and a metadata file so the merge survives a crash.
            // Recovery checks for pending_merge.meta and applies it if source segments match.
            if let Some(ref dir) = segments_dir {
                let segment_path = dir.join(format!("segment_{segment_id}.bin"));
                if let Err(e) = frozen.save(&segment_path) {
                    tracing::warn!("Failed to persist background merge segment: {e}");
                } else {
                    let meta = serde_json::json!({
                        "source_ids": source_ids_for_meta,
                        "total_vectors": total_vectors,
                        "merged_segment_id": segment_id,
                    });
                    let meta_path = dir.join("pending_merge.meta");
                    let meta_tmp = dir.join("pending_merge.meta.tmp");
                    match serde_json::to_vec_pretty(&meta) {
                        Ok(meta_bytes) => {
                            if let Ok(mut f) = std::fs::File::create(&meta_tmp) {
                                use std::io::Write;
                                if f.write_all(&meta_bytes).is_ok() && f.sync_all().is_ok() {
                                    if let Err(e) = std::fs::rename(&meta_tmp, &meta_path) {
                                        tracing::warn!("Failed to write pending_merge.meta: {e}");
                                        let _ = std::fs::remove_file(&meta_tmp);
                                    } else {
                                        tracing::debug!(
                                            merged_segment_id = segment_id,
                                            "Persisted background merge result"
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => tracing::warn!("Failed to serialize pending_merge.meta: {e}"),
                    }
                }
            }

            Ok(Arc::new(frozen))
        });

        self.pending_merge = Some(PendingMergeState::new(handle, source_ids));
    }

    /// Apply a completed background merge if the thread is done.
    ///
    /// Non-blocking: returns immediately if the merge is still running.
    /// When the merge completes, atomically removes the merged segments and
    /// inserts the merged result. Segments added during the merge after the
    /// original frozen prefix are preserved.
    pub fn apply_pending_merge_if_ready(&mut self) -> bool {
        match self.complete_pending_merge(false) {
            Some(PendingMergeCompletion::Finished(Some(drain_count))) => {
                tracing::info!(
                    merged_segments = drain_count,
                    remaining_segments = self.published.frozen_count(),
                    "Applied background merge"
                );
                true
            }
            None
            | Some(PendingMergeCompletion::NotReady | PendingMergeCompletion::Finished(None)) => {
                false
            }
        }
    }

    /// Wait for any pending background merge to complete and apply the result.
    ///
    /// Blocks until the merge thread finishes. Called during flush/close to ensure
    /// merge results are not discarded on clean shutdown.
    pub fn drain_pending_merge(&mut self) {
        if let Some(PendingMergeCompletion::Finished(Some(drain_count))) =
            self.complete_pending_merge(true)
        {
            tracing::info!(
                merged_segments = drain_count,
                "Applied pending background merge during drain"
            );
        }
    }
}
