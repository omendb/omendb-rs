//! Segment manager for coordinating mutable and frozen segments
//!
//! The SegmentManager provides a unified interface over multiple segments:
//! - One active mutable segment for writes
//! - Zero or more frozen segments for reads
//!
//! When the mutable segment reaches capacity, it's frozen and a new
//! mutable segment is created. Searches query all segments, using parallel
//! iteration for 4+ frozen segments and sequential iteration below that.
//!
//! ## Automatic Merging
//!
//! When multiple frozen segments accumulate, they can be merged into a
//! single segment via parallel HNSW construction. Set a merge policy to
//! enable automatic merging.

mod merge;
mod persistence;

pub use merge::MergePolicy;

use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::merge::MergeStats;
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment, SegmentSearchResult};
use crate::vector::hnsw::types::{HNSWParams, Metric};
use std::path::PathBuf;
use std::sync::Arc;

type PendingMergeHandle = std::thread::JoinHandle<Result<Arc<FrozenSegment>>>;

/// Configuration for segment manager
#[derive(Clone, Debug)]
pub struct SegmentConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// HNSW parameters
    pub params: HNSWParams,
    /// Distance function
    pub distance_fn: Metric,
    /// Max vectors per segment before freezing
    pub segment_capacity: usize,
    /// SQ8 quantization enabled (true = 4x compression, false = full precision)
    pub quantization: bool,
}

impl SegmentConfig {
    /// Default segment capacity before freezing
    pub const DEFAULT_CAPACITY: usize = 100_000;

    /// Create default config
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            params: HNSWParams::default(),
            distance_fn: Metric::L2,
            segment_capacity: Self::DEFAULT_CAPACITY,
            quantization: false,
        }
    }

    /// Set HNSW parameters
    #[must_use]
    pub fn with_params(mut self, params: HNSWParams) -> Self {
        self.params = params;
        self
    }

    /// Set distance function
    #[must_use]
    pub fn with_distance(mut self, distance_fn: Metric) -> Self {
        self.distance_fn = distance_fn;
        self
    }

    /// Set segment capacity
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.segment_capacity = capacity;
        self
    }

    /// Set SQ8 quantization (true = enabled, false = full precision)
    #[must_use]
    pub fn with_quantization(mut self, quantization: bool) -> Self {
        self.quantization = quantization;
        self
    }
}

/// Manages mutable and frozen segments
///
/// Provides unified insert and search over multiple segments.
/// The mutable segment is frozen when it reaches capacity.
/// Supports automatic merging of frozen segments via IGTM algorithm.
pub struct SegmentManager {
    /// Configuration
    pub(crate) config: SegmentConfig,
    /// Currently published mutable + frozen topology.
    pub(crate) published: PublishedSegments,
    /// Next segment ID
    pub(crate) next_segment_id: u64,
    /// Merge policy for automatic merging
    pub(crate) merge_policy: MergePolicy,
    /// Statistics from last merge operation
    pub(crate) last_merge_stats: Option<MergeStats>,
    /// Generation counter, incremented on each save for staleness detection
    pub(crate) generation: u64,
    /// Background merge state: None when idle, Some while merge is in progress.
    pub(crate) pending_merge: Option<PendingMergeState>,
    /// Directory where segment files are stored, used to persist background merge results.
    /// Set by VectorStore after loading or creating segments.
    pub(crate) pending_merge_dir: Option<PathBuf>,
}

pub(crate) struct PendingMergeState {
    handle: PendingMergeHandle,
    source_segment_ids: Box<[u64]>,
}

impl PendingMergeState {
    fn new(handle: PendingMergeHandle, source_segment_ids: Vec<u64>) -> Self {
        Self {
            handle,
            source_segment_ids: source_segment_ids.into_boxed_slice(),
        }
    }

    fn source_segment_ids(&self) -> &[u64] {
        &self.source_segment_ids
    }

    fn source_count(&self) -> usize {
        self.source_segment_ids.len()
    }

    fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    fn into_parts(self) -> (PendingMergeHandle, Box<[u64]>) {
        (self.handle, self.source_segment_ids)
    }
}

pub(crate) struct PublishedSegments {
    pub(crate) mutable: MutableSegment,
    pub(crate) frozen: Vec<Arc<FrozenSegment>>,
}

impl PublishedSegments {
    fn new(mutable: MutableSegment) -> Self {
        Self {
            mutable,
            frozen: Vec::new(),
        }
    }

    fn from_parts(mutable: MutableSegment, frozen: Vec<Arc<FrozenSegment>>) -> Self {
        Self { mutable, frozen }
    }

    fn publish_frozen_segment(&mut self, frozen: Arc<FrozenSegment>) {
        self.frozen.push(frozen);
    }

    fn rollover_mutable(
        &mut self,
        next_mutable: MutableSegment,
        next_segment_id: u64,
    ) -> Option<FrozenSegment> {
        let mut old_mutable = std::mem::replace(&mut self.mutable, next_mutable);
        if old_mutable.is_empty() {
            return None;
        }

        old_mutable.set_id(next_segment_id);
        Some(old_mutable.freeze())
    }

    fn publish_completed_merge(&mut self, merged: Arc<FrozenSegment>, drain_count: usize) {
        self.frozen.drain(0..drain_count);
        self.frozen.insert(0, merged);
    }

    fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    fn mutable_len(&self) -> usize {
        self.mutable.len()
    }

    fn len(&self) -> usize {
        self.mutable.len()
            + self
                .frozen
                .iter()
                .map(|segment| segment.len())
                .sum::<usize>()
    }

    fn total_memory(&self) -> usize {
        let mutable = self.mutable.index().memory_usage();
        let frozen: usize = self
            .frozen
            .iter()
            .map(|segment| segment.index().memory_usage())
            .sum();
        mutable + frozen
    }
}

/// Immutable view of the currently published segment state.
///
/// This is the first seam toward snapshot-style read serving: queries only need
/// read access to the current mutable segment and frozen segment list, not the
/// full mutable manager API.
#[derive(Clone, Copy)]
pub struct SegmentReadView<'a> {
    mutable: &'a MutableSegment,
    frozen: &'a [Arc<FrozenSegment>],
    config: &'a SegmentConfig,
    generation: u64,
}

impl SegmentReadView<'_> {
    /// Number of frozen segments visible to readers.
    pub fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    /// Number of vectors in the mutable segment visible to readers.
    pub fn mutable_len(&self) -> usize {
        self.mutable.len()
    }

    /// Total number of vectors visible across mutable and frozen segments.
    pub fn len(&self) -> usize {
        self.mutable.len() + self.frozen.iter().map(|s| s.len()).sum::<usize>()
    }

    /// Check whether the visible segment set is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generation of the published segment state.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Segment capacity for the currently published topology.
    pub fn segment_capacity(&self) -> usize {
        self.config.segment_capacity
    }

    /// Total graph memory visible across mutable and frozen segments.
    pub fn total_memory(&self) -> usize {
        let mutable = self.mutable.index().memory_usage();
        let frozen: usize = self.frozen.iter().map(|s| s.index().memory_usage()).sum();
        mutable + frozen
    }

    /// Search across the visible mutable and frozen segments.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SegmentSearchResult>> {
        let mut results = self.mutable.search(query, k, ef)?;

        if !self.frozen.is_empty() {
            if self.frozen.len() >= 4 {
                use rayon::prelude::*;
                let frozen_results: Vec<SegmentSearchResult> = self
                    .frozen
                    .par_iter()
                    .flat_map(|seg| seg.search(query, k, ef))
                    .collect();
                results.extend(frozen_results);
            } else {
                for seg in self.frozen {
                    results.extend(seg.search(query, k, ef));
                }
            }
        }

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Search across the visible mutable and frozen segments with a filter predicate.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> Result<Vec<SegmentSearchResult>>
    where
        F: Fn(u32) -> bool + Sync,
    {
        let mut results = self.mutable.search_with_filter(query, k, ef, &filter_fn)?;

        if !self.frozen.is_empty() {
            if self.frozen.len() >= 4 {
                use rayon::prelude::*;
                let frozen_results: Vec<SegmentSearchResult> = self
                    .frozen
                    .par_iter()
                    .flat_map(|seg| seg.search_with_filter(query, k, ef, &filter_fn))
                    .collect();
                results.extend(frozen_results);
            } else {
                for seg in self.frozen {
                    results.extend(seg.search_with_filter(query, k, ef, &filter_fn));
                }
            }
        }

        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }
}

impl SegmentManager {
    /// Create new segment manager with default merge policy
    pub fn new(config: SegmentConfig) -> Result<Self> {
        Self::with_merge_policy(config, MergePolicy::default())
    }

    /// Create new segment manager with custom merge policy
    pub fn with_merge_policy(config: SegmentConfig, merge_policy: MergePolicy) -> Result<Self> {
        let mutable = if config.quantization {
            MutableSegment::new_quantized(config.dimensions, config.params, config.distance_fn)?
        } else {
            MutableSegment::with_capacity(
                config.dimensions,
                config.params,
                config.distance_fn,
                config.segment_capacity,
            )?
        };

        Ok(Self {
            config,
            published: PublishedSegments::new(mutable),
            next_segment_id: 0,
            merge_policy,
            last_merge_stats: None,
            generation: 0,
            pending_merge: None,
            pending_merge_dir: None,
        })
    }

    /// Create segment manager from an existing HNSWIndex with slot mapping
    ///
    /// Used for integrating parallel-built indexes into segment system.
    pub fn from_index(config: SegmentConfig, index: HNSWIndex, slots: &[u32]) -> Self {
        Self {
            config,
            published: PublishedSegments::new(MutableSegment::from_index(index, slots)),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
            generation: 0,
            pending_merge: None,
            pending_merge_dir: None,
        }
    }

    /// Create segment manager from parallel-built vectors
    ///
    /// Uses HNSWIndex::build_parallel for fast initial construction.
    /// Slots are sequential starting from 0.
    pub fn build_parallel(config: SegmentConfig, vectors: Vec<Vec<f32>>) -> Result<Self> {
        let index = HNSWIndex::build_parallel(
            config.dimensions,
            config.params,
            config.distance_fn,
            config.quantization,
            vectors,
        )?;
        let mutable = MutableSegment::from_index_sequential(index);

        Ok(Self {
            config,
            published: PublishedSegments::new(mutable),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
            generation: 0,
            pending_merge: None,
            pending_merge_dir: None,
        })
    }

    /// Create segment manager from parallel-built vectors with explicit slots
    ///
    /// Uses HNSWIndex::build_parallel for fast initial construction.
    pub fn build_parallel_with_slots(
        config: SegmentConfig,
        vectors: Vec<Vec<f32>>,
        slots: &[u32],
    ) -> Result<Self> {
        let index = HNSWIndex::build_parallel(
            config.dimensions,
            config.params,
            config.distance_fn,
            config.quantization,
            vectors,
        )?;
        let mutable = MutableSegment::from_index(index, slots);

        Ok(Self {
            config,
            published: PublishedSegments::new(mutable),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
            generation: 0,
            pending_merge: None,
            pending_merge_dir: None,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &SegmentConfig {
        &self.config
    }

    fn debug_assert_invariants(&self) {
        if let Some(ref pending) = self.pending_merge {
            debug_assert!(pending.source_count() >= 2);
            debug_assert!(pending.source_count() <= self.published.frozen.len());
            debug_assert!(
                self.published
                    .frozen
                    .iter()
                    .take(pending.source_count())
                    .map(|segment| segment.id())
                    .eq(pending.source_segment_ids().iter().copied())
            );
        }
    }

    fn new_mutable_segment(&self) -> Result<MutableSegment> {
        if self.config.quantization {
            MutableSegment::new_quantized(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
            )
        } else {
            MutableSegment::with_capacity(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
                self.config.segment_capacity,
            )
        }
    }

    fn publish_frozen_segment(&mut self, frozen: Arc<FrozenSegment>) {
        self.published.publish_frozen_segment(frozen);
    }

    fn finalize_published_frozen_change(&mut self) {
        self.apply_pending_merge_if_ready();
        self.try_start_background_merge();
        self.debug_assert_invariants();
    }

    fn clear_pending_merge_meta(&self) {
        if let Some(ref dir) = self.pending_merge_dir {
            let meta_path = dir.join("pending_merge.meta");
            if meta_path.exists()
                && let Err(e) = std::fs::remove_file(&meta_path)
            {
                tracing::warn!("Failed to remove pending_merge.meta: {e}");
            }
        }
    }

    fn discard_pending_merge_artifacts(&self, merged_segment_id: u64) {
        self.clear_pending_merge_meta();
        if let Some(ref dir) = self.pending_merge_dir {
            let segment_path = dir.join(format!("segment_{merged_segment_id}.bin"));
            if segment_path.exists()
                && let Err(e) = std::fs::remove_file(&segment_path)
            {
                tracing::warn!(
                    segment_id = merged_segment_id,
                    "Failed to remove discarded pending merge segment: {e}"
                );
            }
        }
    }

    fn publish_completed_pending_merge(
        &mut self,
        merged: Arc<FrozenSegment>,
        source_segment_ids: &[u64],
    ) -> Option<usize> {
        let prefix_matches = self
            .published
            .frozen
            .iter()
            .take(source_segment_ids.len())
            .map(|segment| segment.id())
            .eq(source_segment_ids.iter().copied());
        if !prefix_matches {
            tracing::warn!(
                merged_segment_id = merged.id(),
                expected_sources = ?source_segment_ids,
                "Discarding completed background merge because the published frozen prefix changed"
            );
            self.discard_pending_merge_artifacts(merged.id());
            return None;
        }

        let drain_count = source_segment_ids.len();
        self.published.publish_completed_merge(merged, drain_count);
        self.clear_pending_merge_meta();
        Some(drain_count)
    }

    /// Borrow the currently published mutable and frozen segment state.
    pub fn read_view(&self) -> SegmentReadView<'_> {
        self.debug_assert_invariants();
        SegmentReadView {
            mutable: &self.published.mutable,
            frozen: &self.published.frozen,
            config: &self.config,
            generation: self.generation,
        }
    }

    /// Number of frozen segments
    pub fn frozen_count(&self) -> usize {
        self.published.frozen_count()
    }

    /// Number of vectors in mutable segment
    pub fn mutable_len(&self) -> usize {
        self.published.mutable_len()
    }

    /// Total number of vectors across all segments
    pub fn len(&self) -> usize {
        self.published.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total memory usage across all segments (bytes)
    pub fn total_memory(&self) -> usize {
        self.published.total_memory()
    }

    /// Insert a vector with a specific slot
    ///
    /// Inserts into the mutable segment. If the segment reaches capacity,
    /// it's automatically frozen and a new mutable segment is created.
    /// The slot is the global RecordStore slot that will be returned in search results.
    pub fn insert_with_slot(&mut self, vector: &[f32], slot: u32) -> Result<u32> {
        // Freeze mutable if at capacity
        if self.published.mutable.is_full() {
            self.freeze_mutable()?;
        }

        self.published.mutable.insert_with_slot(vector, slot)
    }

    /// Insert a vector (slot == global vector count for consistency)
    ///
    /// Inserts into the mutable segment. If the segment reaches capacity,
    /// it's automatically frozen and a new mutable segment is created.
    /// The slot is assigned as the total vector count (global ID).
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        // Freeze mutable if at capacity
        if self.published.mutable.is_full() {
            self.freeze_mutable()?;
        }

        // Use global vector count as slot to maintain unique IDs across segments
        let slot = self.len() as u32;
        self.published.mutable.insert_with_slot(vector, slot)
    }

    /// Freeze current mutable segment
    ///
    /// After freezing, checks merge policy and triggers automatic merge
    /// if conditions are met.
    pub fn freeze_mutable(&mut self) -> Result<()> {
        self.debug_assert_invariants();

        let new_mutable = self.new_mutable_segment()?;
        if let Some(frozen) = self
            .published
            .rollover_mutable(new_mutable, self.next_segment_id)
        {
            self.next_segment_id += 1;
            self.publish_frozen_segment(Arc::new(frozen));
        }

        self.finalize_published_frozen_change();

        Ok(())
    }

    /// Search across all segments
    ///
    /// Searches mutable and all frozen segments, merging results.
    /// Frozen segments use sequential iteration for <4 segments, parallel for 4+.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SegmentSearchResult>> {
        self.read_view().search(query, k, ef)
    }

    /// Search across all segments with a filter predicate
    ///
    /// Uses ACORN-1 algorithm for efficient filtered search.
    /// The filter predicate receives global slots.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter_fn: F,
    ) -> Result<Vec<SegmentSearchResult>>
    where
        F: Fn(u32) -> bool + Sync,
    {
        self.read_view().search_with_filter(query, k, ef, filter_fn)
    }

    /// Force freeze current mutable segment
    ///
    /// Useful before persistence or when you want to ensure all data
    /// is in frozen segments.
    pub fn flush(&mut self) -> Result<()> {
        if !self.published.mutable.is_empty() {
            self.freeze_mutable()?;
        }
        Ok(())
    }

    /// Get dimensions
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Get HNSW params
    #[inline]
    pub fn params(&self) -> &HNSWParams {
        &self.config.params
    }

    /// Check if using quantization (SQ8)
    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.config.quantization
    }

    /// Get the generation counter (incremented on each save)
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get current merge policy
    pub fn merge_policy(&self) -> &MergePolicy {
        &self.merge_policy
    }

    /// Set merge policy
    pub fn set_merge_policy(&mut self, policy: MergePolicy) {
        self.merge_policy = policy;
    }

    /// Get statistics from last merge operation
    pub fn last_merge_stats(&self) -> Option<&MergeStats> {
        self.last_merge_stats.as_ref()
    }

    /// Add a parallel-built index as a new frozen segment with slot mapping
    pub fn add_frozen_from_index(&mut self, index: HNSWIndex, slots: &[u32]) {
        self.debug_assert_invariants();
        let mut index = index;
        index.remap_slots(slots);
        let frozen = self.create_merged_segment(index);
        self.publish_frozen_segment(frozen);
        self.finalize_published_frozen_change();
    }

    /// Set the directory for persisting background merge results.
    ///
    /// Call this after loading or creating segments to enable merge persistence.
    /// Background merges will write their result to this directory so it can be
    /// recovered if the process crashes before `drain_pending_merge()` runs.
    pub fn set_pending_merge_dir(&mut self, dir: impl Into<PathBuf>) {
        self.pending_merge_dir = Some(dir.into());
    }

    #[cfg(test)]
    fn build_test_merged_segment_from_frozen_prefix(&mut self, count: usize) -> Arc<FrozenSegment> {
        let segments_to_merge = self.published.frozen[0..count].to_vec();
        let (vectors, slots) = Self::collect_from_segments(&segments_to_merge);
        let (index, _) =
            Self::build_merged_index(&self.config, vectors, &slots).expect("build merged index");
        self.create_merged_segment(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SegmentConfig {
        SegmentConfig::new(4)
            .with_params(HNSWParams {
                m: 8,
                ef_construction: 50,
                ..Default::default()
            })
            .with_capacity(10) // Small capacity for testing
    }

    #[test]
    fn test_segment_manager_insert_and_search() {
        let config = test_config();
        let mut manager = SegmentManager::new(config).unwrap();

        // Insert vectors
        for i in 0..5 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        assert_eq!(manager.len(), 5);
        assert_eq!(manager.mutable_len(), 5);
        assert_eq!(manager.frozen_count(), 0);

        // Search
        let results = manager.search(&[2.0, 0.0, 0.0, 0.0], 3, 50).unwrap();
        assert_eq!(results.len(), 3);
        // Closest should be id=2
        assert_eq!(results[0].id, 2);
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_segment_manager_auto_freeze() {
        let config = test_config().with_capacity(5);
        let mut manager = SegmentManager::new(config).unwrap();

        // Insert more than capacity (5)
        for i in 0..7 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // Should have 1 frozen + 2 in mutable
        assert_eq!(manager.frozen_count(), 1);
        assert_eq!(manager.mutable_len(), 2);
        assert_eq!(manager.len(), 7);
    }

    #[test]
    fn test_segment_manager_search_across_segments() {
        let config = test_config().with_capacity(3);
        let mut manager = SegmentManager::new(config).unwrap();

        // Insert 9 vectors (will create 2 frozen segments + 3 in mutable)
        for i in 0..9 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        assert_eq!(manager.frozen_count(), 2);
        assert_eq!(manager.mutable_len(), 3);

        // Search should find vectors from all segments
        let results = manager.search(&[4.0, 0.0, 0.0, 0.0], 5, 50).unwrap();
        assert_eq!(results.len(), 5);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].distance <= results[i].distance);
        }
    }

    #[test]
    fn test_read_view_reports_visible_state() {
        let config = test_config().with_capacity(3);
        let mut manager = SegmentManager::new(config).unwrap();

        for i in 0..8 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        let view = manager.read_view();
        assert_eq!(view.frozen_count(), manager.frozen_count());
        assert_eq!(view.mutable_len(), manager.mutable_len());
        assert_eq!(view.len(), manager.len());
        assert_eq!(view.is_empty(), manager.is_empty());
        assert_eq!(view.generation(), manager.generation());
        assert_eq!(view.segment_capacity(), manager.config().segment_capacity);
        assert_eq!(view.total_memory(), manager.total_memory());
    }

    #[test]
    fn test_read_view_matches_manager_search() {
        let config = test_config().with_capacity(3);
        let mut manager = SegmentManager::new(config).unwrap();

        for i in 0..9 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        let query = [4.0, 0.0, 0.0, 0.0];
        let manager_results = manager.search(&query, 5, 50).unwrap();
        let view_results = manager.read_view().search(&query, 5, 50).unwrap();

        assert_eq!(view_results.len(), manager_results.len());
        for (view, manager) in view_results.iter().zip(manager_results.iter()) {
            assert_eq!(view.id, manager.id);
            assert_eq!(view.slot, manager.slot);
            assert_eq!(view.distance, manager.distance);
        }
    }

    #[test]
    fn test_segment_manager_flush() {
        let config = test_config();
        let mut manager = SegmentManager::new(config).unwrap();

        // Insert some vectors
        for i in 0..5 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // Before flush
        assert_eq!(manager.mutable_len(), 5);
        assert_eq!(manager.frozen_count(), 0);

        // Flush
        manager.flush().unwrap();

        // After flush
        assert_eq!(manager.mutable_len(), 0);
        assert_eq!(manager.frozen_count(), 1);
        assert_eq!(manager.len(), 5); // Total unchanged
    }

    #[test]
    fn test_segment_manager_empty() {
        let config = test_config();
        let manager = SegmentManager::new(config).unwrap();

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        let results = manager.search(&[0.0, 0.0, 0.0, 0.0], 10, 50).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_merge_policy_disabled() {
        let config = test_config().with_capacity(3);
        let mut manager =
            SegmentManager::with_merge_policy(config, MergePolicy::disabled()).unwrap();

        // Insert enough to create multiple frozen segments
        for i in 0..15 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // With disabled policy, should have multiple frozen segments
        assert!(
            manager.frozen_count() >= 2,
            "Should have multiple frozen segments"
        );
        assert!(
            !manager.should_merge(),
            "Disabled policy should not trigger merge"
        );
    }

    #[test]
    fn test_merge_policy_max_segments() {
        let config = test_config().with_capacity(3);
        let policy = MergePolicy {
            min_segments: 2,
            max_segments: 3,
            min_vectors: 1000, // High threshold to not trigger on vector count
            size_ratio_threshold: 100.0, // High to not trigger on ratio
            enabled: true,
            ..Default::default()
        };
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert enough to create 3 frozen segments (9 vectors / 3 capacity)
        // When we hit 3 frozen segments, auto-merge should kick in
        for i in 0..12 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // After auto-merge, should have fewer segments
        // Either merged down or got merged
        assert_eq!(manager.len(), 12, "Should still have all vectors");
    }

    #[test]
    fn test_merge_all_frozen_manually() {
        let config = test_config().with_capacity(5);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert to create 2 frozen segments
        for i in 0..12 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        assert_eq!(manager.frozen_count(), 2, "Should have 2 frozen segments");
        let total_before = manager.len();

        // Manually merge
        let stats = manager.merge_all_frozen().unwrap();
        assert!(stats.is_some(), "Should return merge stats");

        let stats = stats.unwrap();
        // Second segment gets merged into first, so merged count = second segment size = 5
        assert!(stats.vectors_merged > 0, "Should merge vectors");

        // After merge: should have 1 frozen segment (merged)
        assert_eq!(
            manager.frozen_count(),
            1,
            "Should have 1 merged frozen segment"
        );
        assert_eq!(manager.len(), total_before, "Total vectors unchanged");
    }

    #[test]
    fn test_merge_preserves_search() {
        let config = test_config().with_capacity(5);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert vectors
        for i in 0..15 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // Search before merge
        let query = [7.0, 0.0, 0.0, 0.0];
        let _results_before = manager.search(&query, 5, 50).unwrap();

        // Merge
        manager.merge_all_frozen().unwrap();

        // Search after merge - should still work
        let results_after = manager.search(&query, 5, 50).unwrap();
        assert_eq!(results_after.len(), 5, "Should still find 5 results");

        // First result should be close to query
        assert!(
            results_after[0].distance < 1.0,
            "Should find vector close to query"
        );
    }

    #[test]
    fn test_merge_segments_specific() {
        let config = test_config().with_capacity(3);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Create 3 frozen segments
        for i in 0..12 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        assert_eq!(manager.frozen_count(), 3, "Should have 3 frozen segments");

        // Merge only first two segments
        let stats = manager.merge_segments(&[0, 1]).unwrap();
        assert!(stats.is_some());

        // Should now have 2 frozen segments (merged one + original third)
        assert_eq!(
            manager.frozen_count(),
            2,
            "Should have 2 frozen after partial merge"
        );
    }

    #[test]
    fn test_publish_completed_pending_merge_preserves_appended_segments() {
        let config = test_config().with_capacity(3);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        for i in 0..9 {
            manager.insert(&vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        manager.flush().unwrap();

        assert_eq!(manager.frozen_count(), 3);
        let total_before = manager.len();
        let appended_segment_id = manager.published.frozen[2].id();
        let source_ids = vec![
            manager.published.frozen[0].id(),
            manager.published.frozen[1].id(),
        ];
        let merged = manager.build_test_merged_segment_from_frozen_prefix(2);
        let merged_id = merged.id();

        let drained = manager.publish_completed_pending_merge(merged, &source_ids);
        assert_eq!(drained, Some(2));
        assert_eq!(manager.frozen_count(), 2);
        assert_eq!(manager.published.frozen[0].id(), merged_id);
        assert_eq!(manager.published.frozen[1].id(), appended_segment_id);
        assert_eq!(manager.len(), total_before);
    }

    #[test]
    fn test_publish_completed_pending_merge_rejects_prefix_mismatch() {
        let config = test_config().with_capacity(3);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        for i in 0..9 {
            manager.insert(&vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        manager.flush().unwrap();

        assert_eq!(manager.frozen_count(), 3);
        let frozen_ids_before: Vec<u64> = manager
            .published
            .frozen
            .iter()
            .map(|segment| segment.id())
            .collect();
        let source_ids = vec![
            manager.published.frozen[1].id(),
            manager.published.frozen[0].id(),
        ];
        let merged = manager.build_test_merged_segment_from_frozen_prefix(2);

        let drained = manager.publish_completed_pending_merge(merged, &source_ids);
        assert_eq!(drained, None);
        let frozen_ids_after: Vec<u64> = manager
            .published
            .frozen
            .iter()
            .map(|segment| segment.id())
            .collect();
        assert_eq!(frozen_ids_after, frozen_ids_before);
    }

    #[test]
    fn test_merge_preserves_custom_slots() {
        // Test that merge preserves original slot mappings (critical for VectorStore integration)
        let config = test_config().with_capacity(5);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert with non-sequential custom slots (simulating VectorStore behavior)
        // Slots: 100, 200, 300, 400, 500 (segment 1)
        // Slots: 600, 700, 800, 900, 1000 (segment 2)
        for i in 0..10 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            let slot = ((i + 1) * 100) as u32;
            manager.insert_with_slot(&vector, slot).unwrap();
        }

        // Flush to ensure mutable becomes frozen
        manager.flush().unwrap();

        // Should have 2 frozen segments (5 each)
        assert_eq!(manager.frozen_count(), 2, "Should have 2 frozen segments");

        // Search before merge - find vector closest to [5, 0, 0, 0]
        let query = [5.0, 0.0, 0.0, 0.0];
        let results_before = manager.search(&query, 1, 50).unwrap();
        assert_eq!(results_before.len(), 1);
        let slot_before = results_before[0].slot;
        assert_eq!(
            slot_before, 600,
            "Should find slot 600 (vector [5, 0, 0, 0])"
        );

        // Merge all frozen segments
        let stats = manager.merge_all_frozen().unwrap();
        assert!(stats.is_some(), "Should return merge stats");

        // Should have 1 frozen segment after merge
        assert_eq!(
            manager.frozen_count(),
            1,
            "Should have 1 frozen after merge"
        );

        // Search after merge - should find same slot
        let results_after = manager.search(&query, 1, 50).unwrap();
        assert_eq!(results_after.len(), 1);
        let slot_after = results_after[0].slot;
        assert_eq!(
            slot_after, slot_before,
            "Slot should be preserved after merge: expected {}, got {}",
            slot_before, slot_after
        );

        // Verify all slots are preserved by searching for each vector
        for i in 0..10 {
            let q = [i as f32, 0.0, 0.0, 0.0];
            let r = manager.search(&q, 1, 50).unwrap();
            assert_eq!(r.len(), 1);
            let expected_slot = ((i + 1) * 100) as u32;
            assert_eq!(
                r[0].slot, expected_slot,
                "Vector {} should have slot {}, got {}",
                i, expected_slot, r[0].slot
            );
        }
    }

    #[test]
    fn test_should_merge_size_ratio() {
        let config = test_config().with_capacity(10);
        let policy = MergePolicy {
            min_segments: 2,
            max_segments: 100,
            min_vectors: 1_000_000,    // Won't trigger on count
            size_ratio_threshold: 2.0, // Will trigger if one segment is 2x another
            enabled: true,
            ..Default::default()
        };
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert 10 vectors and flush (creates segment with 10 vectors)
        for i in 0..10 {
            manager.insert(&vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
        }
        manager.flush().unwrap();

        // Insert 3 vectors and flush (creates segment with 3 vectors)
        for i in 0..3 {
            manager.insert(&vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        // Don't call flush() here - it would trigger freeze_mutable which auto-merges
        // Instead check the state
        assert_eq!(manager.mutable_len(), 3);
        assert_eq!(manager.frozen_count(), 1);

        // Manually call should_merge to test the logic
        // We need 2 frozen segments for ratio check
        manager.set_merge_policy(MergePolicy::disabled());
        manager.flush().unwrap();
        manager.set_merge_policy(MergePolicy {
            min_segments: 2,
            max_segments: 100,
            min_vectors: 1_000_000,
            size_ratio_threshold: 2.0,
            enabled: true,
            ..Default::default()
        });

        // Now have 2 frozen segments: 10 and 3 vectors
        // Ratio is 10/3 = 3.33 > 2.0
        assert!(manager.should_merge(), "Size ratio should trigger merge");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config().with_capacity(5);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert vectors
        for i in 0..12 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // Should have 2 frozen + some in mutable
        assert_eq!(manager.frozen_count(), 2);
        let total_before = manager.len();

        // Save
        manager.save(dir.path()).unwrap();

        // Load
        let loaded = SegmentManager::load(dir.path()).unwrap();

        // Verify
        assert_eq!(loaded.len(), total_before);
        assert_eq!(loaded.dimensions(), 4);
        assert_eq!(loaded.params().m, 8);

        // Search should work
        let results = loaded.search(&[5.0, 0.0, 0.0, 0.0], 3, 50).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].slot, 5); // Should find exact match (slot is the original ID)
    }

    #[test]
    fn test_save_load_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config();
        let mut manager = SegmentManager::new(config).unwrap();

        // Save empty manager
        manager.save(dir.path()).unwrap();

        // Load
        let loaded = SegmentManager::load(dir.path()).unwrap();
        assert_eq!(loaded.len(), 0);
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_save_load_preserves_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = SegmentConfig::new(128)
            .with_params(HNSWParams {
                m: 32,
                ef_construction: 200,
                max_level: 10,
                ..Default::default()
            })
            .with_distance(Metric::Cosine)
            .with_capacity(50_000);

        let policy = MergePolicy {
            min_segments: 3,
            max_segments: 10,
            min_vectors: 500,
            size_ratio_threshold: 5.0,
            enabled: true,
            ..Default::default()
        };

        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert some vectors
        for i in 0..100 {
            let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
            manager.insert(&vector).unwrap();
        }

        // Save and load
        manager.save(dir.path()).unwrap();
        let loaded = SegmentManager::load(dir.path()).unwrap();

        // Verify config preserved
        assert_eq!(loaded.dimensions(), 128);
        assert_eq!(loaded.params().m, 32);
        assert_eq!(loaded.params().ef_construction, 200);
        assert_eq!(loaded.config().segment_capacity, 50_000);

        // Verify merge policy preserved
        assert_eq!(loaded.merge_policy().min_segments, 3);
        assert_eq!(loaded.merge_policy().max_segments, 10);
        assert!(loaded.merge_policy().enabled);
    }

    #[test]
    fn test_save_load_search_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config().with_capacity(10);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert vectors
        for i in 0..25 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        // Search before save
        let query = [12.0, 0.0, 0.0, 0.0];
        let results_before = manager.search(&query, 5, 50).unwrap();

        // Save
        manager.save(dir.path()).unwrap();

        // Load
        let loaded = SegmentManager::load(dir.path()).unwrap();

        // Search after load
        let results_after = loaded.search(&query, 5, 50).unwrap();

        // Results should match (same IDs, similar distances)
        assert_eq!(results_before.len(), results_after.len());
        for (before, after) in results_before.iter().zip(results_after.iter()) {
            assert_eq!(before.id, after.id);
            assert!((before.distance - after.distance).abs() < 0.001);
        }
    }

    #[test]
    fn test_segment_manager_filtered_search() {
        let config = test_config().with_capacity(10);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert 25 vectors (will create 2 frozen + 5 in mutable)
        for i in 0..25 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }

        assert_eq!(manager.frozen_count(), 2);
        assert_eq!(manager.mutable_len(), 5);

        // Filter: only multiples of 5 (0, 5, 10, 15, 20)
        let results = manager
            .search_with_filter(&[10.0, 0.0, 0.0, 0.0], 3, 50, |slot| slot % 5 == 0)
            .unwrap();

        assert!(!results.is_empty());
        for r in &results {
            assert!(r.slot % 5 == 0, "slot {} should be multiple of 5", r.slot);
        }
        // Closest multiple of 5 to 10 is 10 itself
        assert_eq!(results[0].slot, 10);
    }

    #[test]
    fn test_segment_manager_filtered_search_across_segments() {
        let config = test_config().with_capacity(5);
        let policy = MergePolicy::disabled();
        let mut manager = SegmentManager::with_merge_policy(config, policy).unwrap();

        // Insert 15 vectors (3 frozen segments of 5 each)
        for i in 0..15 {
            let vector = vec![i as f32, 0.0, 0.0, 0.0];
            manager.insert(&vector).unwrap();
        }
        manager.flush().unwrap();

        assert_eq!(manager.frozen_count(), 3);

        // Filter: only slots in first segment (0-4)
        let results = manager
            .search_with_filter(&[2.0, 0.0, 0.0, 0.0], 3, 50, |slot| slot < 5)
            .unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.slot < 5, "slot {} should be < 5", r.slot);
        }
        // Closest to 2.0 in [0-4] is 2
        assert_eq!(results[0].slot, 2);
    }
}
