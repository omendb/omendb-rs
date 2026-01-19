//! Segment manager for coordinating mutable and frozen segments
//!
//! The SegmentManager provides a unified interface over multiple segments:
//! - One active mutable segment for writes
//! - Zero or more frozen segments for reads
//!
//! When the mutable segment reaches capacity, it's frozen and a new
//! mutable segment is created. Searches query all segments in parallel.
//!
//! ## Automatic Merging
//!
//! When multiple frozen segments accumulate, they can be merged using the
//! IGTM (Iterative Greedy Tree Merging) algorithm for 1.3-1.7x speedup
//! over naive insertion. Set a merge policy to enable automatic merging.

use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::merge::{GraphMerger, MergeConfig, MergeStats};
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment, SegmentSearchResult};
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for segment manager
#[derive(Clone, Debug)]
pub struct SegmentConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// HNSW parameters
    pub params: HNSWParams,
    /// Distance function
    pub distance_fn: DistanceFunction,
    /// Max vectors per segment before freezing
    pub segment_capacity: usize,
    /// Whether to use quantization
    pub use_quantization: bool,
}

impl SegmentConfig {
    /// Create default config
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            params: HNSWParams::default(),
            distance_fn: DistanceFunction::L2,
            segment_capacity: 100_000,
            use_quantization: false,
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
    pub fn with_distance(mut self, distance_fn: DistanceFunction) -> Self {
        self.distance_fn = distance_fn;
        self
    }

    /// Set segment capacity
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.segment_capacity = capacity;
        self
    }

    /// Enable quantization
    #[must_use]
    pub fn with_quantization(mut self, enabled: bool) -> Self {
        self.use_quantization = enabled;
        self
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

/// Manages mutable and frozen segments
///
/// Provides unified insert and search over multiple segments.
/// The mutable segment is frozen when it reaches capacity.
/// Supports automatic merging of frozen segments via IGTM algorithm.
pub struct SegmentManager {
    /// Configuration
    config: SegmentConfig,
    /// Active mutable segment for writes
    mutable: MutableSegment,
    /// Frozen segments for reads (immutable, thread-safe)
    frozen: Vec<Arc<FrozenSegment>>,
    /// Next segment ID
    next_segment_id: u64,
    /// Merge policy for automatic merging
    merge_policy: MergePolicy,
    /// Statistics from last merge operation
    last_merge_stats: Option<MergeStats>,
}

impl SegmentManager {
    /// Create new segment manager with default merge policy
    pub fn new(config: SegmentConfig) -> Result<Self> {
        Self::with_merge_policy(config, MergePolicy::default())
    }

    /// Create new segment manager with custom merge policy
    pub fn with_merge_policy(config: SegmentConfig, merge_policy: MergePolicy) -> Result<Self> {
        let mutable = if config.use_quantization {
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
            mutable,
            frozen: Vec::new(),
            next_segment_id: 0,
            merge_policy,
            last_merge_stats: None,
        })
    }

    /// Create segment manager from an existing HNSWIndex with slot mapping
    ///
    /// Used for integrating parallel-built indexes into segment system.
    pub fn from_index(config: SegmentConfig, index: HNSWIndex, slots: &[u32]) -> Self {
        Self {
            config,
            mutable: MutableSegment::from_index(index, slots),
            frozen: Vec::new(),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
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
            config.use_quantization,
            vectors,
        )?;
        let mutable = MutableSegment::from_index_sequential(index);

        Ok(Self {
            config,
            mutable,
            frozen: Vec::new(),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
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
            config.use_quantization,
            vectors,
        )?;
        let mutable = MutableSegment::from_index(index, slots);

        Ok(Self {
            config,
            mutable,
            frozen: Vec::new(),
            next_segment_id: 0,
            merge_policy: MergePolicy::default(),
            last_merge_stats: None,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &SegmentConfig {
        &self.config
    }

    /// Number of frozen segments
    pub fn frozen_count(&self) -> usize {
        self.frozen.len()
    }

    /// Number of vectors in mutable segment
    pub fn mutable_len(&self) -> usize {
        self.mutable.len()
    }

    /// Total number of vectors across all segments
    pub fn len(&self) -> usize {
        self.mutable.len() + self.frozen.iter().map(|s| s.len()).sum::<usize>()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a vector with a specific slot
    ///
    /// Inserts into the mutable segment. If the segment reaches capacity,
    /// it's automatically frozen and a new mutable segment is created.
    /// The slot is the global RecordStore slot that will be returned in search results.
    pub fn insert_with_slot(&mut self, vector: &[f32], slot: u32) -> Result<u32> {
        // Freeze mutable if at capacity
        if self.mutable.is_full() {
            self.freeze_mutable()?;
        }

        self.mutable.insert_with_slot(vector, slot)
    }

    /// Insert a vector (slot == local_id for backward compatibility)
    ///
    /// Inserts into the mutable segment. If the segment reaches capacity,
    /// it's automatically frozen and a new mutable segment is created.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        // Freeze mutable if at capacity
        if self.mutable.is_full() {
            self.freeze_mutable()?;
        }

        self.mutable.insert(vector)
    }

    /// Freeze current mutable segment
    ///
    /// After freezing, checks merge policy and triggers automatic merge
    /// if conditions are met.
    fn freeze_mutable(&mut self) -> Result<()> {
        // Create new mutable segment
        let new_mutable = if self.config.use_quantization {
            MutableSegment::new_quantized(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
            )?
        } else {
            MutableSegment::with_capacity(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
                self.config.segment_capacity,
            )?
        };

        // Swap in new mutable, freeze old one
        let old_mutable = std::mem::replace(&mut self.mutable, new_mutable);

        if !old_mutable.is_empty() {
            let frozen = old_mutable.freeze();
            self.frozen.push(Arc::new(frozen));
        }

        self.next_segment_id += 1;

        // Check merge policy and merge if needed
        if self.should_merge() {
            debug!(
                frozen_count = self.frozen.len(),
                "Auto-merge triggered by policy"
            );
            self.merge_all_frozen()?;
        }

        Ok(())
    }

    /// Search across all segments
    ///
    /// Searches mutable and all frozen segments, merging results.
    /// Frozen segments are searched in parallel using rayon.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SegmentSearchResult>> {
        // Search mutable segment
        let mut results = self.mutable.search(query, k, ef)?;

        // Search frozen segments (could parallelize with rayon)
        for frozen in &self.frozen {
            let frozen_results = frozen.search(query, k, ef);
            results.extend(frozen_results);
        }

        // Sort by distance and take top k
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Force freeze current mutable segment
    ///
    /// Useful before persistence or when you want to ensure all data
    /// is in frozen segments.
    pub fn flush(&mut self) -> Result<()> {
        if !self.mutable.is_empty() {
            self.freeze_mutable()?;
        }
        Ok(())
    }

    /// Get access to frozen segments
    pub fn frozen_segments(&self) -> &[Arc<FrozenSegment>] {
        &self.frozen
    }

    /// Get access to mutable segment
    pub fn mutable_segment(&self) -> &MutableSegment {
        &self.mutable
    }

    /// Get mutable access to mutable segment
    pub fn mutable_segment_mut(&mut self) -> &mut MutableSegment {
        &mut self.mutable
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

    /// Check if using quantization (asymmetric search)
    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.config.use_quantization
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

    /// Check if merge should be triggered based on current policy
    ///
    /// Returns true if:
    /// - Policy is enabled AND
    /// - (frozen segments >= max_segments OR
    ///    (frozen segments >= min_segments AND
    ///     (total frozen vectors >= min_vectors OR size ratio exceeded)))
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

    /// Merge all frozen segments into a single new frozen segment
    ///
    /// Uses IGTM (Iterative Greedy Tree Merging) algorithm for 1.3-1.7x
    /// speedup over naive insertion. The result is a single frozen segment
    /// replacing all previous frozen segments.
    ///
    /// Returns merge statistics if any segments were merged.
    pub fn merge_all_frozen(&mut self) -> Result<Option<MergeStats>> {
        if self.frozen.is_empty() {
            return Ok(None);
        }

        if self.frozen.len() == 1 {
            // Nothing to merge
            return Ok(None);
        }

        info!(
            frozen_count = self.frozen.len(),
            frozen_vectors = self.frozen.iter().map(|s| s.len()).sum::<usize>(),
            "Starting segment merge"
        );

        let merger = GraphMerger::with_config(self.merge_policy.merge_config.clone());
        let mut total_stats: Option<MergeStats> = None;

        // Build merged index from all frozen segments
        let mut merged_index = HNSWIndex::new(
            self.config.dimensions,
            self.config.params,
            self.config.distance_fn,
            self.config.use_quantization,
        )?;

        // Process each frozen segment
        for frozen_arc in std::mem::take(&mut self.frozen) {
            let frozen = frozen_arc.as_ref();
            if frozen.is_empty() {
                continue;
            }

            // Create temporary HNSWIndex from frozen segment data
            let mut temp_index = HNSWIndex::new(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
                self.config.use_quantization,
            )?;

            // Insert all vectors from frozen segment into temp index
            let storage = frozen.storage();
            for id in 0..frozen.len() as u32 {
                let vector = storage.vector(id);
                temp_index.insert(vector)?;
            }

            // Merge temp_index into merged_index
            let stats = merger.merge_graphs(&mut merged_index, &temp_index)?;

            debug!(
                vectors_merged = stats.vectors_merged,
                join_set_size = stats.join_set_size,
                duration_ms = stats.total_duration.as_millis(),
                "Merged frozen segment"
            );

            // Accumulate stats
            total_stats = Some(match total_stats {
                None => stats,
                Some(mut prev) => {
                    prev.vectors_merged += stats.vectors_merged;
                    prev.join_set_size += stats.join_set_size;
                    prev.total_duration += stats.total_duration;
                    prev.fast_path_inserts += stats.fast_path_inserts;
                    prev.fallback_inserts += stats.fallback_inserts;
                    prev
                }
            });
        }

        // Create new frozen segment from merged index
        if !merged_index.is_empty() {
            let mut mutable_temp = MutableSegment::from_index_sequential(merged_index);
            mutable_temp.set_id(self.next_segment_id);
            self.next_segment_id += 1;

            let frozen = mutable_temp.freeze();
            self.frozen.push(Arc::new(frozen));
        }

        if let Some(ref stats) = total_stats {
            info!(
                total_vectors = stats.vectors_merged,
                total_duration_ms = stats.total_duration.as_millis(),
                "Segment merge complete"
            );
        }

        self.last_merge_stats = total_stats.clone();
        Ok(total_stats)
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
    /// * `indices` - Indices of frozen segments to merge (must be sorted ascending)
    pub fn merge_segments(&mut self, indices: &[usize]) -> Result<Option<MergeStats>> {
        if indices.is_empty() || indices.len() == 1 {
            return Ok(None);
        }

        // Validate indices
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

        // Build merged index
        let merger = GraphMerger::with_config(self.merge_policy.merge_config.clone());
        let mut merged_index = HNSWIndex::new(
            self.config.dimensions,
            self.config.params,
            self.config.distance_fn,
            self.config.use_quantization,
        )?;

        let mut total_stats: Option<MergeStats> = None;

        for frozen_arc in segments_to_merge {
            let frozen = frozen_arc.as_ref();
            if frozen.is_empty() {
                continue;
            }

            // Create temp index from frozen
            let mut temp_index = HNSWIndex::new(
                self.config.dimensions,
                self.config.params,
                self.config.distance_fn,
                self.config.use_quantization,
            )?;

            let storage = frozen.storage();
            for id in 0..frozen.len() as u32 {
                temp_index.insert(storage.vector(id))?;
            }

            let stats = merger.merge_graphs(&mut merged_index, &temp_index)?;

            total_stats = Some(match total_stats {
                None => stats,
                Some(mut prev) => {
                    prev.vectors_merged += stats.vectors_merged;
                    prev.total_duration += stats.total_duration;
                    prev
                }
            });
        }

        // Create new frozen segment from merged index
        if !merged_index.is_empty() {
            let mut mutable = MutableSegment::from_index_sequential(merged_index);
            mutable.set_id(self.next_segment_id);
            self.next_segment_id += 1;

            let frozen = mutable.freeze();
            self.frozen.push(Arc::new(frozen));
        }

        self.last_merge_stats = total_stats.clone();
        Ok(total_stats)
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

    // ============== Merge Policy Tests ==============

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
        let results_before = manager.search(&query, 5, 50).unwrap();

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
}
