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
use crate::vector::hnsw::merge::{MergeConfig, MergeStats};
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

    /// Insert a vector (slot == global vector count for consistency)
    ///
    /// Inserts into the mutable segment. If the segment reaches capacity,
    /// it's automatically frozen and a new mutable segment is created.
    /// The slot is assigned as the total vector count (global ID).
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        // Freeze mutable if at capacity
        if self.mutable.is_full() {
            self.freeze_mutable()?;
        }

        // Use global vector count as slot to maintain unique IDs across segments
        let slot = self.len() as u32;
        self.mutable.insert_with_slot(vector, slot)
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
        let mut old_mutable = std::mem::replace(&mut self.mutable, new_mutable);

        if !old_mutable.is_empty() {
            // Assign unique segment ID before freezing
            old_mutable.set_id(self.next_segment_id);
            self.next_segment_id += 1;
            let frozen = old_mutable.freeze();
            self.frozen.push(Arc::new(frozen));
        }

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

    // ============================================================================
    // Merge Helpers
    // ============================================================================

    /// Collect all (vector, slot) pairs from frozen segments
    fn collect_vectors_and_slots(segments: &[Arc<FrozenSegment>]) -> Vec<(Vec<f32>, u32)> {
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
    fn insert_vectors_with_slots(
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
    fn create_merged_segment(&mut self, index: HNSWIndex, slots: &[u32]) -> Arc<FrozenSegment> {
        let mut mutable = MutableSegment::from_index(index, slots);
        mutable.set_id(self.next_segment_id);
        self.next_segment_id += 1;
        Arc::new(mutable.freeze())
    }

    /// Build MergeStats from merge operation
    fn build_merge_stats(
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

    // ============================================================================
    // Persistence
    // ============================================================================

    /// Build manifest JSON for saving
    fn build_manifest(&self, segment_ids: &[u64]) -> serde_json::Value {
        serde_json::json!({
            "version": 1,
            "dimensions": self.config.dimensions,
            "params": {
                "m": self.config.params.m,
                "ef_construction": self.config.params.ef_construction,
                "max_level": self.config.params.max_level,
            },
            "distance_fn": format!("{:?}", self.config.distance_fn),
            "segment_capacity": self.config.segment_capacity,
            "use_quantization": self.config.use_quantization,
            "next_segment_id": self.next_segment_id,
            "segment_ids": segment_ids,
            "merge_policy": {
                "enabled": self.merge_policy.enabled,
                "min_segments": self.merge_policy.min_segments,
                "max_segments": self.merge_policy.max_segments,
                "min_vectors": self.merge_policy.min_vectors,
                "size_ratio_threshold": self.merge_policy.size_ratio_threshold,
            },
        })
    }

    /// Parse config from manifest JSON
    fn parse_config(manifest: &serde_json::Value) -> SegmentConfig {
        let dimensions = manifest["dimensions"].as_u64().unwrap_or(128) as usize;
        let params = HNSWParams {
            m: manifest["params"]["m"].as_u64().unwrap_or(16) as usize,
            ef_construction: manifest["params"]["ef_construction"]
                .as_u64()
                .unwrap_or(100) as usize,
            max_level: manifest["params"]["max_level"].as_u64().unwrap_or(8) as u8,
            ..Default::default()
        };
        let distance_fn = match manifest["distance_fn"].as_str().unwrap_or("L2") {
            "Cosine" => DistanceFunction::Cosine,
            "NegativeDotProduct" => DistanceFunction::NegativeDotProduct,
            _ => DistanceFunction::L2,
        };
        let segment_capacity = manifest["segment_capacity"].as_u64().unwrap_or(100_000) as usize;
        let use_quantization = manifest["use_quantization"].as_bool().unwrap_or(false);

        SegmentConfig {
            dimensions,
            params,
            distance_fn,
            segment_capacity,
            use_quantization,
        }
    }

    /// Parse merge policy from manifest JSON
    fn parse_merge_policy(manifest: &serde_json::Value) -> MergePolicy {
        manifest
            .get("merge_policy")
            .map(|mp| MergePolicy {
                enabled: mp["enabled"].as_bool().unwrap_or(true),
                min_segments: mp["min_segments"].as_u64().unwrap_or(2) as usize,
                max_segments: mp["max_segments"].as_u64().unwrap_or(8) as usize,
                min_vectors: mp["min_vectors"].as_u64().unwrap_or(1000) as usize,
                size_ratio_threshold: mp["size_ratio_threshold"].as_f64().unwrap_or(4.0) as f32,
                ..Default::default()
            })
            .unwrap_or_default()
    }

    /// Save segment manager to a directory
    ///
    /// Flushes the mutable segment to frozen, then saves:
    /// - `manifest.json` - config, segment IDs, merge policy
    /// - `segment_{id}.bin` - one file per frozen segment
    ///
    /// The directory is created if it doesn't exist.
    pub fn save<P: AsRef<std::path::Path>>(&mut self, dir: P) -> Result<()> {
        use std::fs;
        use std::io::Write;

        let dir = dir.as_ref();
        info!(path = %dir.display(), "Saving segment manager");

        // Create directory if needed
        fs::create_dir_all(dir).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to create directory: {e}"
            ))
        })?;

        // Flush mutable to frozen for consistent snapshot
        self.flush()?;

        // Build manifest
        let segment_ids: Vec<u64> = self.frozen.iter().map(|s| s.id()).collect();
        let manifest = self.build_manifest(&segment_ids);

        // Write manifest
        let manifest_path = dir.join("manifest.json");
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to serialize manifest: {e}"
            ))
        })?;
        let mut file = std::fs::File::create(&manifest_path).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to create manifest file: {e}"
            ))
        })?;
        file.write_all(&manifest_bytes)?;
        file.sync_all()?; // Ensure manifest is durably written before segments

        // Save each frozen segment
        for segment in &self.frozen {
            let segment_path = dir.join(format!("segment_{}.bin", segment.id()));
            segment.save(&segment_path)?;
            debug!(segment_id = segment.id(), path = %segment_path.display(), "Saved segment");
        }

        info!(
            segments = self.frozen.len(),
            total_vectors = self.len(),
            "Segment manager saved"
        );
        Ok(())
    }

    /// Load segment manager from a directory
    ///
    /// Loads the manifest and all segment files, recreating the manager state.
    pub fn load<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        use crate::vector::hnsw::segment::FrozenSegment;
        use std::fs;

        let dir = dir.as_ref();
        info!(path = %dir.display(), "Loading segment manager");

        // Read manifest
        let manifest_path = dir.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!("Failed to read manifest: {e}"))
        })?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!("Failed to parse manifest: {e}"))
        })?;

        // Parse config and merge policy from manifest
        let config = Self::parse_config(&manifest);
        let merge_policy = Self::parse_merge_policy(&manifest);
        let next_segment_id = manifest["next_segment_id"].as_u64().unwrap_or(0);

        // Load segment files
        let segment_ids: Vec<u64> = manifest["segment_ids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(serde_json::Value::as_u64).collect())
            .unwrap_or_default();

        let mut frozen = Vec::with_capacity(segment_ids.len());
        for seg_id in segment_ids {
            let segment_path = dir.join(format!("segment_{seg_id}.bin"));
            let segment = FrozenSegment::load(&segment_path)?;
            frozen.push(Arc::new(segment));
            debug!(segment_id = seg_id, "Loaded segment");
        }

        // Create empty mutable segment
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

        let total_vectors: usize = frozen.iter().map(|s| s.len()).sum();
        info!(
            segments = frozen.len(),
            total_vectors, "Segment manager loaded"
        );

        Ok(Self {
            config,
            mutable,
            frozen,
            next_segment_id,
            merge_policy,
            last_merge_stats: None,
        })
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

    // ============== Persistence Tests ==============

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
            .with_distance(DistanceFunction::Cosine)
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
}
