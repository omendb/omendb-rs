//! Segment manager persistence
//!
//! Save and load segment manager state to/from disk.

use super::{MergePolicy, PublishedSegments, SegmentConfig, SegmentManager};
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment};
use crate::vector::hnsw::types::{HNSWParams, Metric};
use std::sync::Arc;
use tracing::{debug, info};

impl SegmentManager {
    /// Build manifest JSON for saving
    pub(super) fn build_manifest(&self, segment_ids: &[u64]) -> serde_json::Value {
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
            "use_quantization": self.config.quantization,
            "quantization_mode": if self.config.quantization { "sq8" } else { "none" },
            "generation": self.generation,
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
    pub(super) fn parse_config(manifest: &serde_json::Value) -> SegmentConfig {
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
            "Cosine" => Metric::Cosine,
            "NegativeDotProduct" | "InnerProduct" => Metric::InnerProduct,
            _ => Metric::L2,
        };
        let segment_capacity = manifest["segment_capacity"].as_u64().unwrap_or(100_000) as usize;

        let quantization = match manifest["quantization_mode"].as_str() {
            Some("sq8") => true,
            _ => manifest["use_quantization"].as_bool().unwrap_or(false),
        };

        SegmentConfig {
            dimensions,
            params,
            distance_fn,
            segment_capacity,
            quantization,
        }
    }

    /// Parse merge policy from manifest JSON
    pub(super) fn parse_merge_policy(manifest: &serde_json::Value) -> MergePolicy {
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

        // Increment generation for staleness detection
        self.generation += 1;

        // Build manifest
        let segment_ids: Vec<u64> = self.published.frozen.iter().map(|s| s.id()).collect();
        let manifest = self.build_manifest(&segment_ids);

        // Write manifest atomically (tmp + fsync + rename)
        let manifest_path = dir.join("manifest.json");
        let manifest_tmp = dir.join("manifest.json.tmp");
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to serialize manifest: {e}"
            ))
        })?;
        let mut file = std::fs::File::create(&manifest_tmp).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to create manifest temp file: {e}"
            ))
        })?;
        file.write_all(&manifest_bytes)?;
        file.sync_all()?;
        fs::rename(&manifest_tmp, &manifest_path).map_err(|e| {
            let _ = fs::remove_file(&manifest_tmp);
            crate::vector::hnsw::error::HNSWError::Storage(format!(
                "Failed to rename manifest file: {e}"
            ))
        })?;
        // Fsync parent directory to ensure rename is durable
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }

        // Save each frozen segment (skip if file already exists — frozen segments are immutable)
        for segment in &self.published.frozen {
            let segment_path = dir.join(format!("segment_{}.bin", segment.id()));
            if !segment_path.exists() {
                segment.save(&segment_path)?;
                debug!(segment_id = segment.id(), path = %segment_path.display(), "Saved segment");
            }
        }

        // Clean orphan segment files and stale temp files
        if let Ok(entries) = fs::read_dir(dir) {
            let active_ids: std::collections::HashSet<u64> = segment_ids.iter().copied().collect();
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                // Clean stale temp files from interrupted saves
                if name_str.ends_with(".bin.tmp") || name_str.ends_with(".json.tmp") {
                    let _ = fs::remove_file(entry.path());
                    tracing::debug!(file = %name_str, "Removed stale temp file");
                    continue;
                }

                // Clean orphan segment files
                if let Some(id_str) = name_str
                    .strip_prefix("segment_")
                    .and_then(|s| s.strip_suffix(".bin"))
                    && let Ok(id) = id_str.parse::<u64>()
                    && !active_ids.contains(&id)
                {
                    if let Err(e) = fs::remove_file(entry.path()) {
                        tracing::warn!(
                            file = %name_str,
                            error = %e,
                            "Failed to remove orphan segment file"
                        );
                    } else {
                        tracing::debug!(
                            file = %name_str,
                            "Removed orphan segment file"
                        );
                    }
                }
            }
        }

        info!(
            segments = self.published.frozen.len(),
            total_vectors = self.len(),
            "Segment manager saved"
        );
        Ok(())
    }

    /// Shared load implementation parameterized by segment loader
    fn load_with<P, F>(dir: P, loader: F) -> Result<Self>
    where
        P: AsRef<std::path::Path>,
        F: Fn(&std::path::Path) -> Result<FrozenSegment>,
    {
        use std::fs;

        let dir = dir.as_ref();

        let manifest_path = dir.join("manifest.json");
        let manifest_bytes = fs::read(&manifest_path).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!("Failed to read manifest: {e}"))
        })?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).map_err(|e| {
            crate::vector::hnsw::error::HNSWError::Storage(format!("Failed to parse manifest: {e}"))
        })?;

        let config = Self::parse_config(&manifest);
        let merge_policy = Self::parse_merge_policy(&manifest);
        let next_segment_id = manifest["next_segment_id"].as_u64().unwrap_or(0);
        let generation = manifest["generation"].as_u64().unwrap_or(0);

        let segment_ids: Vec<u64> = manifest["segment_ids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(serde_json::Value::as_u64).collect())
            .unwrap_or_default();

        let mut frozen = Vec::with_capacity(segment_ids.len());
        for seg_id in segment_ids {
            let segment_path = dir.join(format!("segment_{seg_id}.bin"));
            let segment = loader(&segment_path)?;
            frozen.push(Arc::new(segment));
            debug!(segment_id = seg_id, "Loaded segment");
        }

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

        let total_vectors: usize = frozen.iter().map(|s| s.len()).sum();
        info!(
            segments = frozen.len(),
            total_vectors, "Segment manager loaded"
        );

        let mut manager = Self {
            config,
            published: PublishedSegments::from_parts(mutable, frozen),
            next_segment_id,
            merge_policy,
            last_merge_stats: None,
            generation,
            pending_merge: None,
            pending_merge_dir: None,
        };

        // Apply any background merge that completed before the last crash
        manager.apply_pending_merge_from_disk(dir.as_ref());

        Ok(manager)
    }

    /// Apply a persisted pending merge from disk if it matches current segments.
    ///
    /// Called during load to recover background merges that completed before a crash.
    /// If `pending_merge.meta` exists and its source segment IDs match the currently
    /// loaded segments, loads the merged segment and replaces the source segments.
    /// Otherwise, cleans up the stale meta file and segment.
    fn apply_pending_merge_from_disk(&mut self, dir: &std::path::Path) {
        let meta_path = dir.join("pending_merge.meta");
        if !meta_path.exists() {
            return;
        }

        // Returns Ok(None) on success, Ok(Some(orphan_id)) on signature mismatch (caller
        // removes the orphan segment), Err on hard failure.
        let result = (|| -> Result<Option<u64>> {
            let meta_bytes = std::fs::read(&meta_path).map_err(|e| {
                crate::vector::hnsw::error::HNSWError::Storage(format!(
                    "Failed to read pending_merge.meta: {e}"
                ))
            })?;
            let meta: serde_json::Value = serde_json::from_slice(&meta_bytes).map_err(|e| {
                crate::vector::hnsw::error::HNSWError::Storage(format!(
                    "Failed to parse pending_merge.meta: {e}"
                ))
            })?;

            let source_ids: Vec<u64> = meta["source_ids"]
                .as_array()
                .map(|arr| arr.iter().filter_map(serde_json::Value::as_u64).collect())
                .unwrap_or_default();
            let total_vectors = meta["total_vectors"].as_u64().unwrap_or(0) as usize;
            let merged_segment_id = meta["merged_segment_id"].as_u64().unwrap_or(u64::MAX);

            // Verify source segments are still present with matching total vector count
            let current_ids: std::collections::HashSet<u64> =
                self.published.frozen.iter().map(|s| s.id()).collect();
            let current_total = self.published.frozen_total_len();

            let source_ids_match = source_ids.iter().all(|id| current_ids.contains(id));
            let vectors_match = current_total == total_vectors;

            if !source_ids_match || !vectors_match {
                tracing::info!("Pending merge signature mismatch, discarding stale merge");
                return Ok(Some(merged_segment_id));
            }

            // Load the merged segment
            let segment_path = dir.join(format!("segment_{merged_segment_id}.bin"));
            #[cfg(feature = "mmap")]
            let merged = FrozenSegment::load_mmap(&segment_path)?;
            #[cfg(not(feature = "mmap"))]
            let merged = FrozenSegment::load(&segment_path)?;

            // Apply: remove source segments, insert merged
            self.published
                .replace_frozen_by_id(&source_ids, Arc::new(merged));

            // Advance next_segment_id past the merged segment's ID. The manifest stores
            // next_segment_id as of the last flush before the merge started, so it may
            // be <= merged_segment_id. Without this, the next freeze would reuse the
            // merged segment's ID, causing path collisions.
            self.next_segment_id = self
                .next_segment_id
                .max(merged_segment_id.saturating_add(1));

            tracing::info!(
                merged_segments = source_ids.len(),
                merged_segment_id,
                "Applied persisted background merge on load"
            );

            Ok(None)
        })();

        match result {
            Ok(None) => {
                // Successfully applied — remove the meta file (segment is now a regular segment)
                let _ = std::fs::remove_file(&meta_path);
            }
            Ok(Some(orphan_id)) => {
                // Signature mismatch — remove orphan merged segment and meta
                let _ = std::fs::remove_file(dir.join(format!("segment_{orphan_id}.bin")));
                let _ = std::fs::remove_file(&meta_path);
            }
            Err(e) => {
                tracing::warn!("Failed to apply persisted background merge: {e}");
                let _ = std::fs::remove_file(&meta_path);
            }
        }
    }

    /// Load segment manager with memory-mapped frozen segments
    ///
    /// Like `load()`, but uses `FrozenSegment::load_mmap()` for zero-copy access.
    /// Segment data stays on disk and is paged in on demand by the OS.
    #[cfg(feature = "mmap")]
    pub fn load_mmap<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        info!(path = %dir.as_ref().display(), "Loading segment manager (mmap)");
        Self::load_with(dir, |path| FrozenSegment::load_mmap(path))
    }

    /// Load segment manager from a directory
    ///
    /// Loads the manifest and all segment files, recreating the manager state.
    pub fn load<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        info!(path = %dir.as_ref().display(), "Loading segment manager");
        Self::load_with(dir, |path| FrozenSegment::load(path))
    }
}
