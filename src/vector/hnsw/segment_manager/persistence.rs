//! Segment manager persistence
//!
//! Save and load segment manager state to/from disk.

use super::{MergePolicy, SegmentConfig, SegmentManager};
use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment};
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
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
}
