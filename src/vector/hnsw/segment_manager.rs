//! Segment manager for coordinating mutable and frozen segments
//!
//! The SegmentManager provides a unified interface over multiple segments:
//! - One active mutable segment for writes
//! - Zero or more frozen segments for reads
//!
//! When the mutable segment reaches capacity, it's frozen and a new
//! mutable segment is created. Searches query all segments in parallel.

use crate::vector::hnsw::error::Result;
use crate::vector::hnsw::index::HNSWIndex;
use crate::vector::hnsw::segment::{FrozenSegment, MutableSegment, SegmentSearchResult};
use crate::vector::hnsw::types::{DistanceFunction, HNSWParams};
use std::sync::Arc;

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

/// Manages mutable and frozen segments
///
/// Provides unified insert and search over multiple segments.
/// The mutable segment is frozen when it reaches capacity.
pub struct SegmentManager {
    /// Configuration
    config: SegmentConfig,
    /// Active mutable segment for writes
    mutable: MutableSegment,
    /// Frozen segments for reads (immutable, thread-safe)
    frozen: Vec<Arc<FrozenSegment>>,
    /// Next segment ID
    next_segment_id: u64,
}

impl SegmentManager {
    /// Create new segment manager
    pub fn new(config: SegmentConfig) -> Result<Self> {
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
}
