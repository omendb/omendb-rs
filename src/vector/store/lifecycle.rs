use anyhow::Result;
use rustc_hash::FxHashSet;

use super::VectorStore;
use super::{MetadataIndex, SegmentManager, Vector};

impl VectorStore {
    /// Insert batch of vectors in parallel
    ///
    /// NOTE: This method generates synthetic IDs for the vectors.
    /// For explicit IDs, use `set_batch` instead.
    pub fn batch_insert(&mut self, vectors: Vec<Vector>) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let dimensions = self.dimensions();
        for (i, vector) in vectors.iter().enumerate() {
            if vector.dim() != dimensions {
                anyhow::bail!(
                    "Vector {} dimension mismatch: expected {}, got {}",
                    i,
                    dimensions,
                    vector.dim()
                );
            }
        }

        // Insert into RecordStore with generated IDs
        let mut all_slots = Vec::with_capacity(vectors.len());
        let base_slot = self.records.slot_count();

        for (i, vector) in vectors.iter().enumerate() {
            let id = format!("_batch_{}", base_slot + i as u32);
            let slot = self.records.set(id, vector.data.clone(), None)?;
            all_slots.push(slot as usize);
        }

        // Build or extend segments
        let vector_data: Vec<Vec<f32>> = vectors.iter().map(|v| v.data.clone()).collect();
        let slots: Vec<u32> = all_slots.iter().map(|&s| s as u32).collect();

        if self.segments.read().is_none() {
            // Build new segment with parallel construction
            let config = self.segment_config(dimensions);

            let mut segs = SegmentManager::build_parallel_with_slots(config, vector_data, &slots)
                .map_err(|e| anyhow::anyhow!("Segment build failed: {e}"))?;
            if let Some(ref path) = self.storage_path {
                segs.set_pending_merge_dir(super::persistence::segments_dir_for(path));
            }
            *self.segments.write() = Some(segs);
        } else if let Some(ref mut segments) = *self.segments.write() {
            // Insert into existing segments
            for (vector, &slot) in vector_data.iter().zip(slots.iter()) {
                segments
                    .insert_with_slot(vector, slot)
                    .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
            }
        }

        Ok(all_slots)
    }

    /// Rebuild HNSW index from existing vectors.
    pub fn rebuild_index(&self) -> Result<()> {
        let _lock = self.write_lock.lock();
        self.rebuild_index_locked()
    }

    pub(super) fn rebuild_index_locked(&self) -> Result<()> {
        if self.records.is_empty() {
            return Ok(());
        }

        // Collect live vectors and their slots
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(self.records.len() as usize);
        let mut slots: Vec<u32> = Vec::with_capacity(self.records.len() as usize);
        for (slot, record) in self.records.iter_live() {
            vectors.push(record.vector.clone());
            slots.push(slot);
        }

        // Build segment config
        let dims = self.dimensions();
        let config = self.segment_config(dims);

        // Rebuild with parallel construction
        let mut segs = SegmentManager::build_parallel_with_slots(config, vectors, &slots)
            .map_err(|e| anyhow::anyhow!("Segment rebuild failed: {e}"))?;
        if let Some(ref path) = self.storage_path {
            segs.set_pending_merge_dir(super::persistence::segments_dir_for(path));
        }
        *self.segments.write() = Some(segs);

        Ok(())
    }

    /// Merge another `VectorStore` into this one using IGTM algorithm
    pub fn merge_from(&mut self, other: &VectorStore) -> Result<usize> {
        self.merge_from_with_prefix(other, None)
    }

    /// Merge vectors from another store with optional key prefix.
    ///
    /// When `key_prefix` is provided, all IDs from the source store are
    /// prefixed with it (e.g., `"subdir/"` turns `"foo.py"` into `"subdir/foo.py"`).
    pub fn merge_from_with_prefix(
        &mut self,
        other: &VectorStore,
        key_prefix: Option<&str>,
    ) -> Result<usize> {
        if other.dimensions() != self.dimensions() {
            anyhow::bail!(
                "Dimension mismatch: self={}, other={}",
                self.dimensions(),
                other.dimensions()
            );
        }

        if other.records.is_empty() {
            return Ok(0);
        }

        let mut merged_count = 0;
        let mut merged_slots: Vec<(u32, String)> = Vec::new();

        // Merge records, skipping conflicts
        for (slot, record) in other.records.iter_live() {
            let id = if let Some(prefix) = key_prefix {
                format!("{prefix}{}", record.id)
            } else {
                record.id.clone()
            };

            // Skip if ID already exists in self
            if self.records.get_slot(&id).is_some() {
                continue;
            }

            // Insert into our RecordStore
            self.records
                .set(id.clone(), record.vector.clone(), record.metadata.clone())?;
            merged_slots.push((slot, id));
            merged_count += 1;
        }

        // Copy sparse vectors for merged IDs
        if let Some(other_sparse) = other.sparse_index.read().as_ref()
            && !other_sparse.is_empty()
        {
            if self.sparse_index.read().is_none() {
                *self.sparse_index.write() = Some(crate::vector::sparse::SparseIndex::new());
            }
            let mut self_sparse = self.sparse_index.write();
            let self_sparse = self_sparse.as_mut().unwrap();
            for (other_slot, id) in &merged_slots {
                if let Some(sv) = other_sparse.get(*other_slot)
                    && let Some(new_slot) = self.records.get_slot(id)
                {
                    self_sparse.insert(new_slot, sv);
                }
            }
        }

        // Rebuild index after merge to ensure consistency
        self.rebuild_index()?;

        Ok(merged_count)
    }

    /// Check if index needs to be rebuilt
    #[inline]
    #[must_use]
    pub fn needs_index_rebuild(&self) -> bool {
        self.segments.read().is_none() && self.records.len() > 100
    }

    /// Ensure HNSW index is ready for search
    pub fn ensure_index_ready(&self) -> Result<()> {
        if !self.needs_index_rebuild() {
            return Ok(());
        }

        let _lock = self.write_lock.lock();
        if self.needs_index_rebuild() {
            self.rebuild_index_locked()?;
        }

        Ok(())
    }

    /// Optimize index for cache-efficient search
    ///
    /// Reorders graph nodes using BFS traversal to improve memory locality.
    /// Nodes that are frequently accessed together during search will be stored
    /// adjacently in memory, reducing cache misses and improving QPS.
    ///
    /// Call this after loading/building the index and before querying for best results.
    /// Based on NeurIPS 2021 "Graph Reordering for Cache-Efficient Near Neighbor Search".
    ///
    /// Returns the number of nodes reordered, or 0 if index is empty/not initialized.
    ///
    /// For segment-based storage, this merges all frozen segments into one
    /// for better search locality. Returns the number of vectors in the merged segment.
    pub fn optimize(&self) -> Result<usize> {
        let _lock = self.write_lock.lock();

        // Compact first if there are pending deletes to ensure consistent slot state
        if self.records.deleted_count() > 0 {
            self.compact_locked()?;
        }

        if let Some(ref mut segments) = *self.segments.write() {
            // Flush mutable segment first
            segments.flush().map_err(|e| anyhow::anyhow!("{e}"))?;
            // Merge all frozen segments
            if let Some(stats) = segments
                .merge_all_frozen()
                .map_err(|e| anyhow::anyhow!("{e}"))?
            {
                return Ok(stats.vectors_merged);
            }
        }
        Ok(0)
    }

    /// Compact the database by removing deleted records and reclaiming space.
    ///
    /// This operation:
    /// 1. Removes all tombstoned (deleted) records from storage
    /// 2. Reassigns slot indices to be contiguous
    /// 3. Rebuilds the HNSW index with new slot assignments
    /// 4. Rebuilds the metadata index
    ///
    /// Returns the number of deleted records that were removed.
    ///
    /// # Persistence
    ///
    /// **Important:** Compaction modifies in-memory state only. You MUST call
    /// [`flush()`](Self::flush) after compact() to persist the compacted state.
    /// Without flush, a crash will recover the pre-compaction state from disk.
    ///
    /// # Example
    /// ```ignore
    /// // After deleting many records
    /// db.delete_batch(&old_ids)?;
    ///
    /// // Reclaim space (in-memory only)
    /// let removed = db.compact()?;
    /// println!("Removed {} deleted records", removed);
    ///
    /// // REQUIRED: Persist the compacted state
    /// db.flush()?;
    /// ```
    ///
    /// # Performance
    /// Compaction rebuilds the HNSW index, which is O(n log n) where n is the
    /// number of live records. Call periodically after bulk deletes, not after
    /// every delete.
    pub fn compact(&self) -> Result<usize> {
        let _lock = self.write_lock.lock();
        self.compact_locked()
    }

    pub(super) fn compact_locked(&self) -> Result<usize> {
        // Count tombstones before compacting
        let removed_count = self.records.deleted_count() as usize;

        if removed_count == 0 {
            return Ok(0);
        }

        // Compact RecordStore - reassigns slots, clears tombstones
        let old_to_new = self.records.compact();

        // Compact multi-vector storage if present
        if let Some(ref mut multivec_storage) = *self.multivec_storage.write() {
            multivec_storage.compact(&old_to_new);
        }

        // Compact sparse index if present
        if let Some(ref mut sparse_index) = *self.sparse_index.write() {
            sparse_index.compact(&old_to_new);
        }

        // GC orphaned edges (safety net after slot reassignment)
        if let Some(ref mut edge_store) = *self.edge_store.write() {
            let live_ids: FxHashSet<String> = self
                .records
                .iter_live()
                .map(|(_, r)| r.id.clone())
                .collect();
            edge_store.gc_orphaned(&live_ids);
        }

        // Rebuild segments with new contiguous slots
        if self.records.is_empty() {
            *self.segments.write() = None;
        } else {
            self.rebuild_index_locked()?;
        }

        // Rebuild metadata index from compacted records
        *self.metadata_index.write() = MetadataIndex::new();
        for (slot, record) in self.records.iter_live() {
            if let Some(ref meta) = record.metadata {
                self.metadata_index.write().index_json(slot, meta);
            }
        }

        Ok(removed_count)
    }

    /// Compute tombstone ratio: deleted slots / total slots.
    ///
    /// Returns 0.0 when no slots exist. Used by `flush()` to decide whether
    /// to auto-compact before persisting.
    #[inline]
    pub(crate) fn tombstone_ratio(&self) -> f32 {
        let total = self.records.slot_count();
        if total == 0 {
            return 0.0;
        }
        self.records.deleted_count() as f32 / total as f32
    }

    /// Set the auto-compact threshold.
    ///
    /// `flush()` triggers compaction when the tombstone ratio exceeds this value.
    /// Default: 0.25. Set to 1.0 to disable auto-compact.
    pub fn set_auto_compact_threshold(&self, threshold: f32) {
        self.auto_compact_threshold.store(
            threshold.clamp(0.0, 1.0).to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
