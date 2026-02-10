use anyhow::Result;
use serde_json::Value as JsonValue;

use super::helpers;
use super::VectorStore;
use super::{HNSWParams, MetadataFilter, SegmentConfig, SegmentManager, Vector};

impl VectorStore {
    /// Insert vector and return its slot ID (used in tests)
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, vector: Vector) -> Result<usize> {
        // Generate a unique ID for unnamed vectors
        let slot = self.records.slot_count();
        let id = format!("__auto_{slot}");

        self.set(id, vector, helpers::default_metadata())
    }

    /// Insert vector with string ID and metadata
    ///
    /// This is the primary method for inserting vectors with metadata support.
    /// Returns error if ID already exists (use set for insert-or-update semantics).
    #[allow(dead_code)]
    pub(crate) fn insert_with_metadata(
        &mut self,
        id: String,
        vector: Vector,
        metadata: JsonValue,
    ) -> Result<usize> {
        if self.records.get_slot(&id).is_some() {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        self.set(id, vector, metadata)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    ///
    /// This is the recommended method for most use cases.
    ///
    /// # Durability
    ///
    /// Individual writes are buffered in the WAL but NOT synced to disk immediately.
    /// For guaranteed durability, call [`flush()`](Self::flush) after critical writes.
    /// Batch operations ([`set_batch`](Self::set_batch)) sync the WAL at batch end.
    ///
    /// Without explicit flush:
    /// - Data is recoverable after normal shutdown
    /// - Data may be lost on crash/power failure between set() and next flush/batch
    pub fn set(&mut self, id: String, vector: Vector, metadata: JsonValue) -> Result<usize> {
        // Initialize segments if needed
        if self.segments.is_none() {
            let dimensions = self.resolve_dimensions(vector.dim())?;
            self.records.set_dimensions(dimensions as u32);

            // Create segment manager with initial config
            let config = SegmentConfig::new(dimensions)
                .with_params(HNSWParams {
                    m: self.hnsw_m,
                    ef_construction: self.hnsw_ef_construction,
                    ..Default::default()
                })
                .with_distance(self.distance_metric.into())
                .with_quantization(self.pending_quantization);

            self.segments = Some(
                SegmentManager::new(config)
                    .map_err(|e| anyhow::anyhow!("Failed to create segment manager: {e}"))?,
            );
        } else if vector.dim() != self.dimensions() {
            anyhow::bail!(
                "Vector dimension mismatch: store expects {}, got {}",
                self.dimensions(),
                vector.dim()
            );
        }

        // Check if this is an update
        let old_slot = self.records.get_slot(&id);

        // Upsert into RecordStore - creates new slot (both for insert and update)
        // RecordStore marks old slot deleted internally to maintain slot == HNSW node ID
        let slot = self
            .records
            .upsert(id.clone(), vector.data.clone(), Some(metadata.clone()))?
            as usize;

        // Insert into segments
        if let Some(ref mut segments) = self.segments {
            // Note: mark_deleted not needed - RecordStore filtering handles deleted nodes
            segments
                .insert_with_slot(&vector.data, slot as u32)
                .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
        }

        // Update metadata index
        if let Some(old) = old_slot {
            self.metadata_index.remove(old);
        }
        self.metadata_index.index_json(slot as u32, &metadata);

        // WAL for crash durability
        if let Some(ref mut storage) = self.storage {
            let metadata_bytes = serde_json::to_vec(&metadata)?;
            storage.wal_append_insert(&id, &vector.data, Some(&metadata_bytes))?;
            storage.wal_sync()?;
        }

        Ok(slot)
    }

    /// Batch set vectors (insert or update multiple vectors at once)
    ///
    /// This is the recommended method for bulk operations.
    /// Uses parallel HNSW construction for new indexes.
    ///
    /// # Ordering Guarantee
    ///
    /// This method processes updates before inserts to ensure slot ordering is predictable.
    /// **WARNING:** `set_multi_batch` in `multivec_ops.rs` depends on this ordering to
    /// correctly align tokens with FDEs. Do not change without updating that code.
    pub fn set_batch(&mut self, batch: Vec<(String, Vector, JsonValue)>) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        // Separate batch into updates and inserts (updates processed first - see docstring)
        let mut updates: Vec<(u32, String, Vector, JsonValue)> = Vec::new();
        let mut inserts: Vec<(String, Vector, JsonValue)> = Vec::new();

        for (id, vector, metadata) in batch {
            if let Some(slot) = self.records.get_slot(&id) {
                updates.push((slot, id, vector, metadata));
            } else {
                inserts.push((id, vector, metadata));
            }
        }

        let mut result_indices = Vec::with_capacity(updates.len() + inserts.len());

        // Process updates individually
        for (old_slot, id, vector, metadata) in updates {
            // Update RecordStore - creates new slot, marks old as deleted
            let new_slot =
                self.records
                    .upsert(id.clone(), vector.data.clone(), Some(metadata.clone()))?;

            // Insert into segments
            if let Some(ref mut segments) = self.segments {
                segments
                    .insert_with_slot(&vector.data, new_slot)
                    .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
            }

            // Update metadata index (remove old, add new)
            self.metadata_index.remove(old_slot);
            self.metadata_index.index_json(new_slot, &metadata);

            // WAL for crash durability
            if let Some(ref mut storage) = self.storage {
                let metadata_bytes = serde_json::to_vec(&metadata)?;
                storage.wal_append_insert(&id, &vector.data, Some(&metadata_bytes))?;
            }

            result_indices.push(new_slot as usize);
        }

        // Process inserts with batch optimization
        if !inserts.is_empty() {
            let vectors_data: Vec<Vec<f32>> =
                inserts.iter().map(|(_, v, _)| v.data.clone()).collect();

            // Check if this is a new index (no existing segments)
            let is_new_index = self.segments.is_none();

            if is_new_index {
                let dimensions = self.resolve_dimensions(inserts[0].1.dim())?;
                self.records.set_dimensions(dimensions as u32);

                // Insert into RecordStore first to get slots
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in &inserts {
                    let slot = self.records.upsert(
                        id.clone(),
                        vector.data.clone(),
                        Some(metadata.clone()),
                    )?;
                    slots.push(slot);
                    self.metadata_index.index_json(slot, metadata);
                }

                // Build segment config
                let config = SegmentConfig::new(dimensions)
                    .with_params(HNSWParams {
                        m: self.hnsw_m,
                        ef_construction: self.hnsw_ef_construction,
                        ..Default::default()
                    })
                    .with_distance(self.distance_metric.into())
                    .with_quantization(self.pending_quantization);

                // Use parallel build with slot mapping
                self.segments = Some(
                    SegmentManager::build_parallel_with_slots(config, vectors_data, &slots)
                        .map_err(|e| anyhow::anyhow!("Segment parallel build failed: {e}"))?,
                );

                // Handle quantization mode persistence
                if self.pending_quantization {
                    if let Some(ref mut storage) = self.storage {
                        storage.put_quantization_mode(helpers::quantization_to_id(true))?;
                    }
                }

                // WAL for crash durability
                if let Some(ref mut storage) = self.storage {
                    for (id, vector, metadata) in &inserts {
                        let metadata_bytes = serde_json::to_vec(metadata)?;
                        storage.wal_append_insert(id, &vector.data, Some(&metadata_bytes))?;
                    }
                }

                result_indices.extend(slots.iter().map(|&s| s as usize));
            } else {
                // Existing index - validate dimensions and insert one by one
                let expected_dims = self.dimensions();
                for (i, (_, vector, _)) in inserts.iter().enumerate() {
                    if vector.dim() != expected_dims {
                        anyhow::bail!(
                            "Vector {} dimension mismatch: expected {}, got {}",
                            i,
                            expected_dims,
                            vector.dim()
                        );
                    }
                }

                // Insert into RecordStore and index
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in &inserts {
                    let slot = self.records.upsert(
                        id.clone(),
                        vector.data.clone(),
                        Some(metadata.clone()),
                    )?;
                    slots.push(slot);
                    self.metadata_index.index_json(slot, metadata);

                    // Insert into segments
                    if let Some(ref mut segments) = self.segments {
                        segments
                            .insert_with_slot(&vector.data, slot)
                            .map_err(|e| anyhow::anyhow!("Segment insert failed: {e}"))?;
                    }
                }

                // WAL for crash durability
                if let Some(ref mut storage) = self.storage {
                    for (id, vector, metadata) in &inserts {
                        let metadata_bytes = serde_json::to_vec(metadata)?;
                        storage.wal_append_insert(id, &vector.data, Some(&metadata_bytes))?;
                    }
                }

                result_indices.extend(slots.iter().map(|&s| s as usize));
            }
        }

        // Sync WAL once at end of batch for durability
        if let Some(ref mut storage) = self.storage {
            storage.wal_sync()?;
        }

        Ok(result_indices)
    }

    /// Update existing vector by string ID
    ///
    /// When a new vector is provided, this delegates to `set()` which performs a
    /// full upsert: allocates a new slot, inserts into HNSW, and marks the old
    /// slot as deleted. This ensures the HNSW index reflects the new position.
    ///
    /// Metadata-only updates are done in-place (no re-indexing needed).
    pub fn update(
        &mut self,
        id: &str,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        let slot = self
            .records
            .get_slot(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        if !self.records.is_live(slot) {
            anyhow::bail!("Vector with ID '{id}' has been deleted");
        }

        if let Some(new_vector) = vector {
            // Delegate to set() which handles HNSW re-indexing via upsert
            let merged_metadata = match metadata {
                Some(m) => m,
                None => self
                    .records
                    .get_by_slot(slot)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(|| serde_json::json!({})),
            };
            self.set(id.to_string(), new_vector, merged_metadata)?;
        } else if let Some(ref new_metadata) = metadata {
            // Metadata-only update: no HNSW re-indexing needed
            self.metadata_index.remove(slot);
            self.metadata_index.index_json(slot, new_metadata);
            self.records.update_metadata(slot, new_metadata.clone())?;

            if let Some(ref mut storage) = self.storage {
                storage.put_metadata(slot as usize, new_metadata)?;
            }
        }

        Ok(())
    }

    /// Delete vector by string ID (lazy delete)
    ///
    /// This method:
    /// 1. Marks the vector as deleted in bitmap (O(1) soft delete)
    /// 2. Marks node as deleted in HNSW (filtered during search)
    /// 3. Removes from text index if present
    /// 4. Persists to WAL
    ///
    /// Deleted vectors are filtered during search. Call `compact()` to reclaim space.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        // Delete from RecordStore (single source of truth)
        let slot = self
            .records
            .delete(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.metadata_index.remove(slot);

        // Use OmenFile::delete for WAL-backed persistence
        if let Some(ref mut storage) = self.storage {
            storage.delete(id)?;
        }

        if let Some(ref mut text_index) = self.text_index {
            text_index.delete_document(id)?;
        }

        Ok(())
    }

    /// Delete multiple vectors by string IDs (lazy delete)
    ///
    /// Marks vectors as deleted in bitmap. Deleted vectors are filtered during search.
    /// Call `compact()` to reclaim space after bulk deletes.
    pub fn delete_batch(&mut self, ids: &[String]) -> Result<usize> {
        // Delete from RecordStore and collect slots
        let mut slots: Vec<u32> = Vec::with_capacity(ids.len());
        let mut valid_ids: Vec<String> = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(slot) = self.records.delete(id) {
                self.metadata_index.remove(slot);
                slots.push(slot);
                valid_ids.push(id.clone());
            }
        }

        // Persist deletions
        for id in &valid_ids {
            if let Some(ref mut storage) = self.storage {
                storage.delete(id)?;
            }
            if let Some(ref mut text_index) = self.text_index {
                if let Err(e) = text_index.delete_document(id) {
                    tracing::warn!(id = %id, error = ?e, "Failed to delete from text index");
                }
            }
        }

        Ok(valid_ids.len())
    }

    /// Delete vectors matching a metadata filter
    ///
    /// Evaluates the filter against all vectors and deletes those that match.
    /// This is more efficient than manually iterating and calling delete_batch.
    ///
    /// # Arguments
    /// * `filter` - MongoDB-style metadata filter
    ///
    /// # Returns
    /// Number of vectors deleted
    pub fn delete_by_filter(&mut self, filter: &MetadataFilter) -> Result<usize> {
        // Find matching IDs
        let ids_to_delete: Vec<String> = self
            .records
            .iter_live()
            .filter_map(|(_, record)| {
                let metadata = record.metadata.as_ref()?;
                if filter.matches(metadata) {
                    Some(record.id.clone())
                } else {
                    None
                }
            })
            .collect();

        if ids_to_delete.is_empty() {
            return Ok(0);
        }

        self.delete_batch(&ids_to_delete)
    }

    /// Count vectors matching a metadata filter
    ///
    /// Evaluates the filter against all vectors and returns the count of matches.
    /// More efficient than iterating and counting manually.
    ///
    /// # Arguments
    /// * `filter` - MongoDB-style metadata filter
    ///
    /// # Returns
    /// Number of vectors matching the filter
    #[must_use]
    pub fn count_by_filter(&self, filter: &MetadataFilter) -> usize {
        self.records
            .iter_live()
            .filter(|(_, record)| {
                record
                    .metadata
                    .as_ref()
                    .is_some_and(|metadata| filter.matches(metadata))
            })
            .count()
    }

    /// Get vector by string ID
    ///
    /// Returns owned data since vectors may be loaded from disk for quantized stores.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        let record = self.records.get(id)?;
        let metadata = record
            .metadata
            .clone()
            .unwrap_or_else(helpers::default_metadata);
        Some((Vector::new(record.vector.clone()), metadata))
    }

    /// Get multiple vectors by string IDs
    ///
    /// Returns a vector of results in the same order as input IDs.
    /// Missing/deleted IDs return None in their position.
    #[must_use]
    pub fn get_batch(&self, ids: &[impl AsRef<str>]) -> Vec<Option<(Vector, JsonValue)>> {
        ids.iter().map(|id| self.get(id.as_ref())).collect()
    }

    /// Get metadata by string ID (without loading vector data)
    #[must_use]
    pub fn get_metadata_by_id(&self, id: &str) -> Option<&JsonValue> {
        self.records.get(id).and_then(|r| r.metadata.as_ref())
    }
}
