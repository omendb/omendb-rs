use anyhow::Result;
use serde_json::Value as JsonValue;

use super::VectorStore;
use super::helpers;
use super::{MetadataFilter, Vector};

impl VectorStore {
    /// Insert vector and return its slot ID (used in tests)
    #[allow(dead_code)]
    pub(crate) fn insert(&self, vector: Vector) -> Result<usize> {
        // Serialized via set()
        let slot = self.records.slot_count();
        let id = format!("__auto_{slot}");

        self.set(&id, vector, helpers::default_metadata())
    }

    /// Insert vector with string ID and metadata
    #[allow(dead_code)]
    pub(crate) fn insert_with_metadata(
        &self,
        id: &str,
        vector: Vector,
        metadata: JsonValue,
    ) -> Result<usize> {
        if self.records.get_slot(id).is_some() {
            anyhow::bail!("Vector with ID '{id}' already exists. Use set() to update.");
        }

        self.set(id, vector, metadata)
    }

    /// Upsert vector (insert or update) with string ID and metadata
    pub fn set(&self, id: &str, vector: Vector, metadata: JsonValue) -> Result<usize> {
        let _lock = self.write_lock.read();

        self.ensure_segments_initialized(vector.dim())?;

        // Check if this is an update
        let old_slot = self.records.get_slot(id);

        // Upsert into RecordStore
        let slot = self
            .records
            .set(id.to_string(), vector.data, Some(metadata.clone()))? as usize;

        debug_assert_eq!(
            self.records.get_slot(id),
            Some(slot as u32),
            "Slot consistency violation: id '{id}' does not map to returned slot {slot}",
        );
        debug_assert!(
            (slot as u32) < self.records.slot_count(),
            "RecordStore does not contain the returned slot {slot}",
        );

        self.records.with_vector_by_slot(slot as u32, |vector| {
            self.with_engine_mut(|engine| {
                if let Some(engine) = engine.as_mut()
                    && let Some(vector) = vector
                {
                    engine
                        .insert_with_slot(vector, slot as u32)
                        .map_err(|e| anyhow::anyhow!("Engine insert failed: {e}"))?;
                }
                Ok(())
            })
        })?;

        // Update metadata index and migrate sparse entry to new slot
        if let Some(old) = old_slot {
            self.metadata_index.write().remove(old);
            if let Some(ref mut sparse_index) = *self.sparse_index.write() {
                sparse_index.remap_slot(old, slot as u32);
            }
        }
        self.metadata_index
            .write()
            .index_json(slot as u32, &metadata);

        // WAL for crash durability
        let needs_checkpoint = self.records.with_vector_by_slot(slot as u32, |vector| {
            if let Some(ref storage) = self.storage {
                let mut storage = storage.write();
                let vector = vector.ok_or_else(|| anyhow::anyhow!("Record vector missing"))?;
                storage.log_insert(id, vector, &metadata)?;
                storage.sync()?;
                Ok::<bool, anyhow::Error>(
                    storage.wal_len() >= super::WAL_AUTO_CHECKPOINT_ENTRIES as usize,
                )
            } else {
                Ok(false)
            }
        })?;

        if needs_checkpoint {
            self.checkpoint_wal_locked()?;
        }

        self.check_memory_pressure();

        Ok(slot)
    }

    /// Batch set vectors
    pub fn set_batch<S: Into<String>>(
        &self,
        batch: Vec<(S, Vector, JsonValue)>,
    ) -> Result<Vec<usize>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let _lock = self.write_lock.read();

        // Convert IDs to String up front
        let batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, metadata)| (id.into(), vector, metadata))
            .collect();

        // Separate batch into updates and inserts
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

        // Process updates individually, but keep batch-wide locks stable.
        if !updates.is_empty() {
            let mut metadata_index = self.metadata_index.write();
            let mut sparse_index = self.sparse_index.write();
            self.with_engine_mut(|engine| {
                for (old_slot, id, vector, metadata) in updates {
                    let new_slot =
                        self.records
                            .set(id.clone(), vector.data, Some(metadata.clone()))?;

                    self.records.with_vector_by_slot(new_slot, |vector| {
                        if let Some(engine) = engine.as_mut()
                            && let Some(vector) = vector
                        {
                            engine
                                .insert_with_slot(vector, new_slot)
                                .map_err(|e| anyhow::anyhow!("Engine insert failed: {e}"))?;
                        }
                        Ok::<_, anyhow::Error>(())
                    })?;

                    metadata_index.remove(old_slot);
                    if let Some(sparse_index) = sparse_index.as_mut() {
                        sparse_index.remap_slot(old_slot, new_slot);
                    }
                    metadata_index.index_json(new_slot, &metadata);

                    result_indices.push(new_slot as usize);
                }
                Ok(())
            })?;
        }

        // Process inserts with batch optimization
        if !inserts.is_empty() {
            if self.has_engine() {
                let expected_dims = self.dimensions();
                for (_, vector, _) in &inserts {
                    if vector.dim() != expected_dims {
                        anyhow::bail!("Vector dimension mismatch");
                    }
                }

                let mut metadata_index = self.metadata_index.write();
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in inserts {
                    let slot = self.records.set(id, vector.data, Some(metadata.clone()))?;
                    slots.push(slot);
                    metadata_index.index_json(slot, &metadata);
                }

                self.records.with_vectors_by_slots(&slots, |vectors| {
                    self.with_engine_mut(|engine| {
                        if let Some(engine) = engine.as_mut() {
                            engine.insert_batch_parallel_from_refs(vectors, &slots)?;
                        }
                        Ok(())
                    })
                })?;

                result_indices.extend(slots.iter().map(|&s| s as usize));
            } else {
                let dimensions = self.resolve_dimensions(inserts[0].1.dim())?;
                self.records.set_dimensions(dimensions as u32);

                let mut metadata_index = self.metadata_index.write();
                let mut slots = Vec::with_capacity(inserts.len());
                for (id, vector, metadata) in inserts {
                    let slot = self.records.set(id, vector.data, Some(metadata.clone()))?;
                    slots.push(slot);
                    metadata_index.index_json(slot, &metadata);
                }

                self.records.with_vectors_by_slots(&slots, |vectors| {
                    self.build_and_publish_engine_from_refs(dimensions, vectors, &slots)
                })?;

                if self.is_quantized()
                    && let Some(ref storage) = self.storage
                {
                    storage
                        .write()
                        .put_config("quantization", helpers::quantization_to_id(true))?;
                }

                result_indices.extend(slots.iter().map(|&s| s as usize));
            }
        }

        let needs_checkpoint = self
            .storage
            .as_ref()
            .is_some_and(|s| s.read().wal_len() >= super::WAL_AUTO_CHECKPOINT_ENTRIES as usize);
        if needs_checkpoint {
            self.checkpoint_wal_locked()?;
        }

        Ok(result_indices)
    }

    /// Update existing vector
    pub fn update(
        &self,
        id: &str,
        vector: Option<Vector>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        let _lock = self.write_lock.read();

        let slot = self
            .records
            .get_slot(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        if !self.records.is_live(slot) {
            anyhow::bail!("Vector with ID '{id}' has been deleted");
        }

        if let Some(new_vector) = vector {
            let merged_metadata = match metadata {
                Some(m) => m,
                None => self
                    .records
                    .get_by_slot(slot)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or_else(|| serde_json::json!({})),
            };
            drop(_lock); // Release to allow set() to re-acquire
            self.set(id, new_vector, merged_metadata)?;
        } else if let Some(ref new_metadata) = metadata {
            let existing_vector = self.records.get_vector(slot);

            let needs_checkpoint = if let Some(ref storage) = self.storage {
                let mut storage = storage.write();
                if let Some(vec_data) = &existing_vector {
                    storage.log_insert(id, vec_data.as_slice(), new_metadata)?;
                    storage.sync()?;
                    storage.wal_len() >= super::WAL_AUTO_CHECKPOINT_ENTRIES as usize
                } else {
                    false
                }
            } else {
                false
            };

            self.metadata_index.write().remove(slot);
            self.metadata_index.write().index_json(slot, new_metadata);
            self.records.update_metadata(slot, new_metadata.clone())?;

            if needs_checkpoint {
                self.checkpoint_wal_locked()?;
            }
        }

        Ok(())
    }

    /// Delete vector by string ID
    pub fn delete(&self, id: &str) -> Result<()> {
        let _lock = self.write_lock.read();

        let slot = self
            .records
            .delete(id)
            .ok_or_else(|| anyhow::anyhow!("Vector with ID '{id}' not found"))?;

        self.metadata_index.write().remove(slot);

        if let Some(ref mut sparse_index) = *self.sparse_index.write() {
            sparse_index.remove(slot);
        }

        let edge_deletes: Vec<(String, String, String)> = self
            .edge_store
            .read()
            .as_ref()
            .map(|es| es.edges_involving(id))
            .unwrap_or_default();

        if let Some(ref mut edge_store) = *self.edge_store.write() {
            edge_store.remove_all_for(id);
        }
        // WAL for crash durability
        let needs_checkpoint = if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            for (from_id, to_id, edge_type) in &edge_deletes {
                storage.log_delete_edge(from_id, to_id, edge_type)?;
            }
            storage.log_delete(id)?;
            storage.sync()?;
            storage.wal_len() >= super::WAL_AUTO_CHECKPOINT_ENTRIES as usize
        } else {
            false
        };

        if let Some(ref mut text_index) = *self.text_index.write() {
            text_index.delete_document(id)?;
        }

        if needs_checkpoint {
            self.checkpoint_wal_locked()?;
        }

        Ok(())
    }

    /// Delete multiple vectors
    pub fn delete_batch(&self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let _lock = self.write_lock.read();

        let mut valid_ids: Vec<String> = Vec::with_capacity(ids.len());
        let mut edge_deletes: Vec<(String, String, String)> = Vec::new();

        for id in ids {
            let id = id.as_ref();
            if let Some(slot) = self.records.delete(id) {
                self.metadata_index.write().remove(slot);
                if let Some(ref mut sparse_index) = *self.sparse_index.write() {
                    sparse_index.remove(slot);
                }
                if let Some(ref es) = *self.edge_store.read() {
                    edge_deletes.extend(es.edges_involving(id));
                }
                if let Some(ref mut edge_store) = *self.edge_store.write() {
                    edge_store.remove_all_for(id);
                }
                valid_ids.push(id.to_string());
            }
        }

        if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            for (from_id, to_id, edge_type) in &edge_deletes {
                storage.log_delete_edge(from_id, to_id, edge_type)?;
            }
            for id in &valid_ids {
                storage.log_delete(id)?;
            }
            if !valid_ids.is_empty() {
                storage.sync()?;
            }
        }
        for id in &valid_ids {
            if let Some(ref mut text_index) = *self.text_index.write() {
                let _ = text_index.delete_document(id);
            }
        }

        Ok(valid_ids.len())
    }

    /// Delete vectors matching a metadata filter
    pub fn delete_by_filter(&self, filter: &MetadataFilter) -> Result<usize> {
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

    /// Get vector by string ID
    #[must_use]
    pub fn get(&self, id: &str) -> Option<(Vector, JsonValue)> {
        let record = self.records.get(id)?;
        let metadata = record
            .metadata
            .clone()
            .unwrap_or_else(helpers::default_metadata);
        let vector = record.vector?;
        Some((Vector::new(vector.to_vec()), metadata))
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
    pub fn get_metadata_by_id(&self, id: &str) -> Option<JsonValue> {
        self.records.get(id).and_then(|r| r.metadata.clone())
    }
}
