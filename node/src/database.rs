use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use omendb_lib::vector::{Vector, VectorStore, VectorStoreOptions};
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

use crate::conversions::{convert_error, parse_text_search_config};
use crate::filters::parse_filter;
use crate::types::{CollectionSchemaResult, GetResult, InfoResult, SetItem, StatsResult};

pub(crate) struct VectorDatabaseInner {
    pub(crate) store: VectorStore,
}

/// Type alias for embedding function: (texts: string[]) => Float32Array[]
/// CalleeHandled = false so the JS function is called directly with (value), not (err, value)
pub(crate) type EmbeddingFn =
    ThreadsafeFunction<Vec<String>, Vec<Float32Array>, Vec<String>, Status, false>;

impl VectorDatabase {
    pub(crate) fn live_is_multi_vector(&self) -> bool {
        let inner = self.inner.read();
        inner.store.is_multi_vector()
    }

    pub(crate) fn live_dimensions(&self) -> u32 {
        let inner = self.inner.read();
        if inner.store.is_multi_vector() {
            inner
                .store
                .token_dimension()
                .unwrap_or(inner.store.dimensions()) as u32
        } else {
            inner.store.dimensions() as u32
        }
    }
}

#[napi]
pub struct VectorDatabase {
    pub(crate) inner: Arc<RwLock<VectorDatabaseInner>>,
    pub(crate) path: String,
    pub(crate) is_persistent: bool,
    pub(crate) embedding_fn: Option<Arc<EmbeddingFn>>,
    pub(crate) collections_cache: RwLock<HashMap<String, Arc<RwLock<VectorDatabaseInner>>>>,
}

#[napi]
impl VectorDatabase {
    /// Insert or update vectors.
    ///
    /// Works for both single-vector and multi-vector stores:
    /// - Single-vector: items have `vector` field
    /// - Multi-vector: items have `vectors` field (array of vectors)
    ///
    /// When any item includes a `text` field, text search is automatically enabled.
    /// This allows immediate use of searchHybrid() without calling enableTextSearch().
    ///
    /// @param items - Array of {id, vector, metadata?, text?} or {id, vectors, metadata?}
    /// @returns Number of vectors inserted/updated
    ///
    /// @note Batch inserts (multiple items) skip the WAL for performance. Data is not
    /// durable until flush() is called.
    #[napi]
    pub async fn set(&self, items: Vec<SetItem>) -> Result<u32> {
        let has_documents = items.iter().any(|item| item.document.is_some());
        let items = if has_documents {
            let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "No embedding function configured. Pass embeddingFn to open() or provide vectors directly.",
                )
            })?;

            let doc_indices: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.document.is_some())
                .map(|(i, _)| i)
                .collect();

            let docs: Vec<String> = doc_indices
                .iter()
                .map(|&i| items[i].document.clone().unwrap())
                .collect();

            for &i in &doc_indices {
                if items[i].vector.is_some() {
                    return Err(Error::from_reason(format!(
                        "Item '{}': cannot have both 'vector' and 'document' - use one or the other",
                        items[i].id
                    )));
                }
            }

            let result: Vec<Float32Array> = emb_fn.call_async(docs).await?;

            if result.len() != doc_indices.len() {
                return Err(Error::from_reason(format!(
                    "embeddingFn returned {} vectors for {} documents",
                    result.len(),
                    doc_indices.len()
                )));
            }

            let mut items = items;
            for (idx, embedded) in doc_indices.into_iter().zip(result) {
                items[idx].vector = Some(embedded);
                items[idx].document = None;
            }
            items
        } else {
            items
        };

        if self.live_is_multi_vector() {
            let mut inner = self.inner.write();
            let count = items.len();

            for item in items {
                let vectors = item.vectors.ok_or_else(|| {
                    Error::new(
                        Status::InvalidArg,
                        format!(
                            "Multi-vector store requires 'vectors' field for item '{}'. Got 'vector' field - use an array of vectors instead.",
                            item.id
                        ),
                    )
                })?;

                if vectors.is_empty() {
                    return Err(Error::new(
                        Status::InvalidArg,
                        format!("vectors for '{}' must not be empty", item.id),
                    ));
                }

                let tokens: Vec<Vec<f32>> = vectors.into_iter().map(|v| v.to_vec()).collect();
                let metadata = item.metadata.unwrap_or(serde_json::json!({}));

                inner
                    .store
                    .store(&item.id, tokens, metadata)
                    .map_err(convert_error)?;
            }

            Ok(count as u32)
        } else {
            let has_text = items.iter().any(|item| item.text.is_some());

            let mut inner = self.inner.write();

            if has_text && !inner.store.has_text_search() {
                inner.store.enable_text_search().map_err(convert_error)?;
            }

            if has_text {
                let mut count = 0u32;
                for item in items {
                    let vector = item.vector.ok_or_else(|| {
                        Error::new(
                            Status::InvalidArg,
                            format!(
                                "Single-vector store requires 'vector' field for item '{}'. Got 'vectors' field - use multiVector: true when opening the database.",
                                item.id
                            ),
                        )
                    })?;

                    let mut metadata = item.metadata.unwrap_or(serde_json::json!({}));

                    if let Some(ref text) = item.text {
                        if let Some(obj) = metadata.as_object() {
                            if obj.contains_key("text") {
                                return Err(Error::from_reason(format!(
                                    "Item '{}': cannot have both 'text' field and 'metadata.text' - use one or the other",
                                    item.id
                                )));
                            }
                        }
                        if let Some(obj) = metadata.as_object_mut() {
                            obj.insert("text".to_string(), serde_json::json!(text));
                        }
                        inner
                            .store
                            .set_with_text(&item.id, Vector::new(vector.to_vec()), text, metadata)
                            .map_err(convert_error)?;
                    } else {
                        inner
                            .store
                            .set(&item.id, Vector::new(vector.to_vec()), metadata)
                            .map_err(convert_error)?;
                    }
                    count += 1;
                }
                Ok(count)
            } else {
                let batch: Vec<(String, Vector, JsonValue)> = items
                    .into_iter()
                    .map(|item| {
                        let vector = item.vector.ok_or_else(|| {
                            Error::new(
                                Status::InvalidArg,
                                format!(
                                    "Single-vector store requires 'vector' field for item '{}'. Got 'vectors' field - use multiVector: true when opening the database.",
                                    item.id
                                ),
                            )
                        })?;

                        let metadata = item.metadata.unwrap_or(serde_json::json!({}));
                        Ok((item.id, Vector::new(vector.to_vec()), metadata))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let result = inner.store.set_batch(batch).map_err(convert_error)?;
                Ok(result.len() as u32)
            }
        }
    }

    /// Enable text search with optional typed text-search configuration.
    ///
    /// @param config - Optional text search config with buffer and tokenizer
    #[napi(js_name = "enableTextSearch")]
    pub fn enable_text_search(
        &self,
        #[napi(
            ts_arg_type = "{ bufferMb?: number; writerBufferMb?: number; tokenizer?: 'default' | 'code' | 'raw' } | null"
        )]
        config: Option<JsonValue>,
    ) -> Result<()> {
        let mut inner = self.inner.write();
        let config = config
            .as_ref()
            .map(parse_text_search_config)
            .transpose()?
            .flatten();

        inner
            .store
            .enable_text_search_with_config(config)
            .map_err(convert_error)
    }

    /// Get a vector by ID.
    #[napi]
    pub fn get(&self, id: String) -> Option<GetResult> {
        let inner = self.inner.read();

        inner.store.get(&id).map(|(vec, metadata)| GetResult {
            id,
            vector: Float32Array::new(vec.data),
            metadata,
        })
    }

    /// Delete vectors by ID.
    ///
    /// Accepts either a single ID string or an array of IDs.
    ///
    /// @param ids - Single ID string or array of IDs to delete
    /// @returns Number of vectors deleted
    ///
    /// @example
    /// ```javascript
    /// // Delete single
    /// db.delete("doc1");
    ///
    /// // Delete multiple
    /// db.delete(["doc1", "doc2", "doc3"]);
    /// ```
    #[napi]
    pub fn delete(&self, ids: Either<String, Vec<String>>) -> Result<u32> {
        let id_vec = match ids {
            Either::A(single) => vec![single],
            Either::B(multiple) => multiple,
        };
        let inner = self.inner.write();
        let result = inner.store.delete_batch(&id_vec).map_err(convert_error)?;
        Ok(result as u32)
    }

    /// Delete vectors matching a metadata filter.
    ///
    /// Evaluates the filter against all vectors and deletes those that match.
    /// Uses the same MongoDB-style filter syntax as search().
    ///
    /// @param filter - MongoDB-style metadata filter
    /// @returns Number of vectors deleted
    ///
    /// @example
    /// ```javascript
    /// // Delete by equality
    /// db.deleteByFilter({ status: "archived" });
    ///
    /// // Delete with comparison
    /// db.deleteByFilter({ score: { $lt: 0.5 } });
    ///
    /// // Complex filter
    /// db.deleteByFilter({ $and: [{ type: "draft" }, { age: { $gt: 30 } }] });
    /// ```
    #[napi]
    pub fn delete_by_filter(
        &self,
        #[napi(ts_arg_type = "Record<string, unknown>")] filter: JsonValue,
    ) -> Result<u32> {
        let parsed_filter = parse_filter(&filter)?;

        let inner = self.inner.write();
        let result = inner
            .store
            .delete_by_filter(&parsed_filter)
            .map_err(convert_error)?;

        Ok(result as u32)
    }

    /// Count vectors, optionally filtered by metadata.
    ///
    /// Without a filter, returns total count (same as db.length).
    /// With a filter, returns count of vectors matching the filter.
    ///
    /// @param filter - Optional MongoDB-style metadata filter
    /// @returns Number of vectors (matching filter if provided)
    ///
    /// @example
    /// ```javascript
    /// // Total count
    /// const total = db.count();
    ///
    /// // Filtered count
    /// const active = db.count({ status: "active" });
    ///
    /// // With comparison operators
    /// const highScore = db.count({ score: { $gte: 0.8 } });
    /// ```
    #[napi(js_name = "count")]
    pub fn count_method(
        &self,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] filter: Option<JsonValue>,
    ) -> Result<u32> {
        let inner = self.inner.read();
        match filter {
            Some(f) => {
                let parsed_filter = parse_filter(&f)?;
                Ok(inner.store.count_by_filter(&parsed_filter) as u32)
            }
            None => Ok(inner.store.len() as u32),
        }
    }

    /// Update a vector's data, metadata, and/or text.
    ///
    /// @param id - Vector ID to update
    /// @param options - Update options: {vector?, metadata?, text?}
    ///
    /// @example
    /// ```javascript
    /// // Update vector only
    /// db.update("doc1", { vector: [1, 0, 0, 0] });
    ///
    /// // Update metadata only
    /// db.update("doc1", { metadata: { status: "active" } });
    ///
    /// // Update text (re-indexed for BM25 search)
    /// db.update("doc1", { text: "Updated content for search" });
    ///
    /// // Update multiple fields
    /// db.update("doc1", { vector: [...], metadata: {...}, text: "..." });
    /// ```
    #[napi]
    pub fn update(
        &self,
        id: String,
        #[napi(
            ts_arg_type = "{ vector?: number[] | Float32Array; metadata?: Record<string, unknown>; text?: string }"
        )]
        options: JsonValue,
    ) -> Result<()> {
        let vector_val = options.get("vector");
        let metadata_val = options.get("metadata").cloned();
        let text_val = options
            .get("text")
            .and_then(|v| v.as_str())
            .map(String::from);

        if vector_val.is_none() && metadata_val.is_none() && text_val.is_none() {
            return Err(Error::from_reason(
                "update() requires at least one of vector, metadata, or text",
            ));
        }

        let vector = if let Some(v) = vector_val {
            if let Some(arr) = v.as_array() {
                let floats: Vec<f32> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        x.as_f64()
                            .ok_or_else(|| {
                                Error::from_reason(format!(
                                    "vector[{}] must be a number, got {:?}",
                                    i, x
                                ))
                            })
                            .map(|n| n as f32)
                    })
                    .collect::<Result<Vec<f32>>>()?;
                Some(Vector::new(floats))
            } else {
                return Err(Error::from_reason("vector must be an array of numbers"));
            }
        } else {
            None
        };

        let mut inner = self.inner.write();

        if let Some(ref new_text) = text_val {
            let (existing_vec, existing_meta) = inner
                .store
                .get(&id)
                .ok_or_else(|| Error::from_reason(format!("Vector with ID '{}' not found", id)))?;

            let final_vec = vector.unwrap_or(existing_vec);

            let mut final_meta = metadata_val.unwrap_or(existing_meta);
            if let Some(obj) = final_meta.as_object_mut() {
                obj.insert("text".to_string(), serde_json::json!(new_text));
            } else {
                final_meta = serde_json::json!({"text": new_text});
            }

            if inner.store.has_text_search() {
                inner
                    .store
                    .set_with_text(&id, final_vec, new_text, final_meta)
                    .map_err(convert_error)?;
            } else {
                inner
                    .store
                    .set(&id, final_vec, final_meta)
                    .map_err(convert_error)?;
            }
        } else {
            inner
                .store
                .update(&id, vector, metadata_val)
                .map_err(convert_error)?;
        }

        Ok(())
    }

    /// Get number of vectors in database.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        let inner = self.inner.read();
        inner.store.len() as u32
    }

    /// Get vector dimensions of this database.
    #[napi(getter)]
    pub fn dimensions(&self) -> u32 {
        self.live_dimensions()
    }

    /// Check if this is a multi-vector store.
    #[napi(getter, js_name = "isMultiVector")]
    pub fn is_multi_vector(&self) -> bool {
        self.live_is_multi_vector()
    }

    /// Check if an embedding function is configured.
    #[napi(getter, js_name = "hasEmbeddingFn")]
    pub fn has_embedding_fn(&self) -> bool {
        self.embedding_fn.is_some()
    }

    /// Check if database is empty.
    #[napi]
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.read();
        inner.store.len() == 0
    }

    /// Get database statistics.
    #[napi]
    pub fn stats(&self) -> StatsResult {
        let inner = self.inner.read();
        StatsResult {
            dimensions: self.live_dimensions(),
            count: inner.store.len() as u32,
            path: self.path.clone(),
        }
    }

    /// Get comprehensive database diagnostics.
    #[napi]
    pub fn info(&self) -> InfoResult {
        let inner = self.inner.read();
        let info = inner.store.info();
        InfoResult {
            vector_count: info.vector_count as u32,
            deleted_count: info.deleted_count as u32,
            dimensions: info.dimensions as u32,
            metric: info.metric.canonical_name().to_string(),
            frozen_segment_count: info.frozen_segment_count as u32,
            mutable_segment_vectors: info.mutable_segment_vectors as u32,
            vector_bytes: info.vector_bytes as u32,
            graph_bytes: info.graph_bytes as u32,
            total_memory_bytes: info.total_memory_bytes as u32,
            wal_entries: info.wal_entries as u32,
            is_persistent: info.is_persistent,
            hnsw_m: info.hnsw_m as u32,
            hnsw_ef_construction: info.hnsw_ef_construction as u32,
            hnsw_ef_search: info.hnsw_ef_search as u32,
            quantization: info.quantization,
            segment_capacity: info.segment_capacity as u32,
            schema: info.schema.into(),
        }
    }

    /// Get the authoritative collection schema for this database.
    #[napi]
    pub fn schema(&self) -> CollectionSchemaResult {
        let inner = self.inner.read();
        inner.store.schema().into()
    }

    /// Get current ef_search value.
    #[napi(getter, js_name = "efSearch")]
    pub fn get_ef_search(&self) -> u32 {
        let inner = self.inner.read();
        inner.store.ef_search() as u32
    }

    /// Set ef_search value.
    #[napi(setter, js_name = "efSearch")]
    pub fn set_ef_search(&self, ef_search: u32) {
        let inner = self.inner.write();
        inner.store.set_ef_search(ef_search as usize);
    }

    /// Compact the database by removing deleted records and reclaiming space.
    ///
    /// This operation removes tombstoned records, reassigns indices to be
    /// contiguous, and rebuilds the search index. Call after bulk deletes
    /// to reclaim memory and improve search performance.
    ///
    /// @returns Number of deleted records that were removed
    ///
    /// @example
    /// ```typescript
    /// // After bulk delete
    /// db.delete(staleIds);
    /// const removed = db.compact();
    /// console.log(`Removed ${removed} deleted records`);
    /// ```
    #[napi]
    pub fn compact(&self) -> Result<u32> {
        let inner = self.inner.write();
        let removed = inner.store.compact().map_err(convert_error)?;
        Ok(removed as u32)
    }

    /// Close the database and release file locks.
    ///
    /// After calling close(), the database is no longer usable.
    /// Any subsequent operations will fail or return empty results.
    ///
    /// This is useful when you need to reopen the same database path
    /// in the same process, since JavaScript doesn't have deterministic
    /// object destruction like Python's `del`.
    #[napi]
    pub fn close(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)?;
        let dummy_store = VectorStoreOptions::default()
            .dimensions(self.live_dimensions() as usize)
            .build()
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        inner.store = dummy_store;
        Ok(())
    }

    /// Optimize index for cache-efficient search.
    ///
    /// Reorders nodes for better memory locality, improving search performance by 6-40%.
    /// Call after inserting a large batch of vectors.
    ///
    /// @returns Number of nodes reordered
    #[napi]
    pub fn optimize(&self) -> Result<u32> {
        let inner = self.inner.write();
        let stats = inner.store.optimize().map_err(convert_error)?;
        Ok(stats.vectors_reordered as u32)
    }

    /// Merge another database into this one.
    ///
    /// @param other - Source database to merge from
    /// @param keyPrefix - Optional prefix for all source IDs (e.g., "subdir/")
    /// @returns Number of vectors merged
    #[napi]
    pub fn merge_from(&self, other: &VectorDatabase, key_prefix: Option<String>) -> Result<u32> {
        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Err(napi::Error::from_reason(
                "cannot merge a database into itself",
            ));
        }

        // Acquire locks in pointer-address order to prevent AB/BA deadlock
        // when a.merge_from(b) and b.merge_from(a) run concurrently.
        let self_addr = Arc::as_ptr(&self.inner) as usize;
        let other_addr = Arc::as_ptr(&other.inner) as usize;

        let count = if self_addr < other_addr {
            let mut inner = self.inner.write();
            let other_inner = other.inner.read();
            inner
                .store
                .merge_from_with_prefix(&other_inner.store, key_prefix.as_deref())
                .map_err(convert_error)?
        } else {
            let other_inner = other.inner.read();
            let mut inner = self.inner.write();
            inner
                .store
                .merge_from_with_prefix(&other_inner.store, key_prefix.as_deref())
                .map_err(convert_error)?
        };

        Ok(count as u32)
    }

    /// List all vector IDs (without loading vector data).
    ///
    /// Efficient way to get all IDs for iteration, export, or debugging.
    /// @returns Array of all vector IDs in the database
    #[napi]
    pub fn ids(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner.store.ids()
    }

    /// Get all items as array of {id, vector, metadata}.
    ///
    /// Returns all vectors with their IDs and metadata.
    /// For large datasets, consider using ids() and get() in batches.
    #[napi]
    pub fn items(&self) -> Vec<GetResult> {
        let inner = self.inner.read();
        inner
            .store
            .items()
            .into_iter()
            .map(|(id, vector, metadata)| GetResult {
                id,
                vector: Float32Array::new(vector),
                metadata,
            })
            .collect()
    }

    /// Check if an ID exists in the database.
    ///
    /// @param id - Vector ID to check
    /// @returns true if ID exists and is not deleted
    #[napi]
    pub fn exists(&self, id: String) -> bool {
        let inner = self.inner.read();
        inner.store.contains(&id)
    }

    /// Get multiple vectors by ID.
    ///
    /// Batch version of get(). More efficient than calling get() in a loop.
    ///
    /// @param ids - Array of vector IDs to retrieve
    /// @returns Array of results in same order as input, null for missing IDs
    #[napi]
    pub fn get_batch(&self, ids: Vec<String>) -> Vec<Option<GetResult>> {
        let inner = self.inner.read();
        ids.iter()
            .map(|id| {
                inner.store.get(id).map(|(vec, metadata)| GetResult {
                    id: id.clone(),
                    vector: Float32Array::new(vec.data),
                    metadata,
                })
            })
            .collect()
    }
}
