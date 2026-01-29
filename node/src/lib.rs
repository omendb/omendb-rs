//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

// Allow these lints - they're structural API design choices that work correctly
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::{
    muvera::MultiVectorConfig, MetadataFilter, QuantizationMode, Vector, VectorStore,
    VectorStoreOptions,
};
use omendb_lib::{Rerank, SearchOptions};
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract query vector from JS - accepts number[] or Float32Array
fn extract_query_vector(query: Either<Vec<f64>, Float32Array>) -> Vec<f32> {
    match query {
        Either::A(arr) => arr.into_iter().map(|x| x as f32).collect(),
        Either::B(typed) => typed.to_vec(),
    }
}

/// Extract multi-vector query from JS - accepts number[][] or Float32Array[]
fn extract_multi_vector_query(
    query: Either<Vec<Vec<f64>>, Vec<Float32Array>>,
) -> Result<Vec<Vec<f32>>> {
    match query {
        Either::A(nested) => {
            if nested.is_empty() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "multi-vector query must not be empty",
                ));
            }
            Ok(nested
                .into_iter()
                .map(|arr| arr.into_iter().map(|x| x as f32).collect())
                .collect())
        }
        Either::B(typed_arrays) => {
            if typed_arrays.is_empty() {
                return Err(Error::new(
                    Status::InvalidArg,
                    "multi-vector query must not be empty",
                ));
            }
            Ok(typed_arrays.into_iter().map(|t| t.to_vec()).collect())
        }
    }
}

/// Parse multi_vector option from JS value
fn parse_multi_vector(value: &serde_json::Value) -> Result<Option<MultiVectorConfig>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some(MultiVectorConfig::default())),
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::Object(obj) => {
            let mut config = MultiVectorConfig::default();
            if let Some(reps) = obj.get("repetitions") {
                config.repetitions = reps.as_u64().ok_or_else(|| {
                    Error::new(Status::InvalidArg, "repetitions must be a number")
                })? as u8;
            }
            if let Some(bits) = obj.get("partitionBits") {
                config.partition_bits = bits.as_u64().ok_or_else(|| {
                    Error::new(Status::InvalidArg, "partitionBits must be a number")
                })? as u8;
            }
            if let Some(seed) = obj.get("seed") {
                config.seed = seed
                    .as_u64()
                    .ok_or_else(|| Error::new(Status::InvalidArg, "seed must be a number"))?;
            }
            if let Some(d_proj) = obj.get("dProj") {
                config.d_proj = if d_proj.is_null() {
                    None
                } else {
                    Some(d_proj.as_u64().ok_or_else(|| {
                        Error::new(Status::InvalidArg, "dProj must be a number or null")
                    })? as u8)
                };
            }
            Ok(Some(config))
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "multiVector must be true, false, or { repetitions?, partitionBits?, seed?, dProj? }",
        )),
    }
}

/// Parse quantization option from JS value (bool or string)
fn parse_quantization(value: &serde_json::Value) -> Result<Option<QuantizationMode>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some(QuantizationMode::SQ8)),
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::String(s) => match s.to_lowercase().as_str() {
            "sq8" | "scalar" => Ok(Some(QuantizationMode::SQ8)),
            _ => Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Unknown quantization mode: '{}'\n\
                     Valid modes: true, 'sq8', or 'scalar' (4x smaller, ~99% recall)",
                    s
                ),
            )),
        },
        _ => Err(Error::new(
            Status::InvalidArg,
            "quantization must be true, false, or 'sq8'/'scalar'",
        )),
    }
}

/// Convert Rust error to napi Error with appropriate status
fn convert_error(err: anyhow::Error) -> Error {
    let msg = err.to_string();
    if msg.contains("dimension") {
        Error::new(Status::InvalidArg, msg)
    } else {
        Error::new(Status::GenericFailure, msg)
    }
}

/// Parse a numeric comparison operator ($gt, $gte, $lt, $lte)
fn parse_numeric_op(op: &str, key: &str, value: &JsonValue) -> Result<MetadataFilter> {
    let num = value
        .as_f64()
        .ok_or_else(|| Error::new(Status::InvalidArg, format!("{} requires a number", op)))?;
    Ok(match op {
        "$gt" => MetadataFilter::Gt(key.to_string(), num),
        "$gte" => MetadataFilter::Gte(key.to_string(), num),
        "$lt" => MetadataFilter::Lt(key.to_string(), num),
        "$lte" => MetadataFilter::Lte(key.to_string(), num),
        _ => unreachable!(),
    })
}

/// Parse JavaScript filter object into MetadataFilter
/// Supports: equality, $gt, $gte, $lt, $lte, $in, $contains, $and, $or
fn parse_filter(filter: &JsonValue) -> Result<MetadataFilter> {
    let obj = filter
        .as_object()
        .ok_or_else(|| Error::new(Status::InvalidArg, "Filter must be an object"))?;

    // Handle $and
    if let Some(and_value) = obj.get("$and") {
        let arr = and_value
            .as_array()
            .ok_or_else(|| Error::new(Status::InvalidArg, "$and must be an array"))?;
        let sub_filters: Result<Vec<MetadataFilter>> = arr.iter().map(parse_filter).collect();
        return Ok(MetadataFilter::And(sub_filters?));
    }

    // Handle $or
    if let Some(or_value) = obj.get("$or") {
        let arr = or_value
            .as_array()
            .ok_or_else(|| Error::new(Status::InvalidArg, "$or must be an array"))?;
        let sub_filters: Result<Vec<MetadataFilter>> = arr.iter().map(parse_filter).collect();
        return Ok(MetadataFilter::Or(sub_filters?));
    }

    // Parse field filters
    let mut filters = Vec::new();

    for (key, value) in obj {
        if let Some(op_obj) = value.as_object() {
            // Operator object: {"field": {"$gt": 5}}
            for (op, op_value) in op_obj {
                let filter = match op.as_str() {
                    "$eq" => MetadataFilter::Eq(key.clone(), op_value.clone()),
                    "$ne" => MetadataFilter::Ne(key.clone(), op_value.clone()),
                    "$gt" | "$gte" | "$lt" | "$lte" => parse_numeric_op(op, key, op_value)?,
                    "$in" => {
                        let arr = op_value.as_array().ok_or_else(|| {
                            Error::new(Status::InvalidArg, "$in requires an array")
                        })?;
                        MetadataFilter::In(key.clone(), arr.clone())
                    }
                    "$contains" => {
                        let s = op_value.as_str().ok_or_else(|| {
                            Error::new(Status::InvalidArg, "$contains requires a string")
                        })?;
                        MetadataFilter::Contains(key.clone(), s.to_string())
                    }
                    _ => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            format!("Unknown filter operator: {}", op),
                        ));
                    }
                };
                filters.push(filter);
            }
        } else {
            // Direct equality: {"field": value}
            filters.push(MetadataFilter::Eq(key.clone(), value.clone()));
        }
    }

    if filters.len() == 1 {
        Ok(filters.into_iter().next().expect("checked len == 1"))
    } else {
        Ok(MetadataFilter::And(filters))
    }
}

// ============================================================================
// Search Result - returned from search operations
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f64,
    /// Metadata as JSON (using serde-json feature)
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

// ============================================================================
// Vector Item - input for set operations (single-vector stores)
// ============================================================================

#[napi(object)]
pub struct VectorItem {
    pub id: String,
    /// Vector data as Float32Array
    pub vector: Float32Array,
    /// Optional metadata
    #[napi(ts_type = "Record<string, unknown> | undefined")]
    pub metadata: Option<JsonValue>,
    /// Optional document text (stored in metadata.document)
    pub document: Option<String>,
}

// ============================================================================
// Multi-Vector Item - input for set operations (multi-vector stores)
// ============================================================================

#[napi(object)]
pub struct MultiVectorItem {
    pub id: String,
    /// Multi-vector data as array of Float32Arrays
    #[napi(ts_type = "Float32Array[]")]
    pub vectors: Vec<Float32Array>,
    /// Optional metadata
    #[napi(ts_type = "Record<string, unknown> | undefined")]
    pub metadata: Option<JsonValue>,
}

// ============================================================================
// Unified Item - input for set() on any store type
// ============================================================================

#[napi(object)]
pub struct SetItem {
    pub id: String,
    /// Single vector data (for regular stores)
    pub vector: Option<Float32Array>,
    /// Multi-vector data (for multi-vector stores)
    #[napi(ts_type = "Float32Array[] | undefined")]
    pub vectors: Option<Vec<Float32Array>>,
    /// Optional metadata
    #[napi(ts_type = "Record<string, unknown> | undefined")]
    pub metadata: Option<JsonValue>,
    /// Optional document text (stored in metadata.document)
    pub document: Option<String>,
}

// ============================================================================
// Get Result - returned from get operations
// ============================================================================

#[napi(object)]
pub struct GetResult {
    pub id: String,
    pub vector: Float32Array,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

// ============================================================================
// Vector Item With Text - input for hybrid search set operations
// ============================================================================

#[napi(object)]
pub struct VectorItemWithText {
    pub id: String,
    pub vector: Float32Array,
    pub text: String,
    #[napi(ts_type = "Record<string, unknown> | undefined")]
    pub metadata: Option<JsonValue>,
}

// ============================================================================
// Text Search Result - returned from text/hybrid search
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct TextSearchResult {
    pub id: String,
    pub score: f64,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

// ============================================================================
// Hybrid Search Result with Subscores - returned from hybridSearch(subscores=true)
// ============================================================================

#[napi(object)]
#[derive(Clone)]
pub struct HybridSearchResult {
    pub id: String,
    pub score: f64,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
    /// BM25 keyword matching score (null if document only matched vector search)
    pub keyword_score: Option<f64>,
    /// Vector similarity score (null if document only matched text search)
    pub semantic_score: Option<f64>,
}

// ============================================================================
// Stats Result - returned from stats()
// ============================================================================

#[napi(object)]
pub struct StatsResult {
    pub dimensions: u32,
    pub count: u32,
    pub path: String,
}

// ============================================================================
// Inner State
// ============================================================================

struct VectorDatabaseInner {
    store: VectorStore,
}

// ============================================================================
// VectorDatabase Class
// ============================================================================

#[napi]
pub struct VectorDatabase {
    inner: Arc<RwLock<VectorDatabaseInner>>,
    path: String,
    dimensions: u32,
    is_persistent: bool,
    is_multi_vector: bool,
    /// Cache of open collection handles (same name = shared state)
    collections_cache: RwLock<HashMap<String, Arc<RwLock<VectorDatabaseInner>>>>,
}

#[napi]
impl VectorDatabase {
    /// Insert or update vectors.
    ///
    /// Works for both single-vector and multi-vector stores:
    /// - Single-vector: items have `vector` field
    /// - Multi-vector: items have `vectors` field (array of vectors)
    ///
    /// @param items - Array of {id, vector, metadata?} or {id, vectors, metadata?}
    /// @returns Array of internal indices
    #[napi]
    pub fn set(&self, items: Vec<SetItem>) -> Result<Vec<u32>> {
        if self.is_multi_vector {
            // Multi-vector store: use "vectors" field
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

            Ok((0..count as u32).collect())
        } else {
            // Single-vector store: use "vector" field
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

                    let mut metadata = item.metadata.unwrap_or(serde_json::json!({}));

                    // Handle document field - requires metadata to be an object
                    if let Some(doc) = item.document {
                        match metadata.as_object_mut() {
                            Some(obj) => {
                                obj.insert("document".to_string(), serde_json::json!(doc));
                            }
                            None => {
                                return Err(Error::from_reason(
                                    "metadata must be an object when document field is provided",
                                ));
                            }
                        }
                    }

                    Ok((item.id, Vector::new(vector.to_vec()), metadata))
                })
                .collect::<Result<Vec<_>>>()?;

            let mut inner = self.inner.write();
            let result = inner.store.set_batch(batch).map_err(convert_error)?;
            Ok(result.into_iter().map(|x| x as u32).collect())
        }
    }

    /// Search for k nearest neighbors.
    ///
    /// @param query - Query vector (number[] or Float32Array)
    /// @param k - Number of results to return
    /// @param ef - Optional search width override
    /// @param filter - Optional metadata filter (e.g., {category: "foo"} or {price: {$gt: 10}})
    /// @param maxDistance - Optional max distance threshold (filter out distant results)
    /// @returns Array of {id, distance, metadata}
    #[napi]
    pub fn search(
        &self,
        query: Either<Vec<f64>, Float32Array>,
        k: u32,
        ef: Option<u32>,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] filter: Option<JsonValue>,
        max_distance: Option<f64>,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(Error::from_reason(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }
        if let Some(max_dist) = max_distance {
            if max_dist < 0.0 {
                return Err(Error::from_reason("max_distance must be non-negative"));
            }
        }

        let query_vec = Vector::new(extract_query_vector(query));
        let ef_usize = ef.map(|e| e as usize);
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let max_dist_f32 = max_distance.map(|d| d as f32);

        // Ensure index is ready
        {
            let inner = self.inner.read();
            if inner.store.needs_index_rebuild() {
                drop(inner);
                let mut inner = self.inner.write();
                inner.store.ensure_index_ready().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();
        let results = inner
            .store
            .search_with_options_readonly(
                &query_vec,
                k as usize,
                metadata_filter.as_ref(),
                ef_usize,
                max_dist_f32,
            )
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance as f64,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Search multi-vector store with query tokens.
    ///
    /// Internal method used by unified search() for multi-vector stores.
    ///
    /// @param query - Query tokens (number[][] or Float32Array[])
    /// @param k - Number of results to return
    /// @param rerank - Enable MaxSim reranking for better quality (default: true)
    /// @param rerankFactor - Fetch k*rerankFactor candidates before reranking (default: 32)
    /// @returns Array of {id, distance, metadata}
    #[napi]
    pub fn search_multi(
        &self,
        query: Either<Vec<Vec<f64>>, Vec<Float32Array>>,
        k: u32,
        rerank: Option<bool>,
        rerank_factor: Option<u32>,
    ) -> Result<Vec<SearchResult>> {
        if !self.is_multi_vector {
            return Err(Error::new(
                Status::InvalidArg,
                "searchMulti requires a multi-vector store. Use open() with multiVector: true",
            ));
        }

        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let query_tokens = extract_multi_vector_query(query)?;

        // Build search options
        let rerank_opt = match (rerank, rerank_factor) {
            (Some(false), _) => Rerank::Off,
            (_, Some(factor)) => Rerank::Factor(factor as usize),
            _ => Rerank::On, // default: rerank enabled
        };
        let options = SearchOptions::default().rerank(rerank_opt);

        let inner = self.inner.read();
        let results = inner
            .store
            .query_with_options(&query_tokens, k as usize, &options)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance as f64,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Batch search with parallel execution (async).
    ///
    /// Runs searches in parallel using rayon on a blocking thread pool,
    /// keeping the Node.js event loop free.
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<Either<Vec<f64>, Float32Array>>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(Error::from_reason(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        // Extract query vectors upfront (cheap)
        let query_vecs: Vec<Vector> = queries
            .into_iter()
            .map(|q| Vector::new(extract_query_vector(q)))
            .collect();

        // Ensure index is ready (may block briefly)
        {
            let mut inner = self.inner.write();
            inner.store.ensure_index_ready().map_err(convert_error)?;
        }

        // Clone Arc for spawn_blocking
        let inner_arc = Arc::clone(&self.inner);
        let k_usize = k as usize;
        let ef_usize = ef.map(|e| e as usize);

        // Run CPU-intensive search on blocking thread pool
        let output = tokio::task::spawn_blocking(move || {
            let inner = inner_arc.read();
            let all_results =
                inner
                    .store
                    .search_batch_with_metadata(&query_vecs, k_usize, ef_usize);

            // Convert results
            let mut output = Vec::with_capacity(all_results.len());
            for result in all_results {
                let results = result?;
                output.push(
                    results
                        .into_iter()
                        .map(|r| SearchResult {
                            id: r.id,
                            distance: r.distance as f64,
                            metadata: r.metadata,
                        })
                        .collect(),
                );
            }
            Ok::<_, anyhow::Error>(output)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task join error: {e}")))?
        .map_err(convert_error)?;

        Ok(output)
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
    /// @returns Number of vectors deleted
    #[napi]
    pub fn delete(&self, ids: Vec<String>) -> Result<u32> {
        let mut inner = self.inner.write();
        let result = inner.store.delete_batch(&ids).map_err(convert_error)?;
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

        let mut inner = self.inner.write();
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

    /// Update a vector's data and/or metadata.
    #[napi]
    pub fn update(
        &self,
        id: String,
        vector: Either<Vec<f64>, Float32Array>,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] metadata: Option<JsonValue>,
    ) -> Result<()> {
        let vec = Some(Vector::new(extract_query_vector(vector)));
        let mut inner = self.inner.write();
        inner
            .store
            .update(&id, vec, metadata)
            .map_err(convert_error)?;
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
        self.dimensions
    }

    /// Check if this is a multi-vector store.
    #[napi(getter, js_name = "isMultiVector")]
    pub fn is_multi_vector(&self) -> bool {
        self.is_multi_vector
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
            dimensions: self.dimensions,
            count: inner.store.len() as u32,
            path: self.path.clone(),
        }
    }

    /// Get current ef_search value.
    #[napi(getter, js_name = "efSearch")]
    pub fn get_ef_search(&self) -> u32 {
        let inner = self.inner.read();
        inner.store.get_ef_search().unwrap_or(100) as u32
    }

    /// Set ef_search value.
    #[napi(setter, js_name = "efSearch")]
    pub fn set_ef_search(&self, ef_search: u32) {
        let mut inner = self.inner.write();
        inner.store.set_ef_search(ef_search as usize);
    }

    /// Get or create a named collection.
    ///
    /// Collection handles share state - changes made through one handle
    /// are immediately visible through another (no flush required).
    #[napi]
    pub fn collection(&self, name: String) -> Result<VectorDatabase> {
        if name.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name cannot be empty",
            ));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        if !self.is_persistent {
            return Err(Error::new(
                Status::InvalidArg,
                "Collections require persistent storage",
            ));
        }

        // Check cache first
        {
            let cache = self.collections_cache.read();
            if let Some(cached_inner) = cache.get(&name) {
                let base_path = std::path::Path::new(&self.path);
                let collection_path = base_path.join("collections").join(&name);
                return Ok(VectorDatabase {
                    inner: Arc::clone(cached_inner),
                    path: collection_path.to_string_lossy().to_string(),
                    dimensions: self.dimensions,
                    is_persistent: true,
                    is_multi_vector: false,
                    collections_cache: RwLock::new(HashMap::new()),
                });
            }
        }

        // Not in cache - create new collection
        let mut cache = self.collections_cache.write();

        // Double-check after acquiring write lock
        if let Some(cached_inner) = cache.get(&name) {
            let base_path = std::path::Path::new(&self.path);
            let collection_path = base_path.join("collections").join(&name);
            return Ok(VectorDatabase {
                inner: Arc::clone(cached_inner),
                path: collection_path.to_string_lossy().to_string(),
                dimensions: self.dimensions,
                is_persistent: true,
                is_multi_vector: false,
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        std::fs::create_dir_all(collection_path.parent().unwrap()).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to create collections directory: {}", e),
            )
        })?;

        let store = if self.dimensions == 0 || self.dimensions == 128 {
            VectorStore::open(&collection_path).map_err(convert_error)?
        } else {
            VectorStore::open_with_dimensions(&collection_path, self.dimensions as usize)
                .map_err(convert_error)?
        };

        let inner = Arc::new(RwLock::new(VectorDatabaseInner { store }));

        // Cache and return
        cache.insert(name, Arc::clone(&inner));

        Ok(VectorDatabase {
            inner,
            path: collection_path.to_string_lossy().to_string(),
            dimensions: self.dimensions,
            is_persistent: true,
            is_multi_vector: false,
            collections_cache: RwLock::new(HashMap::new()),
        })
    }

    /// List all collections.
    #[napi]
    pub fn collections(&self) -> Result<Vec<String>> {
        if !self.is_persistent {
            return Ok(Vec::new());
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");

        if !collections_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        let entries = std::fs::read_dir(&collections_dir).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to read collections: {}", e),
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to read entry: {}", e),
                )
            })?;
            // Collections are stored as .omen files
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(collection_name) = name.strip_suffix(".omen") {
                        names.push(collection_name.to_string());
                    }
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Delete a collection.
    #[napi]
    pub fn delete_collection(&self, name: String) -> Result<()> {
        if !self.is_persistent {
            return Err(Error::new(
                Status::InvalidArg,
                "Collections require persistent storage",
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");
        let omen_path = collections_dir.join(format!("{}.omen", name));
        let wal_path = collections_dir.join(format!("{}.wal", name));

        if !omen_path.exists() {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Collection '{}' does not exist", name),
            ));
        }

        // Remove from cache first
        {
            let mut cache = self.collections_cache.write();
            cache.remove(&name);
        }

        // Remove .omen file
        std::fs::remove_file(&omen_path).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to delete collection: {}", e),
            )
        })?;

        // Remove .wal file if it exists
        let _ = std::fs::remove_file(&wal_path);

        Ok(())
    }

    // =========================================================================
    // Hybrid Search Methods
    // =========================================================================

    /// Enable text search for hybrid (vector + text) search.
    ///
    /// Must be called before using setWithText() or hybridSearch().
    #[napi]
    pub fn enable_text_search(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.store.enable_text_search().map_err(convert_error)
    }

    /// Check if text search is enabled.
    #[napi(getter)]
    pub fn has_text_search(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_text_search()
    }

    /// Set vectors with associated text for hybrid search.
    ///
    /// @param items - Array of {id, vector, text, metadata?}
    /// @returns Array of internal indices
    #[napi]
    pub fn set_with_text(&self, items: Vec<VectorItemWithText>) -> Result<Vec<u32>> {
        let mut inner = self.inner.write();

        if !inner.store.has_text_search() {
            return Err(Error::new(
                Status::GenericFailure,
                "Text search not enabled. Call enableTextSearch() first.",
            ));
        }

        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let metadata = item.metadata.unwrap_or(serde_json::json!({}));

            let index = inner
                .store
                .set_with_text(
                    item.id,
                    Vector::new(item.vector.to_vec()),
                    &item.text,
                    metadata,
                )
                .map_err(convert_error)?;

            results.push(index as u32);
        }

        Ok(results)
    }

    /// Search using text only (BM25 scoring).
    ///
    /// @param query - Text query
    /// @param k - Number of results
    /// @returns Array of {id, score, metadata}
    #[napi]
    pub fn text_search(&self, query: String, k: u32) -> Result<Vec<TextSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let inner = self.inner.read();

        let results = inner
            .store
            .text_search(&query, k as usize)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|(id, score)| {
                let metadata = inner
                    .store
                    .get_metadata_by_id(&id)
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                TextSearchResult {
                    id,
                    score: score as f64,
                    metadata,
                }
            })
            .collect())
    }

    /// Hybrid search combining vector similarity and text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine HNSW and BM25 results.
    ///
    /// @param queryVector - Query embedding
    /// @param queryText - Text query for BM25
    /// @param k - Number of results
    /// @param filter - Optional metadata filter
    /// @param alpha - Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5)
    /// @param rrfK - RRF constant (default=60, higher reduces rank influence)
    /// @param subscores - Return separate keyword_score and semantic_score (default: false)
    /// @returns Array of {id, score, metadata, keyword_score?, semantic_score?}
    #[napi]
    pub fn hybrid_search(
        &self,
        query_vector: Either<Vec<f64>, Float32Array>,
        query_text: String,
        k: u32,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] filter: Option<JsonValue>,
        alpha: Option<f64>,
        rrf_k: Option<u32>,
        subscores: Option<bool>,
    ) -> Result<Vec<HybridSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }
        if let Some(a) = alpha {
            if !(0.0..=1.0).contains(&a) {
                return Err(Error::from_reason(format!(
                    "alpha must be between 0.0 and 1.0, got {}",
                    a
                )));
            }
        }
        if let Some(rrf) = rrf_k {
            if rrf == 0 {
                return Err(Error::from_reason("rrf_k must be greater than 0"));
            }
        }

        let query_vec = Vector::new(extract_query_vector(query_vector));
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let alpha_f32 = alpha.map(|a| a as f32);
        let rrf_k_usize = rrf_k.map(|k| k as usize);

        let inner = self.inner.read();

        // Use subscores path when requested
        if subscores.unwrap_or(false) {
            let results = if let Some(f) = metadata_filter {
                inner
                    .store
                    .hybrid_search_with_filter_subscores(
                        &query_vec,
                        &query_text,
                        k as usize,
                        &f,
                        alpha_f32,
                        rrf_k_usize,
                    )
                    .map_err(convert_error)?
            } else {
                inner
                    .store
                    .hybrid_search_with_subscores(
                        &query_vec,
                        &query_text,
                        k as usize,
                        alpha_f32,
                        rrf_k_usize,
                    )
                    .map_err(convert_error)?
            };

            return Ok(results
                .into_iter()
                .map(|(hybrid_result, metadata)| HybridSearchResult {
                    id: hybrid_result.id,
                    score: hybrid_result.score as f64,
                    metadata,
                    keyword_score: hybrid_result.keyword_score.map(|s| s as f64),
                    semantic_score: hybrid_result.semantic_score.map(|s| s as f64),
                })
                .collect());
        }

        // Standard path without subscores
        let results = if let Some(f) = metadata_filter {
            inner
                .store
                .hybrid_search_with_filter_rrf_k(
                    &query_vec,
                    &query_text,
                    k as usize,
                    &f,
                    alpha_f32,
                    rrf_k_usize,
                )
                .map_err(convert_error)?
        } else {
            inner
                .store
                .hybrid_search_with_rrf_k(
                    &query_vec,
                    &query_text,
                    k as usize,
                    alpha_f32,
                    rrf_k_usize,
                )
                .map_err(convert_error)?
        };

        Ok(results
            .into_iter()
            .map(|(id, score, metadata)| HybridSearchResult {
                id,
                score: score as f64,
                metadata,
                keyword_score: None,
                semantic_score: None,
            })
            .collect())
    }

    /// Flush pending changes to disk.
    ///
    /// For hybrid search, this commits text index changes.
    #[napi]
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)
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
        let mut inner = self.inner.write();
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
        // Flush first to ensure all data is persisted
        inner.store.flush().map_err(convert_error)?;
        // Replace with minimal in-memory store to release file lock
        let dummy_store = VectorStoreOptions::default()
            .dimensions(self.dimensions as usize)
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
        let mut inner = self.inner.write();
        let result = inner.store.optimize().map_err(convert_error)?;
        Ok(result as u32)
    }

    // =========================================================================
    // Merge Methods
    // =========================================================================

    /// Merge another database into this one.
    #[napi]
    pub fn merge_from(&self, other: &VectorDatabase) -> Result<u32> {
        let mut inner = self.inner.write();
        let other_inner = other.inner.read();

        let count = inner
            .store
            .merge_from(&other_inner.store)
            .map_err(convert_error)?;

        Ok(count as u32)
    }

    // =========================================================================
    // Iteration Methods
    // =========================================================================

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

// ============================================================================
// Open Options
// ============================================================================

/// Configuration options for opening a vector database.
///
/// All fields are optional with sensible defaults:
/// - dimensions: 128 (auto-detected on first insert if not specified)
/// - m: 16 (HNSW neighbors per node, higher = better recall, more memory)
/// - efConstruction: 100 (build quality, higher = better graph, slower build)
/// - efSearch: 100 (search quality, higher = better recall, slower search)
/// - quantization: null (true/"sq8" for 4x compression, ~99% recall)
/// - rescore: true when quantization enabled (rerank candidates with exact distance)
/// - oversample: 3.0 (fetch k*oversample candidates when rescoring)
/// - metric: "l2" (distance metric: "l2", "euclidean", "cosine", "dot", "ip")
#[napi(object)]
pub struct OpenOptions {
    /// Vector dimensions (default: 128, auto-detected on first insert)
    pub dimensions: Option<u32>,
    /// HNSW M parameter: neighbors per node (default: 16, range: 4-64)
    pub m: Option<u32>,
    /// HNSW ef_construction: build quality (default: 100, must be >= m)
    pub ef_construction: Option<u32>,
    /// HNSW ef_search: search quality/speed tradeoff (default: 100)
    pub ef_search: Option<u32>,
    /// Quantization mode (default: null = no quantization)
    /// - true or "sq8": SQ8 4x compression, ~99% recall (RECOMMENDED)
    /// - false/null: Full precision (no quantization)
    #[napi(ts_type = "boolean | string | number | null | undefined")]
    pub quantization: Option<serde_json::Value>,
    /// Rescore candidates with exact distance (default: true when quantization enabled)
    /// Set to false for maximum speed at the cost of ~20% recall
    pub rescore: Option<bool>,
    /// Oversampling factor for rescoring (default: 3.0)
    /// Fetches k*oversample candidates then reranks to return top k
    pub oversample: Option<f64>,
    /// Distance metric: "l2"/"euclidean" (default), "cosine", "dot"/"ip"
    pub metric: Option<String>,
    /// Enable multi-vector mode for ColBERT-style retrieval
    /// - true: Enable with default config (repetitions=8, partition_bits=4, dProj=16)
    /// - { repetitions?, partitionBits?, seed?, dProj? }: Custom config
    /// - dProj: Dimension projection (16 = 8x smaller FDE, null = full token dim)
    /// - false/null: Disabled (default, single-vector mode)
    #[napi(ts_type = "boolean | { repetitions?: number; partitionBits?: number; seed?: number; dProj?: number | null } | null | undefined")]
    pub multi_vector: Option<serde_json::Value>,
}

// ============================================================================
// Module-level open function
// ============================================================================

/// Open or create a vector database.
///
/// @param path - Database directory path (use ":memory:" for in-memory)
/// @param options - Optional configuration (see OpenOptions for defaults)
/// @returns VectorDatabase instance
///
/// @example
/// ```javascript
/// // Simple usage with defaults
/// const db = omendb.open("./mydb");
///
/// // With custom HNSW parameters
/// const db = omendb.open("./mydb", {
///   dimensions: 384,
///   m: 32,
///   efConstruction: 200,
///   efSearch: 150
/// });
///
/// // With SQ8 quantization (4x memory reduction, ~99% recall)
/// const db = omendb.open("./mydb", {
///   dimensions: 128,
///   quantization: true  // or "sq8"
/// });
///
/// // Quantization with custom rescore settings
/// const db = omendb.open("./mydb", {
///   dimensions: 128,
///   quantization: true,
///   rescore: false,    // Disable rescore for max speed
///   oversample: 5.0    // Or increase oversample for better recall
/// });
/// ```
#[napi]
pub fn open(path: String, options: Option<OpenOptions>) -> Result<VectorDatabase> {
    let opts = options.unwrap_or(OpenOptions {
        dimensions: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        quantization: None,
        rescore: None,
        oversample: None,
        metric: None,
        multi_vector: None,
    });

    let dimensions = opts.dimensions.unwrap_or(128) as usize;
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);
    let ef_search = opts.ef_search.map(|v| v as usize);
    let rescore = opts.rescore;
    let oversample = opts.oversample;

    // Parse quantization (handles true or "sq8")
    let quant_mode = opts
        .quantization
        .as_ref()
        .map(parse_quantization)
        .transpose()?
        .flatten();

    // Parse multi_vector config
    let multi_vector_config = opts
        .multi_vector
        .as_ref()
        .map(parse_multi_vector)
        .transpose()?
        .flatten();

    // Check if multi-vector mode is enabled
    let is_multi_vector = multi_vector_config.is_some();

    // Validate multi-vector constraints
    if is_multi_vector && quant_mode.is_some() {
        return Err(Error::new(
            Status::InvalidArg,
            "multi-vector stores do not support quantization yet",
        ));
    }

    // Validate parameters
    if dimensions == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "dimensions must be greater than 0",
        ));
    }

    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("m must be between 4 and 64, got {}", m_val),
            ));
        }
    }

    if let (Some(ef_val), Some(m_val)) = (ef_construction, m) {
        if ef_val < m_val {
            return Err(Error::new(
                Status::InvalidArg,
                format!("ef_construction ({}) must be >= m ({})", ef_val, m_val),
            ));
        }
    }

    if let Some(factor) = oversample {
        if factor < 1.0 {
            return Err(Error::new(
                Status::InvalidArg,
                format!("oversample must be >= 1.0, got {}", factor),
            ));
        }
    }

    // Validate metric
    if let Some(ref m) = opts.metric {
        match m.to_lowercase().as_str() {
            "l2" | "euclidean" | "cosine" | "dot" | "ip" => {}
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "Unknown metric: '{}'. Valid: l2, euclidean, cosine, dot, ip",
                        m
                    ),
                ));
            }
        }
    }

    // Build options from parameters
    let mut store_options = VectorStoreOptions::default().dimensions(dimensions);

    if let Some(m_val) = m {
        store_options = store_options.m(m_val);
    }
    if let Some(ef_con) = ef_construction {
        store_options = store_options.ef_construction(ef_con);
    }
    if let Some(ef_s) = ef_search {
        store_options = store_options.ef_search(ef_s);
    }
    if let Some(ref mode) = quant_mode {
        store_options = store_options.quantization(mode.clone());
    }
    if let Some(rescore_val) = rescore {
        store_options = store_options.rescore(rescore_val);
    }
    if let Some(oversample_val) = oversample {
        store_options = store_options.oversample(oversample_val as f32);
    }
    if let Some(ref metric_str) = opts.metric {
        store_options = store_options
            .metric(metric_str)
            .map_err(|e| Error::new(Status::InvalidArg, e))?;
    }

    // Handle :memory: for in-memory database
    if path == ":memory:" {
        let store = if let Some(mv_config) = multi_vector_config {
            VectorStore::multi_vector_with(dimensions, mv_config)
        } else {
            store_options.build().map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to create store: {}", e),
                )
            })?
        };

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: dimensions as u32,
            is_persistent: false,
            is_multi_vector,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Check if .omen file exists
    let db_path = std::path::Path::new(&path);
    let omen_path = if db_path.extension().is_some_and(|ext| ext == "omen") {
        db_path.to_path_buf()
    } else {
        let mut omen = db_path.as_os_str().to_os_string();
        omen.push(".omen");
        std::path::PathBuf::from(omen)
    };

    // Open existing database (may have multi-vector config)
    if omen_path.exists() {
        let store = VectorStore::open(&path).map_err(convert_error)?;
        let is_mv = store.is_multi_vector();

        // Conflict check: opening existing single-vector store with multiVector: true
        if is_multi_vector && !is_mv {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot open existing single-vector database with multiVector: true",
            ));
        }

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: dimensions as u32,
            is_persistent: true,
            is_multi_vector: is_mv,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Create new persistent store
    if let Some(mv_config) = multi_vector_config {
        // New multi-vector persistent store
        let store = VectorStore::multi_vector_with(dimensions, mv_config)
            .persist(&path)
            .map_err(convert_error)?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: dimensions as u32,
            is_persistent: true,
            is_multi_vector: true,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Check if enabling quantization on existing non-empty database
    if db_path.exists() && quant_mode.is_some() {
        let existing = VectorStore::open(&path).map_err(convert_error)?;
        if !existing.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot enable quantization on existing database. Create a new database with quantization.",
            ));
        }
    }

    // Open single-vector persistent store with options
    let store = store_options.open(&path).map_err(convert_error)?;

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
        path,
        dimensions: dimensions as u32,
        is_persistent: true,
        is_multi_vector: false,
        collections_cache: RwLock::new(HashMap::new()),
    })
}
