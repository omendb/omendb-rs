//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

// Allow these lints - they're structural API design choices that work correctly
#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use omendb_lib::omen::Metric;
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

/// Convert raw distance to a normalized similarity score (0-1, higher = more similar).
fn distance_to_score(distance: f64, metric: Metric) -> f64 {
    match metric {
        Metric::L2 => 1.0 / (1.0 + distance),
        Metric::Cosine => 1.0 - distance,
        Metric::InnerProduct => -distance, // IP distance is -dot, so score is dot product
    }
}

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
            if let Some(pool_factor) = obj.get("poolFactor") {
                config.pool_factor = if pool_factor.is_null() {
                    None
                } else {
                    Some(pool_factor.as_u64().ok_or_else(|| {
                        Error::new(Status::InvalidArg, "poolFactor must be a number or null")
                    })? as u8)
                };
            }
            Ok(Some(config))
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "multiVector must be true, false, or { repetitions?, partitionBits?, seed?, dProj?, poolFactor? }",
        )),
    }
}

/// Parse quantization option from JS value (bool or string)
fn parse_quantization(value: &serde_json::Value) -> Result<Option<QuantizationMode>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Bool(true) => Ok(Some(QuantizationMode::SQ8)),
        serde_json::Value::Bool(false) => Ok(None),
        serde_json::Value::String(s) => {
            let lower = s.to_lowercase();
            match lower.as_str() {
                "sq8" | "scalar" => Ok(Some(QuantizationMode::SQ8)),
                "rabitq" | "binary" => Ok(Some(QuantizationMode::rabitq())),
                _ => Err(Error::new(
                    Status::InvalidArg,
                    format!(
                        "Unknown quantization mode: '{}'. Valid: true, 'sq8', 'rabitq'",
                        s
                    ),
                )),
            }
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            "quantization must be true, false, 'sq8', or 'rabitq'",
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

    // Handle $not
    if let Some(not_value) = obj.get("$not") {
        let inner = parse_filter(not_value)?;
        return Ok(MetadataFilter::Not(Box::new(inner)));
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
    /// Normalized similarity score (0-1, higher = more similar)
    pub score: f64,
    /// Metadata as JSON (using serde-json feature)
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
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
    /// Optional text for hybrid search (auto-enables text search, stored in metadata.text)
    pub text: Option<String>,
    /// Optional document for auto-embedding via embeddingFn
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

/// Type alias for embedding function: (texts: string[]) => Float32Array[]
/// CalleeHandled = false so the JS function is called directly with (value), not (err, value)
type EmbeddingFn = ThreadsafeFunction<Vec<String>, Vec<Float32Array>, Vec<String>, Status, false>;

#[napi]
pub struct VectorDatabase {
    inner: Arc<RwLock<VectorDatabaseInner>>,
    path: String,
    dimensions: u32,
    is_persistent: bool,
    is_multi_vector: bool,
    /// Optional embedding function for auto-embedding documents
    embedding_fn: Option<Arc<EmbeddingFn>>,
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
    /// When any item includes a `text` field, text search is automatically enabled.
    /// This allows immediate use of searchHybrid() without calling enableTextSearch().
    ///
    /// @param items - Array of {id, vector, metadata?, text?} or {id, vectors, metadata?}
    /// @returns Number of vectors inserted/updated
    #[napi]
    pub async fn set(&self, items: Vec<SetItem>) -> Result<u32> {
        // If any items have documents, embed them first
        let has_documents = items.iter().any(|item| item.document.is_some());
        let items = if has_documents {
            let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                Error::from_reason(
                    "No embedding function configured. Pass embeddingFn to open() or provide vectors directly.",
                )
            })?;

            // Collect documents that need embedding
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

            // Validate: can't have both vector and document
            for &i in &doc_indices {
                if items[i].vector.is_some() {
                    return Err(Error::from_reason(format!(
                        "Item '{}': cannot have both 'vector' and 'document' - use one or the other",
                        items[i].id
                    )));
                }
            }

            // Call embedding function (async - returns Float32Array[])
            let result: Vec<Float32Array> = emb_fn.call_async(docs).await?;

            if result.len() != doc_indices.len() {
                return Err(Error::from_reason(format!(
                    "embeddingFn returned {} vectors for {} documents",
                    result.len(),
                    doc_indices.len()
                )));
            }

            // Replace document items with embedded vectors
            let mut items = items;
            for (idx, embedded) in doc_indices.into_iter().zip(result) {
                items[idx].vector = Some(embedded);
                items[idx].document = None;
            }
            items
        } else {
            items
        };

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

            Ok(count as u32)
        } else {
            // Check if any items have text - auto-enable text search
            let has_text = items.iter().any(|item| item.text.is_some());

            let mut inner = self.inner.write();

            // Auto-enable text search if any item has text
            if has_text && !inner.store.has_text_search() {
                inner.store.enable_text_search().map_err(convert_error)?;
            }

            // Process items - use different paths for text vs non-text
            if has_text {
                // Mixed path: some items may have text, process individually
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
                        // Check for conflict: text field + metadata.text
                        if let Some(obj) = metadata.as_object() {
                            if obj.contains_key("text") {
                                return Err(Error::from_reason(format!(
                                    "Item '{}': cannot have both 'text' field and 'metadata.text' - use one or the other",
                                    item.id
                                )));
                            }
                        }
                        // Store text in metadata.text for retrieval
                        if let Some(obj) = metadata.as_object_mut() {
                            obj.insert("text".to_string(), serde_json::json!(text));
                        }
                        inner
                            .store
                            .set_with_text(item.id, Vector::new(vector.to_vec()), text, metadata)
                            .map_err(convert_error)?;
                    } else {
                        inner
                            .store
                            .set(item.id, Vector::new(vector.to_vec()), metadata)
                            .map_err(convert_error)?;
                    }
                    count += 1;
                }
                Ok(count)
            } else {
                // Fast path: no text, use batch insert
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

    /// Search for k nearest neighbors.
    ///
    /// @param query - Query vector (number[] or Float32Array)
    /// @param k - Number of results to return
    /// @param options - Optional search options: {filter?, ef?, maxDistance?}
    /// @returns Array of {id, distance, score, metadata}
    ///
    /// @example
    /// ```javascript
    /// // Basic search
    /// db.search([1, 0, 0, 0], 10);
    ///
    /// // With options
    /// db.search([1, 0, 0, 0], 10, { filter: { category: "A" }, ef: 200 });
    /// db.search([1, 0, 0, 0], 10, { maxDistance: 0.5 });
    /// ```
    #[napi]
    pub async fn search(
        &self,
        #[napi(ts_arg_type = "Array<number> | Float32Array | string")]
        query: Either3<Vec<f64>, Float32Array, String>,
        k: u32,
        #[napi(ts_arg_type = "{ filter?: Record<string, unknown>; ef?: number; maxDistance?: number } | undefined")]
        options: Option<JsonValue>,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        // Parse options object
        let (filter, ef, max_distance) = if let Some(ref opts) = options {
            let filter = opts.get("filter").cloned();
            let ef = opts
                .get("ef")
                .and_then(|v| v.as_u64().map(|n| n as u32));
            let max_distance = opts.get("maxDistance").and_then(|v| v.as_f64());
            (filter, ef, max_distance)
        } else {
            (None, None, None)
        };

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
                return Err(Error::from_reason("maxDistance must be non-negative"));
            }
        }

        // Handle string query via embedding function
        let query_vec = match query {
            Either3::C(text) => {
                let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "String query requires an embedding function. Pass embeddingFn to open() or provide a vector query.",
                    )
                })?;
                let result: Vec<Float32Array> = emb_fn.call_async(vec![text]).await?;
                if result.is_empty() {
                    return Err(Error::from_reason("embeddingFn returned empty result"));
                }
                Vector::new(result[0].to_vec())
            }
            Either3::A(arr) => Vector::new(arr.into_iter().map(|x| x as f32).collect()),
            Either3::B(typed) => Vector::new(typed.to_vec()),
        };

        // Validate query dimensions match the database
        let expected_dims = self.dimensions;
        if expected_dims > 0 && query_vec.dim() != expected_dims as usize {
            return Err(Error::from_reason(format!(
                "Query vector dimension ({}) does not match database dimension ({})",
                query_vec.dim(),
                expected_dims
            )));
        }

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
        let metric = inner.store.metric();
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
            .map(|r| {
                let distance = r.distance as f64;
                SearchResult {
                    id: r.id,
                    distance,
                    score: distance_to_score(distance, metric),
                    metadata: r.metadata,
                }
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
        let metric = inner.store.metric();
        let results = inner
            .store
            .query_with_options(&query_tokens, k as usize, &options)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let distance = r.distance as f64;
                SearchResult {
                    id: r.id,
                    distance,
                    score: distance_to_score(distance, metric),
                    metadata: r.metadata,
                }
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

        // Capture metric before moving into closure
        let metric = {
            let inner = self.inner.read();
            inner.store.metric()
        };

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
                        .map(|r| {
                            let distance = r.distance as f64;
                            SearchResult {
                                id: r.id,
                                distance,
                                score: distance_to_score(distance, metric),
                                metadata: r.metadata,
                            }
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
        let mut inner = self.inner.write();
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
        #[napi(ts_arg_type = "{ vector?: number[] | Float32Array; metadata?: Record<string, unknown>; text?: string }")]
        options: JsonValue,
    ) -> Result<()> {
        let vector_val = options.get("vector");
        let metadata_val = options.get("metadata").cloned();
        let text_val = options.get("text").and_then(|v| v.as_str()).map(String::from);

        if vector_val.is_none() && metadata_val.is_none() && text_val.is_none() {
            return Err(Error::from_reason(
                "update() requires at least one of vector, metadata, or text",
            ));
        }

        // Parse vector if provided
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
                return Err(Error::from_reason(
                    "vector must be an array of numbers",
                ));
            }
        } else {
            None
        };

        let mut inner = self.inner.write();

        // Handle text update - requires get, modify, set
        if let Some(ref new_text) = text_val {
            // Get existing data
            let (existing_vec, existing_meta) = inner.store.get(&id).ok_or_else(|| {
                Error::from_reason(format!("Vector with ID '{}' not found", id))
            })?;

            // Determine final vector
            let final_vec = vector.unwrap_or(existing_vec);

            // Determine final metadata, incorporating new text
            let mut final_meta = metadata_val.unwrap_or(existing_meta);
            if let Some(obj) = final_meta.as_object_mut() {
                obj.insert("text".to_string(), serde_json::json!(new_text));
            } else {
                // metadata was not an object - create new object with text
                final_meta = serde_json::json!({"text": new_text});
            }

            // Re-index text and update vector/metadata
            if inner.store.has_text_search() {
                inner
                    .store
                    .set_with_text(id, final_vec, new_text, final_meta)
                    .map_err(convert_error)?;
            } else {
                inner
                    .store
                    .set(id, final_vec, final_meta)
                    .map_err(convert_error)?;
            }
        } else {
            // No text update - use standard update path
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
        self.dimensions
    }

    /// Check if this is a multi-vector store.
    #[napi(getter, js_name = "isMultiVector")]
    pub fn is_multi_vector(&self) -> bool {
        self.is_multi_vector
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
    pub fn collection(
        &self,
        name: String,
        #[napi(ts_arg_type = "((texts: string[]) => Float32Array[]) | undefined")]
        embedding_fn: Option<EmbeddingFn>,
    ) -> Result<VectorDatabase> {
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

        // Resolve embedding function: override > inherit from parent
        let col_embedding_fn = embedding_fn
            .map(Arc::new)
            .or_else(|| self.embedding_fn.clone());

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
                    embedding_fn: col_embedding_fn.clone(),
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
                embedding_fn: col_embedding_fn.clone(),
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

        let store = if self.dimensions == 0 {
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
            embedding_fn: col_embedding_fn,
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

        // Remove files first, then cache (if file deletion fails, cache stays consistent)
        std::fs::remove_file(&omen_path).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to delete collection: {}", e),
            )
        })?;

        // Remove .wal file if it exists
        let _ = std::fs::remove_file(&wal_path);

        // Remove from cache only after files are deleted
        {
            let mut cache = self.collections_cache.write();
            cache.remove(&name);
        }

        Ok(())
    }

    // =========================================================================
    // Hybrid Search Methods
    // =========================================================================

    /// Check if text search is enabled.
    ///
    /// Text search is automatically enabled when using set() with text field.
    #[napi(getter)]
    pub fn has_text_search(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_text_search()
    }

    /// Search using text only (BM25 scoring).
    ///
    /// @param query - Text query
    /// @param k - Number of results
    /// @returns Array of {id, score, metadata}
    #[napi(js_name = "searchText")]
    pub fn search_text(&self, query: String, k: u32) -> Result<Vec<TextSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        // Auto-flush text index to ensure search sees latest inserts
        {
            let mut inner = self.inner.write();
            if inner.store.has_text_search() {
                inner.store.flush().map_err(convert_error)?;
            }
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
    /// @param options - Optional: {filter?, alpha?, rrfK?, subscores?}
    /// @returns Array of {id, score, metadata, keywordScore?, semanticScore?}
    ///
    /// @example
    /// ```javascript
    /// // Basic hybrid search
    /// db.searchHybrid([1, 0, 0, 0], "machine learning", 10);
    ///
    /// // With options
    /// db.searchHybrid([1, 0, 0, 0], "query", 10, {
    ///   filter: { type: "ml" },
    ///   alpha: 0.7,
    ///   rrfK: 60,
    ///   subscores: true
    /// });
    /// ```
    #[napi(js_name = "searchHybrid")]
    pub async fn search_hybrid(
        &self,
        #[napi(ts_arg_type = "Array<number> | Float32Array | string")]
        query_vector: Either3<Vec<f64>, Float32Array, String>,
        query_text: Option<String>,
        k: u32,
        #[napi(ts_arg_type = "{ filter?: Record<string, unknown>; alpha?: number; rrfK?: number; subscores?: boolean } | undefined")]
        options: Option<JsonValue>,
    ) -> Result<Vec<HybridSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        // Parse options object
        let (filter, alpha, rrf_k, subscores) = if let Some(ref opts) = options {
            let filter = opts.get("filter").cloned();
            let alpha = opts.get("alpha").and_then(|v| v.as_f64());
            let rrf_k = opts.get("rrfK").and_then(|v| v.as_u64().map(|n| n as u32));
            let subscores = opts.get("subscores").and_then(|v| v.as_bool());
            (filter, alpha, rrf_k, subscores)
        } else {
            (None, None, None, None)
        };

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
                return Err(Error::from_reason("rrfK must be greater than 0"));
            }
        }

        // Handle string query: auto-embed and use as text query
        let (query_vec, actual_query_text) = match query_vector {
            Either3::C(text) => {
                let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "String query requires an embedding function. Pass embeddingFn to open() or provide (vector, text) arguments.",
                    )
                })?;
                let result: Vec<Float32Array> = emb_fn.call_async(vec![text.clone()]).await?;
                if result.is_empty() {
                    return Err(Error::from_reason("embeddingFn returned empty result"));
                }
                let vec = Vector::new(result[0].to_vec());
                let text_q = query_text.unwrap_or(text);
                (vec, text_q)
            }
            Either3::A(arr) => {
                let text_q = query_text.ok_or_else(|| {
                    Error::from_reason("query_text is required when query_vector is provided")
                })?;
                (
                    Vector::new(arr.into_iter().map(|x| x as f32).collect()),
                    text_q,
                )
            }
            Either3::B(typed) => {
                let text_q = query_text.ok_or_else(|| {
                    Error::from_reason("query_text is required when query_vector is provided")
                })?;
                (Vector::new(typed.to_vec()), text_q)
            }
        };
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let alpha_f32 = alpha.map(|a| a as f32);
        let rrf_k_usize = rrf_k.map(|k| k as usize);

        // Auto-flush text index to ensure search sees latest inserts
        {
            let mut inner = self.inner.write();
            if inner.store.has_text_search() {
                inner.store.flush().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();

        // Use subscores path when requested
        if subscores.unwrap_or(false) {
            let results = if let Some(f) = metadata_filter {
                inner
                    .store
                    .hybrid_search_with_filter_subscores(
                        &query_vec,
                        &actual_query_text,
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
                        &actual_query_text,
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
                    &actual_query_text,
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
                    &actual_query_text,
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
/// - quantization: null (true/"sq8" for 4x compression, "rabitq" for 32x compression)
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
/// ```
#[napi]
pub fn open(
    path: String,
    options: Option<OpenOptions>,
    #[napi(ts_arg_type = "((texts: string[]) => Float32Array[]) | undefined")]
    embedding_fn: Option<EmbeddingFn>,
) -> Result<VectorDatabase> {
    let opts = options.unwrap_or(OpenOptions {
        dimensions: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        quantization: None,
        metric: None,
        multi_vector: None,
    });

    let embedding_tsfn = embedding_fn.map(Arc::new);

    let dimensions = opts.dimensions.unwrap_or(128) as usize;
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);
    let ef_search = opts.ef_search.map(|v| v as usize);

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

    // Quantization only supports L2 distance
    if quant_mode.is_some() {
        let m = opts.metric.as_deref().unwrap_or("l2");
        if m != "l2" && m != "euclidean" {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Quantization only supports L2 distance, got metric='{m}'"),
            ));
        }
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
            embedding_fn: embedding_tsfn.clone(),
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
        let actual_dims = store.dimensions();

        // Conflict check: opening existing single-vector store with multiVector: true
        if is_multi_vector && !is_mv {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot open existing single-vector database with multiVector: true",
            ));
        }

        let resolved_dims = if actual_dims > 0 {
            actual_dims as u32
        } else {
            dimensions as u32
        };

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner { store })),
            path,
            dimensions: resolved_dims,
            is_persistent: true,
            is_multi_vector: is_mv,
            embedding_fn: embedding_tsfn.clone(),
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
            embedding_fn: embedding_tsfn.clone(),
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
        embedding_fn: embedding_tsfn,
        collections_cache: RwLock::new(HashMap::new()),
    })
}
