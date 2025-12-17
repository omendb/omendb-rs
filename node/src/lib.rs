//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb::vector::{MetadataFilter, RaBitQParams, Vector, VectorStore, VectorStoreOptions};
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

/// Convert Rust error to napi Error with appropriate status
fn convert_error(err: anyhow::Error) -> Error {
    let msg = err.to_string();
    if msg.contains("dimension") {
        Error::new(Status::InvalidArg, msg)
    } else if msg.contains("not found") || msg.contains("does not exist") {
        Error::new(Status::GenericFailure, msg)
    } else {
        Error::new(Status::GenericFailure, msg)
    }
}

/// Parse a numeric comparison operator ($gt, $gte, $lt, $lte)
fn parse_numeric_op(op: &str, key: &str, value: &JsonValue) -> Result<MetadataFilter> {
    let num = value.as_f64().ok_or_else(|| {
        Error::new(Status::InvalidArg, format!("{} requires a number", op))
    })?;
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
// Vector Item - input for set operations
// ============================================================================

#[napi(object)]
pub struct VectorItem {
    pub id: String,
    /// Vector data as array of numbers
    pub vector: Vec<f64>,
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
    pub vector: Vec<f64>,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

// ============================================================================
// Vector Item With Text - input for hybrid search set operations
// ============================================================================

#[napi(object)]
pub struct VectorItemWithText {
    pub id: String,
    pub vector: Vec<f64>,
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
// Inner State
// ============================================================================

struct VectorDatabaseInner {
    store: VectorStore,
    /// Cached reverse index (index -> id) for fast lookups during search
    index_to_id_cache: HashMap<usize, String>,
    /// Track if cache is valid (invalidated on set/delete)
    cache_valid: bool,
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
    /// Cache of open collection handles (same name = shared state)
    collections_cache: RwLock<HashMap<String, Arc<RwLock<VectorDatabaseInner>>>>,
}

#[napi]
impl VectorDatabase {
    /// Insert or update vectors.
    ///
    /// Accepts an array of items with id, vector, and optional metadata.
    #[napi]
    pub fn set(&self, items: Vec<VectorItem>) -> Result<Vec<u32>> {
        let batch: Vec<(String, Vector, JsonValue)> = items
            .into_iter()
            .map(|item| {
                let mut metadata = item.metadata.unwrap_or(serde_json::json!({}));

                // Handle document field
                if let Some(doc) = item.document {
                    if let Some(obj) = metadata.as_object_mut() {
                        obj.insert("document".to_string(), serde_json::json!(doc));
                    }
                }

                let vector_data: Vec<f32> = item.vector.into_iter().map(|x| x as f32).collect();
                (item.id, Vector::new(vector_data), metadata)
            })
            .collect();

        let mut inner = self.inner.write();
        let result = inner.store.set_batch(batch).map_err(convert_error)?;
        inner.cache_valid = false;

        Ok(result.into_iter().map(|x| x as u32).collect())
    }

    /// Search for k nearest neighbors.
    ///
    /// @param query - Query vector (number[] or Float32Array)
    /// @param k - Number of results to return
    /// @param ef - Optional search width override
    /// @param filter - Optional metadata filter (e.g., {category: "foo"} or {price: {$gt: 10}})
    /// @returns Array of {id, distance, metadata}
    #[napi]
    pub fn search(
        &self,
        query: Either<Vec<f64>, Float32Array>,
        k: u32,
        ef: Option<u32>,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] filter: Option<JsonValue>,
    ) -> Result<Vec<SearchResult>> {
        let query_vec = Vector::new(extract_query_vector(query));
        let ef_usize = ef.map(|e| e as usize);
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;

        // Fast path: read lock when cache is valid
        {
            let inner = self.inner.read();
            if inner.cache_valid && !inner.store.needs_index_rebuild() {
                let results = inner
                    .store
                    .search_with_ef_readonly(
                        &query_vec,
                        k as usize,
                        metadata_filter.as_ref(),
                        ef_usize,
                    )
                    .map_err(convert_error)?;

                return Ok(results
                    .into_iter()
                    .map(|(idx, dist, meta)| {
                        let id = inner
                            .index_to_id_cache
                            .get(&idx)
                            .cloned()
                            .unwrap_or_else(|| idx.to_string());
                        SearchResult {
                            id,
                            distance: dist as f64,
                            metadata: meta,
                        }
                    })
                    .collect());
            }
        }

        // Slow path: rebuild cache if needed
        let mut inner = self.inner.write();
        inner.store.ensure_index_ready().map_err(convert_error)?;

        if !inner.cache_valid {
            inner.index_to_id_cache = inner
                .store
                .id_to_index
                .iter()
                .map(|(id, &idx)| (idx, id.clone()))
                .collect();
            inner.cache_valid = true;
        }

        let results = inner
            .store
            .search_with_ef_readonly(&query_vec, k as usize, metadata_filter.as_ref(), ef_usize)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|(idx, dist, meta)| {
                let id = inner
                    .index_to_id_cache
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| idx.to_string());
                SearchResult {
                    id,
                    distance: dist as f64,
                    metadata: meta,
                }
            })
            .collect())
    }

    /// Batch search with parallel execution (async).
    ///
    /// Runs searches in parallel using rayon, returns Promise.
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<Either<Vec<f64>, Float32Array>>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let query_vecs: Vec<Vector> = queries
            .into_iter()
            .map(|q| Vector::new(extract_query_vector(q)))
            .collect();

        // Ensure index and cache are ready
        {
            let mut inner = self.inner.write();
            inner.store.ensure_index_ready().map_err(convert_error)?;
            if !inner.cache_valid {
                inner.index_to_id_cache = inner
                    .store
                    .id_to_index
                    .iter()
                    .map(|(id, &idx)| (idx, id.clone()))
                    .collect();
                inner.cache_valid = true;
            }
        }

        // Run parallel search
        let inner = self.inner.read();
        let all_results = inner.store.batch_search_parallel_with_metadata(
            &query_vecs,
            k as usize,
            ef.map(|e| e as usize),
        );

        // Convert results
        let mut output = Vec::with_capacity(all_results.len());
        for result in all_results {
            let results = result.map_err(convert_error)?;
            output.push(
                results
                    .into_iter()
                    .map(|(idx, dist, meta)| {
                        let id = inner
                            .index_to_id_cache
                            .get(&idx)
                            .cloned()
                            .unwrap_or_else(|| idx.to_string());
                        SearchResult {
                            id,
                            distance: dist as f64,
                            metadata: meta,
                        }
                    })
                    .collect(),
            );
        }

        Ok(output)
    }

    /// Get a vector by ID.
    #[napi]
    pub fn get(&self, id: String) -> Option<GetResult> {
        let inner = self.inner.read();

        inner.store.get_by_id(&id).map(|(vec, metadata)| GetResult {
            id,
            vector: vec.data.iter().map(|&x| x as f64).collect(),
            metadata: metadata.clone(),
        })
    }

    /// Delete vectors by ID.
    ///
    /// @returns Number of vectors deleted
    #[napi]
    pub fn delete(&self, ids: Vec<String>) -> Result<u32> {
        let mut inner = self.inner.write();
        let result = inner.store.delete_batch(&ids).map_err(convert_error)?;
        inner.cache_valid = false;
        Ok(result as u32)
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
            .map_err(convert_error)
    }

    /// Get number of vectors in database.
    #[napi(getter)]
    pub fn count(&self) -> u32 {
        let inner = self.inner.read();
        inner.store.len() as u32
    }

    /// Get current ef_search value.
    #[napi(getter, js_name = "efSearch")]
    pub fn get_ef_search(&self) -> u32 {
        let inner = self.inner.read();
        // VectorStore always returns Some now
        inner.store.get_ef_search().unwrap() as u32
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

        let inner = Arc::new(RwLock::new(VectorDatabaseInner {
            store,
            index_to_id_cache: HashMap::new(),
            cache_valid: false,
        }));

        // Cache and return
        cache.insert(name, Arc::clone(&inner));

        Ok(VectorDatabase {
            inner,
            path: collection_path.to_string_lossy().to_string(),
            dimensions: self.dimensions,
            is_persistent: true,
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
            let vector_data: Vec<f32> = item.vector.into_iter().map(|x| x as f32).collect();
            let metadata = item.metadata.unwrap_or(serde_json::json!({}));

            let index = inner
                .store
                .set_with_text(item.id, Vector::new(vector_data), &item.text, metadata)
                .map_err(convert_error)?;

            results.push(index as u32);
        }

        inner.cache_valid = false;
        Ok(results)
    }

    /// Search using text only (BM25 scoring).
    ///
    /// @param query - Text query
    /// @param k - Number of results
    /// @returns Array of {id, score, metadata}
    #[napi]
    pub fn text_search(&self, query: String, k: u32) -> Result<Vec<TextSearchResult>> {
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
    /// @returns Array of {id, score, metadata}
    #[napi]
    pub fn hybrid_search(
        &self,
        query_vector: Either<Vec<f64>, Float32Array>,
        query_text: String,
        k: u32,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] filter: Option<JsonValue>,
        alpha: Option<f64>,
        rrf_k: Option<u32>,
    ) -> Result<Vec<TextSearchResult>> {
        let query_vec = Vector::new(extract_query_vector(query_vector));
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let alpha_f32 = alpha.map(|a| a as f32);
        let rrf_k_usize = rrf_k.map(|k| k as usize);

        let mut inner = self.inner.write();

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
                .hybrid_search_with_rrf_k(&query_vec, &query_text, k as usize, alpha_f32, rrf_k_usize)
                .map_err(convert_error)?
        };

        Ok(results
            .into_iter()
            .map(|(id, score, metadata)| TextSearchResult {
                id,
                score: score as f64,
                metadata,
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
        inner.cache_valid = false;

        Ok(count as u32)
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
/// - quantization: null (RaBitQ bit width: 2, 4, or 8 for compression)
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
    /// RaBitQ quantization bits: 2, 4, or 8 (default: null = no quantization)
    /// Enables 4-16x memory compression with ~1-2% recall loss
    pub quantization: Option<u8>,
    /// Rescore candidates with exact distance (default: true when quantization enabled)
    /// Set to false for maximum speed at the cost of ~20% recall
    pub rescore: Option<bool>,
    /// Oversampling factor for rescoring (default: 3.0)
    /// Fetches k*oversample candidates then reranks to return top k
    pub oversample: Option<f64>,
    /// Distance metric: "l2"/"euclidean" (default), "cosine", "dot"/"ip"
    pub metric: Option<String>,
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
/// // With RaBitQ quantization (8x memory reduction)
/// const db = omendb.open("./mydb", {
///   dimensions: 128,
///   quantization: 4  // 4-bit quantization
/// });
///
/// // Quantization with custom rescore settings
/// const db = omendb.open("./mydb", {
///   dimensions: 128,
///   quantization: 4,
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
    });

    let dimensions = opts.dimensions.unwrap_or(128) as usize;
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);
    let ef_search = opts.ef_search.map(|v| v as usize);
    let quantization = opts.quantization;
    let rescore = opts.rescore;
    let oversample = opts.oversample;

    // Validate parameters
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

    if let Some(bits) = quantization {
        if !matches!(bits, 2 | 4 | 8) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("quantization must be 2, 4, or 8, got {}", bits),
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
                    format!("Unknown metric: '{}'. Valid: l2, euclidean, cosine, dot, ip", m),
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
    if let Some(bits) = quantization {
        let params = match bits {
            2 => RaBitQParams::bits2(),
            4 => RaBitQParams::bits4(),
            8 => RaBitQParams::bits8(),
            _ => unreachable!(),
        };
        store_options = store_options.quantization_rabitq_params(params);
    }
    if let Some(rescore_val) = rescore {
        store_options = store_options.rescore(rescore_val);
    }
    if let Some(oversample_val) = oversample {
        store_options = store_options.oversample(oversample_val as f32);
    }
    if let Some(ref metric_str) = opts.metric {
        store_options = store_options.metric(metric_str).map_err(|e| {
            Error::new(Status::InvalidArg, e)
        })?;
    }

    // Handle :memory: for in-memory database
    if path == ":memory:" {
        let store = store_options.build().map_err(|e| {
            Error::new(Status::GenericFailure, format!("Failed to create store: {}", e))
        })?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: true,
            })),
            path,
            dimensions: dimensions as u32,
            is_persistent: false,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Check if enabling quantization on existing non-empty database
    let db_path = std::path::Path::new(&path);
    if db_path.exists() && quantization.is_some() {
        let existing = VectorStore::open(&path).map_err(convert_error)?;
        if existing.len() > 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot enable quantization on existing database. Create a new database with quantization.",
            ));
        }
    }

    // Open persistent store with options
    let store = store_options.open(&path).map_err(convert_error)?;

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner {
            store,
            index_to_id_cache: HashMap::new(),
            cache_valid: false,
        })),
        path,
        dimensions: dimensions as u32,
        is_persistent: true,
        collections_cache: RwLock::new(HashMap::new()),
    })
}
