//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb::vector::{MetadataFilter, Vector, VectorStore};
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

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
    inner: RwLock<VectorDatabaseInner>,
    path: String,
    dimensions: u32,
    is_persistent: bool,
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

    /// Save database to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        let inner = self.inner.read();
        inner.store.save_to_disk(&self.path).map_err(convert_error)
    }

    /// Get number of vectors in database.
    #[napi(getter)]
    pub fn count(&self) -> u32 {
        let inner = self.inner.read();
        inner.store.len() as u32
    }

    /// Get current ef_search value.
    #[napi(getter, js_name = "efSearch")]
    pub fn get_ef_search(&self) -> Option<u32> {
        let inner = self.inner.read();
        inner.store.get_ef_search().map(|e| e as u32)
    }

    /// Set ef_search value.
    #[napi(setter, js_name = "efSearch")]
    pub fn set_ef_search(&self, ef_search: u32) {
        let mut inner = self.inner.write();
        inner.store.set_ef_search(ef_search as usize);
    }

    /// Get or create a named collection.
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

        Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,
            }),
            path: collection_path.to_string_lossy().to_string(),
            dimensions: self.dimensions,
            is_persistent: true,
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
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
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
        let collection_path = base_path.join("collections").join(&name);

        if !collection_path.exists() {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Collection '{}' does not exist", name),
            ));
        }

        std::fs::remove_dir_all(&collection_path).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to delete collection: {}", e),
            )
        })
    }

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
/// ```
#[napi]
pub fn open(path: String, options: Option<OpenOptions>) -> Result<VectorDatabase> {
    use omendb::vector::rabitq::RaBitQParams;

    let opts = options.unwrap_or(OpenOptions {
        dimensions: None,
        m: None,
        ef_construction: None,
        ef_search: None,
        quantization: None,
    });

    let dimensions = opts.dimensions.unwrap_or(128) as usize;
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);
    let ef_search = opts.ef_search.map(|v| v as usize);
    let quantization = opts.quantization;

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

    // Handle :memory: for in-memory database
    if path == ":memory:" {
        let mut store = if let Some(bits) = quantization {
            // Create quantized store
            let params = match bits {
                2 => RaBitQParams::bits2(),
                4 => RaBitQParams::bits4(),
                8 => RaBitQParams::bits8(),
                _ => unreachable!(), // Already validated above
            };
            VectorStore::new_with_quantization(dimensions, params)
        } else if m.is_some() || ef_construction.is_some() {
            let m_val = m.unwrap_or(16);
            let ef_con = ef_construction.unwrap_or(100);
            let ef_s = ef_search.unwrap_or(100.max(m_val * 4));
            VectorStore::new_with_params(dimensions, m_val, ef_con, ef_s).map_err(|e| {
                Error::new(
                    Status::InvalidArg,
                    format!("Failed to create HNSW index: {}", e),
                )
            })?
        } else {
            VectorStore::new_with_capacity(dimensions, 10_000)
        };

        // Apply ef_search if specified
        if let Some(ef_s) = ef_search {
            store.set_ef_search(ef_s);
        }

        return Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: true,
            }),
            path,
            dimensions: dimensions as u32,
            is_persistent: false,
        });
    }

    // Persistent storage
    let db_path = std::path::Path::new(&path);
    let mut store = if db_path.exists() {
        // Load existing database
        let loaded = if dimensions == 0 {
            VectorStore::open(&path).map_err(convert_error)?
        } else {
            VectorStore::open_with_dimensions(&path, dimensions).map_err(convert_error)?
        };

        // Cannot enable quantization on existing database with data
        if quantization.is_some() && loaded.len() > 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "Cannot enable quantization on existing database. Create a new database with quantization.",
            ));
        }
        loaded
    } else if let Some(bits) = quantization {
        // Create new quantized persistent store
        let params = match bits {
            2 => RaBitQParams::bits2(),
            4 => RaBitQParams::bits4(),
            8 => RaBitQParams::bits8(),
            _ => unreachable!(),
        };
        VectorStore::new_with_quantization(dimensions, params)
    } else {
        // Create new persistent store with default settings
        VectorStore::open_with_dimensions(&path, dimensions).map_err(convert_error)?
    };

    // Apply ef_search if specified
    if let Some(ef_s) = ef_search {
        store.set_ef_search(ef_s);
    }

    Ok(VectorDatabase {
        inner: RwLock::new(VectorDatabaseInner {
            store,
            index_to_id_cache: HashMap::new(),
            cache_valid: false,
        }),
        path,
        dimensions: dimensions as u32,
        is_persistent: true,
    })
}
