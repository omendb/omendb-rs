//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb::vector::{Vector, VectorStore};
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
    /// Embedding as array of numbers
    pub embedding: Vec<f64>,
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
    pub embedding: Vec<f64>,
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
    /// Accepts an array of items with id, embedding, and optional metadata.
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

                let embedding: Vec<f32> = item.embedding.into_iter().map(|x| x as f32).collect();
                (item.id, Vector::new(embedding), metadata)
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
    /// @returns Array of {id, distance, metadata}
    #[napi]
    pub fn search(
        &self,
        query: Either<Vec<f64>, Float32Array>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<SearchResult>> {
        let query_vec = Vector::new(extract_query_vector(query));
        let ef_usize = ef.map(|e| e as usize);

        // Fast path: read lock when cache is valid
        {
            let inner = self.inner.read();
            if inner.cache_valid && !inner.store.needs_index_rebuild() {
                let results = inner
                    .store
                    .search_with_ef_readonly(&query_vec, k as usize, None, ef_usize)
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
            .search_with_ef_readonly(&query_vec, k as usize, None, ef_usize)
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
        let all_results = inner
            .store
            .batch_search_parallel_with_metadata(&query_vecs, k as usize, ef.map(|e| e as usize));

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

        inner.store.get_by_id(&id).map(|(vector, metadata)| GetResult {
            id,
            embedding: vector.data.iter().map(|&x| x as f64).collect(),
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

    /// Update a vector's embedding and/or metadata.
    #[napi]
    pub fn update(
        &self,
        id: String,
        embedding: Either<Vec<f64>, Float32Array>,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] metadata: Option<JsonValue>,
    ) -> Result<()> {
        let vector = Some(Vector::new(extract_query_vector(embedding)));
        let mut inner = self.inner.write();
        inner
            .store
            .update(&id, vector, metadata)
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

#[napi(object)]
pub struct OpenOptions {
    pub dimensions: Option<u32>,
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
}

// ============================================================================
// Module-level open function
// ============================================================================

/// Open or create a vector database.
///
/// @param path - Database directory path (use ":memory:" for in-memory)
/// @param options - Optional configuration
/// @returns VectorDatabase instance
#[napi]
pub fn open(path: String, options: Option<OpenOptions>) -> Result<VectorDatabase> {
    let opts = options.unwrap_or(OpenOptions {
        dimensions: None,
        m: None,
        ef_construction: None,
    });

    let dimensions = opts.dimensions.unwrap_or(128) as usize;
    let m = opts.m.map(|v| v as usize);
    let ef_construction = opts.ef_construction.map(|v| v as usize);

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

    // Handle :memory: for in-memory database
    if path == ":memory:" {
        let store = if m.is_some() || ef_construction.is_some() {
            let m_val = m.unwrap_or(16);
            let ef_con = ef_construction.unwrap_or(100);
            let ef_search = 100.max(m_val * 4);
            VectorStore::new_with_params(dimensions, m_val, ef_con, ef_search).map_err(|e| {
                Error::new(
                    Status::InvalidArg,
                    format!("Failed to create HNSW index: {}", e),
                )
            })?
        } else {
            VectorStore::new_with_capacity(dimensions, 10_000)
        };

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
    let store = if dimensions == 0 {
        VectorStore::open(&path).map_err(convert_error)?
    } else {
        VectorStore::open_with_dimensions(&path, dimensions).map_err(convert_error)?
    };

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
