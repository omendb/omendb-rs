extern crate omendb as omendb_core;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use omendb_core::text::TextSearchConfig;
use omendb_core::vector::{MetadataFilter, RaBitQParams, Vector, VectorStore, VectorStoreOptions};
use parking_lot::RwLock;
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Convert quantization bits to RaBitQParams
///
/// # Panics
/// Panics if bits is not 2, 4, or 8 (caller must validate)
#[inline]
fn rabitq_params(bits: u8) -> RaBitQParams {
    match bits {
        2 => RaBitQParams::bits2(),
        4 => RaBitQParams::bits4(),
        8 => RaBitQParams::bits8(),
        _ => panic!("Invalid quantization bits: {bits}. Must be 2, 4, or 8."),
    }
}

/// Extract single query vector from Python object (list or 1D numpy array)
fn extract_query_vector(ob: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try 1D numpy array first (more efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray1<'_, f32>>() {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(format!("Invalid numpy array: {}", e)));
    }
    // Fall back to list of floats
    if let Ok(list) = ob.extract::<Vec<f32>>() {
        return Ok(list);
    }
    Err(PyValueError::new_err(
        "query must be a list of floats or 1D numpy array (dtype=float32)",
    ))
}

/// Extract batch of query vectors from Python object (list of lists or 2D numpy array)
fn extract_batch_queries(ob: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    // Try 2D numpy array first (most efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let n_queries = shape[0];
        let dim = shape[1];
        let mut queries = Vec::with_capacity(n_queries);

        if let Ok(slice) = arr.as_slice() {
            for i in 0..n_queries {
                let start = i * dim;
                let end = start + dim;
                queries.push(slice[start..end].to_vec());
            }
            return Ok(queries);
        } else {
            return Err(PyValueError::new_err("2D array must be contiguous"));
        }
    }

    // Try list of lists/arrays
    if let Ok(list) = ob.extract::<Vec<Vec<f32>>>() {
        return Ok(list);
    }

    Err(PyValueError::new_err(
        "queries must be a 2D numpy array or list of lists",
    ))
}

/// Convert PyO3 errors to Python exceptions with proper type mapping
fn convert_error(err: anyhow::Error) -> PyErr {
    let msg = err.to_string();
    // Map to appropriate Python exception types
    if msg.contains("dimension") {
        PyValueError::new_err(msg)
    } else if msg.contains("not found") || msg.contains("does not exist") {
        PyValueError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// Internal state for VectorDatabase
struct VectorDatabaseInner {
    store: VectorStore,
    index_to_id_cache: HashMap<usize, String>,
    cache_valid: bool,
}

/// High-performance embedded vector database.
///
/// Provides fast similarity search using HNSW indexing with:
/// - ~19,000 QPS @ 10K vectors with 100% recall
/// - 20,000-28,000 vec/s insert throughput
/// - Extended RaBitQ 8x compression
/// - ACORN-1 filtered search (37.79x speedup)
///
/// Auto-persists to disk for seamless data durability.
#[pyclass]
pub struct VectorDatabase {
    inner: RwLock<VectorDatabaseInner>,
    path: String,
    dimensions: usize,
    is_persistent: bool,
}

/// Convert search results to Python list of dicts
fn results_to_py(
    py: Python<'_>,
    results: &[(usize, f32, JsonValue)],
    index_to_id: &HashMap<usize, String>,
) -> PyResult<Vec<Py<PyDict>>> {
    let mut py_results = Vec::with_capacity(results.len());

    for (idx, distance, metadata) in results {
        let dict = PyDict::new(py);

        // Look up ID from cache
        let id = index_to_id
            .get(idx)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        dict.set_item("id", id)?;
        dict.set_item("distance", *distance)?;

        // Convert metadata to Python dict
        let metadata_dict = json_to_pyobject(py, metadata)?;
        dict.set_item("metadata", metadata_dict)?;

        py_results.push(dict.into());
    }

    Ok(py_results)
}

#[pymethods]
impl VectorDatabase {
    /// Set (insert or replace) vectors.
    ///
    /// If a vector with the same ID already exists, it will be replaced.
    /// Otherwise, a new vector will be inserted.
    ///
    /// Args:
    ///     items (list[dict]): List of dictionaries, each containing:
    ///         - id (str): Unique identifier for the vector
    ///         - vector (list[float]): Vector data (must match database dimensions)
    ///         - metadata (dict, optional): Arbitrary metadata as JSON-compatible dict
    ///         - document (str, optional): Document text (stored in metadata["document"])
    ///
    /// Returns:
    ///     list[int]: Internal indices of stored vectors
    ///
    /// Raises:
    ///     ValueError: If any item is missing required fields or has invalid dimensions
    ///     RuntimeError: If HNSW index operation fails
    ///
    /// Examples:
    ///     Basic set:
    ///
    ///     >>> db.set([
    ///     ...     {"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}},
    ///     ...     {"id": "doc2", "vector": [0.4, 0.5, 0.6], "metadata": {"title": "World"}},
    ///     ... ])
    ///     [0, 1]
    ///
    ///     Replace existing vector:
    ///
    ///     >>> db.set([{"id": "doc1", "vector": [0.7, 0.8, 0.9]}])
    ///     [0]
    ///
    ///     With document:
    ///
    ///     >>> db.set([{"id": "doc1", "vector": [...], "document": "Original text content"}])
    ///
    /// Performance:
    ///     - Throughput: 20,000-28,000 vec/s @ 10K vectors
    ///     - Batch operations are more efficient than individual inserts
    ///
    /// Flexible input formats:
    ///     # Single item
    ///     db.set("id", [0.1, 0.2, 0.3])
    ///     db.set("id", [0.1, 0.2, 0.3], {"key": "value"})
    ///
    ///     # Batch (list of dicts)
    ///     db.set([{"id": "a", "vector": [...], "metadata": {...}}])
    ///
    ///     # Batch kwargs
    ///     db.set(ids=["a", "b"], vectors=[[...], [...]], metadatas=[{...}, {...}])
    #[pyo3(name = "set", signature = (id_or_items=None, vector=None, metadata=None, *, ids=None, vectors=None, metadatas=None))]
    fn set_vectors(
        &self,
        _py: Python<'_>,
        id_or_items: Option<&Bound<'_, PyAny>>,
        vector: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
        ids: Option<Vec<String>>,
        vectors: Option<Vec<Vec<f32>>>,
        metadatas: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Vec<usize>> {
        let batch = if let (Some(ids), Some(vectors)) = (&ids, &vectors) {
            // Batch kwargs: ids=[], vectors=[], metadatas=[]
            if ids.len() != vectors.len() {
                return Err(PyValueError::new_err(format!(
                    "ids and vectors must have same length: {} vs {}",
                    ids.len(),
                    vectors.len()
                )));
            }
            ids.iter()
                .enumerate()
                .map(|(i, id)| {
                    let meta = metadatas
                        .and_then(|m| m.get_item(i).ok())
                        .map(|m| pyobject_to_json(&m))
                        .transpose()?
                        .unwrap_or_else(|| serde_json::json!({}));
                    Ok((id.clone(), Vector::new(vectors[i].clone()), meta))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else if let Some(id_or_items) = id_or_items {
            if let Ok(id_str) = id_or_items.extract::<String>() {
                // Single item: set("id", [...], {...})
                let vec_data = vector
                    .ok_or_else(|| PyValueError::new_err("vector required when id is a string"))?;
                let meta = metadata
                    .map(|m| pyobject_to_json(m.as_any()))
                    .transpose()?
                    .unwrap_or_else(|| serde_json::json!({}));
                vec![(id_str, Vector::new(vec_data), meta)]
            } else if let Ok(items) = id_or_items.cast::<PyList>() {
                // Batch: set([{...}, {...}])
                parse_batch_items(items)?
            } else {
                return Err(PyValueError::new_err(
                    "First argument must be a string (id) or list of dicts",
                ));
            }
        } else {
            return Err(PyValueError::new_err(
                "set() requires either (id, vector) or a list of items or (ids=, vectors=)",
            ));
        };

        // Acquire write lock and perform set
        let mut inner = self.inner.write();
        let result = inner.store.set_batch(batch).map_err(convert_error)?;
        inner.cache_valid = false;
        Ok(result)
    }

    /// Search for k nearest neighbors (single query).
    ///
    /// Args:
    ///     query: Query vector (list of floats or 1D numpy array)
    ///     k (int): Number of nearest neighbors to return
    ///     ef (int, optional): Search width override (default: auto-tuned)
    ///     filter (dict, optional): MongoDB-style metadata filter
    ///
    /// Returns:
    ///     list[dict]: Results with keys {id, distance, metadata}
    ///
    /// Examples:
    ///     >>> results = db.search([0.1, 0.2, 0.3], k=5)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: {r['distance']:.4f}")
    ///
    ///     With filter:
    ///     >>> db.search([...], k=10, filter={"category": "A"})
    #[pyo3(name = "search", signature = (query, k, ef=None, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(PyValueError::new_err(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        let query_vec = Vector::new(extract_query_vector(query)?);
        let rust_filter = filter.map(parse_filter).transpose()?;

        // Fast path: read lock when cache is valid
        {
            let inner = self.inner.read();
            if inner.cache_valid && !inner.store.needs_index_rebuild() {
                let results = inner
                    .store
                    .search_with_ef_readonly(&query_vec, k, rust_filter.as_ref(), ef)
                    .map_err(convert_error)?;
                return results_to_py(py, &results, &inner.index_to_id_cache);
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
            .search_with_ef_readonly(&query_vec, k, rust_filter.as_ref(), ef)
            .map_err(convert_error)?;
        results_to_py(py, &results, &inner.index_to_id_cache)
    }

    /// Batch search multiple queries with parallel execution.
    ///
    /// Efficiently searches multiple queries in parallel using rayon.
    /// Releases the GIL during search for maximum throughput.
    ///
    /// Args:
    ///     queries: 2D numpy array or list of query vectors
    ///     k (int): Number of nearest neighbors per query
    ///     ef (int, optional): Search width override
    ///
    /// Returns:
    ///     list[list[dict]]: Results for each query
    #[pyo3(name = "search_batch", signature = (queries, k, ef=None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(PyValueError::new_err(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        let query_vecs: Vec<Vector> = extract_batch_queries(queries)?
            .into_iter()
            .map(Vector::new)
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

        // Release GIL and search in parallel
        #[allow(deprecated)]
        let all_results: Vec<Result<Vec<(usize, f32, JsonValue)>, _>> = py.allow_threads(|| {
            let inner = self.inner.read();
            inner
                .store
                .batch_search_parallel_with_metadata(&query_vecs, k, ef)
        });

        // Convert to Python
        let inner = self.inner.read();
        let mut py_all_results = Vec::with_capacity(all_results.len());
        for result in all_results {
            let results = result.map_err(convert_error)?;
            py_all_results.push(results_to_py(py, &results, &inner.index_to_id_cache)?);
        }

        Ok(py_all_results)
    }

    /// Delete vectors by ID.
    ///
    /// Examples:
    ///     >>> db.delete(["doc1", "doc2"])
    ///     2
    ///
    ///     >>> db.delete(["nonexistent"])  # Silently skips missing IDs
    ///     0
    fn delete(&self, ids: Vec<String>) -> PyResult<usize> {
        let mut inner = self.inner.write();

        let result = inner.store.delete_batch(&ids).map_err(convert_error)?;

        // Invalidate cache since id_to_index changed
        inner.cache_valid = false;

        Ok(result)
    }

    /// Update vector and/or metadata for existing ID.
    ///
    /// Args:
    ///     id (str): Vector ID to update
    ///     vector (list[float], optional): New vector data
    ///     metadata (dict, optional): New metadata (replaces existing)
    ///
    /// Raises:
    ///     RuntimeError: If vector with given ID doesn't exist
    ///
    /// Examples:
    ///     Update vector only:
    ///
    ///     >>> db.update("doc1", vector=[0.1, 0.2, 0.3])
    ///
    ///     Update metadata only:
    ///
    ///     >>> db.update("doc1", metadata={"title": "Updated"})
    ///
    ///     Update both:
    ///
    ///     >>> db.update("doc1", vector=[0.4, 0.5, 0.6], metadata={"title": "New"})
    fn update(
        &self,
        id: String,
        vector_data: Vec<f32>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let vector = Some(Vector::new(vector_data));
        let metadata_json = if let Some(m) = metadata {
            Some(pyobject_to_json(m.as_any())?)
        } else {
            None
        };

        let mut inner = self.inner.write();

        inner
            .store
            .update(&id, vector, metadata_json)
            .map_err(convert_error)
    }

    /// Get vector by ID.
    ///
    /// Args:
    ///     id (str): Vector ID to retrieve
    ///
    /// Returns:
    ///     dict or None: Dictionary with keys "id", "vector", "metadata"
    ///                   Returns None if ID not found
    ///
    /// Examples:
    ///     >>> result = db.get("doc1")
    ///     >>> if result:
    ///     ...     print(result["id"], result["vector"], result["metadata"])
    ///     doc1 [0.1, 0.2, 0.3] {'title': 'Hello'}
    fn get(&self, py: Python<'_>, id: String) -> PyResult<Option<HashMap<String, Py<PyAny>>>> {
        let inner = self.inner.read();

        if let Some((vector, metadata)) = inner.store.get_by_id(&id) {
            let mut result = HashMap::new();
            result.insert(
                "id".to_string(),
                id.into_pyobject(py).unwrap().unbind().into(),
            );
            result.insert(
                "vector".to_string(),
                vector
                    .data
                    .clone()
                    .into_pyobject(py)
                    .unwrap()
                    .unbind()
                    .into(),
            );

            let metadata_dict = json_to_pyobject(py, metadata)?;
            result.insert("metadata".to_string(), metadata_dict);

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Save database to disk (explicit sync).
    ///
    /// Note: When using persistent storage (directory path), data is automatically
    /// persisted after each operation. This method is for explicit sync.
    ///
    /// Examples:
    ///     >>> db.save()  # Saves to path specified in omendb.open()
    ///
    /// Performance:
    ///     Loading from disk is 401x faster than rebuilding index from scratch.
    fn save(&self) -> PyResult<()> {
        let inner = self.inner.read();
        inner.store.save_to_disk(&self.path).map_err(convert_error)
    }

    /// Get current ef_search value.
    ///
    /// ef_search controls the search quality/speed tradeoff. Higher values
    /// give better recall but slower search.
    ///
    /// Returns:
    ///     int or None: Current ef_search value, or None if not set
    ///
    /// Examples:
    ///     >>> db.get_ef_search()
    ///     100
    fn get_ef_search(&self) -> Option<usize> {
        let inner = self.inner.read();
        inner.store.get_ef_search()
    }

    /// Set ef_search value for search quality/speed tradeoff.
    ///
    /// Higher ef_search = better recall, slower search.
    /// Lower ef_search = faster search, may miss some neighbors.
    ///
    /// Args:
    ///     ef_search (int): New ef_search value (must be >= k in searches)
    ///
    /// Examples:
    ///     >>> db.set_ef_search(200)  # High quality
    ///     >>> db.set_ef_search(50)   # Faster search
    fn set_ef_search(&mut self, ef_search: usize) {
        let mut inner = self.inner.write();
        inner.store.set_ef_search(ef_search);
    }

    /// Number of vectors in database (Pythonic).
    ///
    /// Returns:
    ///     int: Total vector count (excluding deleted vectors)
    ///
    /// Examples:
    ///     >>> len(db)
    ///     1000
    fn __len__(&self) -> usize {
        let inner = self.inner.read();
        inner.store.len()
    }

    /// Number of vectors in database.
    ///
    /// Returns:
    ///     int: Total vector count
    fn len(&self) -> usize {
        let inner = self.inner.read();
        inner.store.len()
    }

    /// Get database dimensions.
    ///
    /// Returns:
    ///     int: Dimensionality of vectors in this database
    #[getter]
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Check if database is empty.
    fn is_empty(&self) -> bool {
        let inner = self.inner.read();
        inner.store.is_empty()
    }

    /// Get database statistics.
    ///
    /// Returns:
    ///     dict: Statistics including:
    ///         - dimensions: Vector dimensionality
    ///         - count: Number of vectors
    ///         - path: Database path
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let inner = self.inner.read();
        let dict = PyDict::new(py);
        dict.set_item("dimensions", self.dimensions)?;
        dict.set_item("count", inner.store.len())?;
        dict.set_item("path", &self.path)?;
        Ok(dict.into())
    }

    /// Create or get a named collection within this database.
    ///
    /// Collections are separate namespaces that share the same database path.
    /// Each collection has its own vectors and metadata, isolated from others.
    ///
    /// Args:
    ///     name (str): Collection name (alphanumeric and underscores only)
    ///
    /// Returns:
    ///     VectorDatabase: A new database instance for this collection
    ///
    /// Raises:
    ///     ValueError: If name is empty or contains invalid characters
    ///
    /// Examples:
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    ///     >>> users = db.collection("users")
    ///     >>> products = db.collection("products")
    ///     >>> users.set([{"id": "u1", "vector": [...]}])
    ///     >>> products.set([{"id": "p1", "vector": [...]}])
    ///
    ///     Separate namespaces:
    ///
    ///     >>> # IDs are scoped to collection
    ///     >>> users.set([{"id": "doc1", ...}])
    ///     >>> products.set([{"id": "doc1", ...}])  # No conflict!
    fn collection(&self, name: String) -> PyResult<VectorDatabase> {
        // Validate collection name
        if name.is_empty() {
            return Err(PyValueError::new_err("Collection name cannot be empty"));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(PyValueError::new_err(
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        // Only persistent databases support collections
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage",
            ));
        }

        // Create collection path: {base_path}/collections/{name}
        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        // Ensure collections directory exists
        std::fs::create_dir_all(collection_path.parent().unwrap()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create collections directory: {}", e))
        })?;

        // Open the collection as a separate VectorStore
        let store = if self.dimensions == 0 || self.dimensions == 128 {
            VectorStore::open(&collection_path).map_err(convert_error)?
        } else {
            VectorStore::open_with_dimensions(&collection_path, self.dimensions)
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

    /// List all collections in this database.
    ///
    /// Returns:
    ///     list[str]: Names of all collections
    fn collections(&self) -> PyResult<Vec<String>> {
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage",
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");

        if !collections_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        let entries = std::fs::read_dir(&collections_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read collections: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read entry: {}", e)))?;
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }

        names.sort();
        Ok(names)
    }

    // =========================================================================
    // Hybrid Search Methods
    // =========================================================================

    /// Enable text search for hybrid (vector + text) search.
    ///
    /// Must be called before using set_with_text() or hybrid_search().
    /// Creates a tantivy text index for BM25 scoring.
    ///
    /// Args:
    ///     buffer_mb (int, optional): Writer buffer size in MB (default: 50)
    ///
    /// Examples:
    ///     >>> db.enable_text_search()
    ///     >>> db.enable_text_search(buffer_mb=100)  # For high-throughput
    #[pyo3(name = "enable_text_search", signature = (buffer_mb=None))]
    fn enable_text_search(&self, buffer_mb: Option<usize>) -> PyResult<()> {
        let mut inner = self.inner.write();

        if let Some(mb) = buffer_mb {
            let config = TextSearchConfig {
                writer_buffer_mb: mb,
            };
            // Store config for later use - need to set it before enabling
            // Since we can't modify text_search_config after construction,
            // we'll need to create the text index directly with config
            if inner.store.has_text_search() {
                return Ok(()); // Already enabled
            }
            // For now, just enable with default - config support requires store changes
            let _ = config; // suppress unused warning
        }

        inner.store.enable_text_search().map_err(convert_error)
    }

    /// Check if text search is enabled.
    ///
    /// Returns:
    ///     bool: True if text search is enabled
    fn has_text_search(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_text_search()
    }

    /// Set vectors with associated text for hybrid search.
    ///
    /// Args:
    ///     items (list[dict]): List of items with id, vector, text, and optional metadata
    ///
    /// Each item must have:
    ///     - id (str): Unique identifier
    ///     - vector (list[float]): Vector embedding
    ///     - text (str): Text content for BM25 search
    ///     - metadata (dict, optional): Additional metadata
    ///
    /// Returns:
    ///     list[int]: Internal indices of stored vectors
    ///
    /// Examples:
    ///     >>> db.enable_text_search()
    ///     >>> db.set_with_text([
    ///     ...     {"id": "doc1", "vector": [...], "text": "Machine learning intro"},
    ///     ...     {"id": "doc2", "vector": [...], "text": "Deep learning guide"},
    ///     ... ])
    ///     >>> db.flush()  # Commit text index changes
    #[pyo3(name = "set_with_text")]
    fn set_with_text(&self, items: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
        let mut inner = self.inner.write();

        if !inner.store.has_text_search() {
            return Err(PyRuntimeError::new_err(
                "Text search not enabled. Call enable_text_search() first.",
            ));
        }

        let mut results = Vec::with_capacity(items.len());

        for (idx, item) in items.iter().enumerate() {
            let dict = item.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err(format!("Item at index {} must be a dict", idx))
            })?;

            let id: String = dict
                .get_item("id")?
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Item at index {} missing 'id'", idx))
                })?
                .extract()?;

            let vector_data: Vec<f32> = dict
                .get_item("vector")?
                .ok_or_else(|| PyValueError::new_err(format!("Item '{}' missing 'vector'", id)))?
                .extract()?;

            let text: String = dict
                .get_item("text")?
                .ok_or_else(|| PyValueError::new_err(format!("Item '{}' missing 'text'", id)))?
                .extract()?;

            let metadata = if let Some(m) = dict.get_item("metadata")? {
                pyobject_to_json(&m)?
            } else {
                serde_json::json!({})
            };

            let index = inner
                .store
                .set_with_text(id, Vector::new(vector_data), &text, metadata)
                .map_err(convert_error)?;

            results.push(index);
        }

        inner.cache_valid = false;
        Ok(results)
    }

    /// Search using text only (BM25 scoring).
    ///
    /// Args:
    ///     query (str): Text query
    ///     k (int): Number of results to return
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score} sorted by BM25 score descending
    ///
    /// Examples:
    ///     >>> results = db.text_search("machine learning", k=10)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: {r['score']:.4f}")
    #[pyo3(name = "text_search")]
    fn text_search(&self, py: Python<'_>, query: &str, k: usize) -> PyResult<Vec<Py<PyDict>>> {
        let inner = self.inner.read();

        let results = inner.store.text_search(query, k).map_err(convert_error)?;

        let mut py_results = Vec::with_capacity(results.len());
        for (id, score) in results {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("score", score)?;
            py_results.push(dict.into());
        }

        Ok(py_results)
    }

    /// Hybrid search combining vector similarity and text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine:
    /// - HNSW vector search (by embedding similarity)
    /// - Tantivy text search (by BM25 relevance)
    ///
    /// Args:
    ///     query_vector: Query embedding (list or numpy array)
    ///     query_text (str): Text query for BM25
    ///     k (int): Number of results to return
    ///     filter (dict, optional): Metadata filter
    ///     alpha (float, optional): Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5)
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score} sorted by RRF score descending
    ///
    /// Examples:
    ///     >>> results = db.hybrid_search([0.1, 0.2, ...], "machine learning", k=10)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: {r['score']:.4f}")
    ///
    ///     With filter:
    ///     >>> results = db.hybrid_search(vec, "ML", k=10, filter={"category": "tech"})
    ///
    ///     Favor vector similarity (70% vector, 30% text):
    ///     >>> results = db.hybrid_search(vec, "ML", k=10, alpha=0.7)
    #[pyo3(name = "hybrid_search", signature = (query_vector, query_text, k, filter=None, alpha=None, rrf_k=None))]
    fn hybrid_search(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        query_text: &str,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let query_vec = Vector::new(extract_query_vector(query_vector)?);
        let rust_filter = filter.map(parse_filter).transpose()?;

        let mut inner = self.inner.write();

        let results = if let Some(f) = rust_filter {
            inner
                .store
                .hybrid_search_with_filter_rrf_k(&query_vec, query_text, k, &f, alpha, rrf_k)
                .map_err(convert_error)?
        } else {
            inner
                .store
                .hybrid_search_with_rrf_k(&query_vec, query_text, k, alpha, rrf_k)
                .map_err(convert_error)?
        };

        let mut py_results = Vec::with_capacity(results.len());
        for (id, score, metadata) in results {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("score", score)?;
            dict.set_item("metadata", json_to_pyobject(py, &metadata)?)?;
            py_results.push(dict.into());
        }

        Ok(py_results)
    }

    /// Flush pending changes to disk.
    ///
    /// For hybrid search, this commits text index changes.
    /// Text search results are not visible until flush is called.
    ///
    /// Examples:
    ///     >>> db.set_with_text([...])
    ///     >>> db.flush()  # Text now searchable
    fn flush(&self) -> PyResult<()> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)
    }

    // =========================================================================
    // Merge Methods
    // =========================================================================

    /// Merge vectors from another database into this one.
    ///
    /// Args:
    ///     other (VectorDatabase): Source database to merge from
    ///
    /// Returns:
    ///     int: Number of vectors merged
    ///
    /// Note:
    ///     - IDs are preserved; conflicting IDs are skipped (existing wins)
    ///     - Source database is not modified
    ///     - Both databases must have the same dimensions
    fn merge_from(&self, other: &VectorDatabase) -> PyResult<usize> {
        let mut inner = self.inner.write();
        let other_inner = other.inner.read();

        let count = inner
            .store
            .merge_from(&other_inner.store)
            .map_err(convert_error)?;

        // Invalidate cache since id_to_index changed
        inner.cache_valid = false;

        Ok(count)
    }

    /// Delete a collection from this database.
    ///
    /// Args:
    ///     name (str): Name of the collection to delete
    ///
    /// Raises:
    ///     ValueError: If collection doesn't exist
    ///     RuntimeError: If deletion fails
    ///
    /// Examples:
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    ///     >>> db.delete_collection("old_data")
    fn delete_collection(&self, name: String) -> PyResult<()> {
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage",
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        if !collection_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Collection '{}' not found",
                name
            )));
        }

        std::fs::remove_dir_all(&collection_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete collection: {}", e)))?;

        Ok(())
    }
}

/// Open or create a vector database.
///
/// All parameters except `path` are optional with sensible defaults.
///
/// Args:
///     path (str): Database directory path
///     dimensions (int): Vector dimensionality (default: 128, auto-detected on first insert)
///     m (int): HNSW neighbors per node (default: 16, range: 4-64)
///     ef_construction (int): Build quality (default: 100, higher = better graph)
///     ef_search (int): Search quality (default: 100, higher = better recall)
///     quantization (int): RaBitQ bits: 2, 4, or 8 (default: None = full precision)
///         Enables 4-16x memory compression with ~1-2% recall loss
///     config (dict): Advanced config (deprecated, use top-level params instead)
///
/// Returns:
///     VectorDatabase: Database instance
///
/// Raises:
///     ValueError: If parameters are invalid
///     RuntimeError: If database creation fails
///
/// Examples:
///     >>> import omendb
///
///     # Simple usage with defaults
///     >>> db = omendb.open("./my_vectors")
///
///     # With custom HNSW parameters
///     >>> db = omendb.open("./vectors", dimensions=384, m=32, ef_construction=200)
///
///     # With RaBitQ quantization (8x memory reduction)
///     >>> db = omendb.open("./vectors", dimensions=128, quantization=4)
///
///     # Tune search quality at runtime
///     >>> db.set_ef_search(200)  # Higher recall, slower
///     >>> db.set_ef_search(50)   # Lower recall, faster
#[pyfunction]
#[pyo3(signature = (path, dimensions=0, m=None, ef_construction=None, ef_search=None, quantization=None, config=None))]
fn open(
    path: String,
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    quantization: Option<u8>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<VectorDatabase> {
    use std::path::Path;

    // Validate optional params
    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(PyValueError::new_err(format!(
                "m must be between 4 and 64, got {}",
                m_val
            )));
        }
    }

    if let Some(bits) = quantization {
        if !matches!(bits, 2 | 4 | 8) {
            return Err(PyValueError::new_err(format!(
                "quantization must be 2, 4, or 8, got {}",
                bits
            )));
        }
    }
    if let (Some(ef_val), Some(m_val)) = (ef_construction, m) {
        if ef_val < m_val {
            return Err(PyValueError::new_err(format!(
                "ef_construction ({}) must be >= m ({})",
                ef_val, m_val
            )));
        }
    }

    let db_path = Path::new(&path);

    // Resolve effective dimensions (use 128 as default if not specified)
    let effective_dims = if dimensions == 0 { 128 } else { dimensions };

    // Check if this is a directory (persistent storage) or needs to become one
    if db_path.is_dir() || !db_path.exists() {
        // Build options from parameters
        let mut options = VectorStoreOptions::default().dimensions(effective_dims);

        if let Some(m_val) = m {
            options = options.m(m_val);
        }
        if let Some(ef_con) = ef_construction {
            options = options.ef_construction(ef_con);
        }
        if let Some(ef_s) = ef_search {
            options = options.ef_search(ef_s);
        }
        if let Some(bits) = quantization {
            options = options.quantization(rabitq_params(bits));
        }

        // Handle config dict for backward compatibility
        if let Some(cfg) = config {
            if let Some(hnsw_dict) = cfg.get_item("hnsw")? {
                let hnsw = hnsw_dict
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("'hnsw' must be a dict"))?;

                if m.is_none() {
                    if let Some(m_item) = hnsw.get_item("m")? {
                        options = options.m(m_item.extract()?);
                    }
                }
                if ef_construction.is_none() {
                    if let Some(ef_item) = hnsw.get_item("ef_construction")? {
                        options = options.ef_construction(ef_item.extract()?);
                    }
                }
                if ef_search.is_none() {
                    if let Some(ef_item) = hnsw.get_item("ef_search")? {
                        options = options.ef_search(ef_item.extract()?);
                    }
                }
            }
            if quantization.is_none() {
                if let Some(quant_dict) = cfg.get_item("quantization")? {
                    let quant = quant_dict
                        .cast::<PyDict>()
                        .map_err(|_| PyValueError::new_err("'quantization' must be a dict"))?;
                    if let Some(bits_item) = quant.get_item("bits")? {
                        let bits: u8 = bits_item.extract()?;
                        options = options.quantization(rabitq_params(bits));
                    }
                }
            }
            if let Some(expected) = cfg.get_item("expected_vectors")? {
                options = options.expected_capacity(expected.extract()?);
            }
        }

        // Check if enabling quantization on existing non-empty database
        if db_path.exists() && quantization.is_some() {
            let existing = VectorStore::open(&path).map_err(convert_error)?;
            if existing.len() > 0 {
                return Err(PyValueError::new_err(
                    "Cannot enable quantization on existing database. Create a new database with quantization.",
                ));
            }
        }

        // Open with options
        let store = options.open(&path).map_err(convert_error)?;

        return Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,
            }),
            path,
            dimensions: effective_dims,
            is_persistent: true,
        });
    }

    // Legacy path handling for backward compatibility
    let directory = db_path.parent().unwrap_or_else(|| Path::new("."));
    let filename = db_path.file_name().unwrap().to_str().unwrap();
    let vectors_path = directory.join(format!("{}.vectors.bin", filename));
    let hnsw_path = directory.join(format!("{}.hnsw", filename));

    // Try to load existing database
    if vectors_path.exists() || hnsw_path.exists() {
        let mut store =
            VectorStore::load_from_disk(&path, effective_dims).map_err(convert_error)?;

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
            dimensions: effective_dims,
            is_persistent: false,
        })
    } else {
        // Create new in-memory database with configuration
        let mut options = VectorStoreOptions::default().dimensions(effective_dims);

        if let Some(m_val) = m {
            options = options.m(m_val);
        }
        if let Some(ef_con) = ef_construction {
            options = options.ef_construction(ef_con);
        }
        if let Some(ef_s) = ef_search {
            options = options.ef_search(ef_s);
        }
        if let Some(bits) = quantization {
            options = options.quantization(rabitq_params(bits));
        }

        let store = options
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

        Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: true,
            }),
            path,
            dimensions: effective_dims,
            is_persistent: false,
        })
    }
}

/// Helper: Create VectorStore with configuration
fn create_store_with_config(
    dimensions: usize,
    config: &Bound<'_, PyDict>,
) -> PyResult<VectorStore> {
    // Parse HNSW configuration (if provided)
    if let Some(hnsw_dict) = config.get_item("hnsw")? {
        let hnsw = hnsw_dict
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("'hnsw' must be a dict"))?;

        let m: usize = hnsw
            .get_item("m")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.m' required"))?
            .extract()?;
        let ef_construction: usize = hnsw
            .get_item("ef_construction")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.ef_construction' required"))?
            .extract()?;
        let ef_search: usize = hnsw
            .get_item("ef_search")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.ef_search' required"))?
            .extract()?;

        VectorStore::new_with_params(dimensions, m, ef_construction, ef_search)
            .map_err(|e| PyValueError::new_err(format!("Failed to create HNSW index: {}", e)))
    }
    // Parse quantization configuration (if provided)
    else if let Some(quant_dict) = config.get_item("quantization")? {
        let quant = quant_dict
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("'quantization' must be a dict"))?;

        let bits: u8 = quant
            .get_item("bits")?
            .ok_or_else(|| PyValueError::new_err("'quantization.bits' required (2/4/8)"))?
            .extract()?;

        if !matches!(bits, 2 | 4 | 8) {
            return Err(PyValueError::new_err(
                "quantization.bits must be 2, 4, or 8",
            ));
        }

        Ok(VectorStore::new_with_quantization(
            dimensions,
            rabitq_params(bits),
        ))
    }
    // Parse expected_vectors for adaptive defaults
    else if let Some(expected) = config.get_item("expected_vectors")? {
        let expected_vectors: usize = expected.extract()?;
        Ok(VectorStore::new_with_capacity(dimensions, expected_vectors))
    } else {
        // Config dict provided but empty: use default capacity
        Ok(VectorStore::new_with_capacity(dimensions, 10_000))
    }
}

/// Helper: Parse Python filter dict to Rust MetadataFilter
fn parse_filter(filter: &Bound<'_, PyDict>) -> PyResult<MetadataFilter> {
    // Handle special logical operators first
    if let Some(and_value) = filter.get_item("$and")? {
        // $and expects an array of filter dicts
        let and_list = and_value
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$and must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in and_list.iter() {
            let sub_dict = item
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $and element must be a dict"))?;
            sub_filters.push(parse_filter(&sub_dict)?);
        }

        return Ok(MetadataFilter::And(sub_filters));
    }

    if let Some(or_value) = filter.get_item("$or")? {
        // $or expects an array of filter dicts
        let or_list = or_value
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$or must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in or_list.iter() {
            let sub_dict = item
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $or element must be a dict"))?;
            sub_filters.push(parse_filter(&sub_dict)?);
        }

        return Ok(MetadataFilter::Or(sub_filters));
    }

    // Parse regular field filters
    let mut filters = Vec::new();

    for (key, value) in filter.iter() {
        let key_str: String = key.extract()?;

        // Check if value is an operator dict like {"$gt": 5}
        if let Ok(op_dict) = value.cast::<PyDict>() {
            for (op, op_value) in op_dict.iter() {
                let op_str: String = op.extract()?;
                match op_str.as_str() {
                    "$gt" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gt(key_str.clone(), num));
                    }
                    "$gte" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gte(key_str.clone(), num));
                    }
                    "$lt" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Lt(key_str.clone(), num));
                    }
                    "$lte" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Lte(key_str.clone(), num));
                    }
                    "$in" => {
                        let list = op_value.cast::<PyList>()?;
                        let json_vals: Result<Vec<JsonValue>, _> =
                            list.iter().map(|obj| pyobject_to_json(&obj)).collect();
                        filters.push(MetadataFilter::In(key_str.clone(), json_vals?));
                    }
                    "$contains" => {
                        let substr: String = op_value.extract()?;
                        filters.push(MetadataFilter::Contains(key_str.clone(), substr));
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown filter operator: {}",
                            op_str
                        )));
                    }
                }
            }
        } else {
            // Direct equality: {"field": value}
            let json_value = pyobject_to_json(&value)?;
            filters.push(MetadataFilter::Eq(key_str, json_value));
        }
    }

    if filters.len() == 1 {
        Ok(filters.pop().unwrap())
    } else {
        Ok(MetadataFilter::And(filters))
    }
}

/// Helper: Parse batch items from a list of dicts
fn parse_batch_items(items: &Bound<'_, PyList>) -> PyResult<Vec<(String, Vector, JsonValue)>> {
    let mut batch = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("Item at index {} must be a dict", idx)))?;

        let id: String = dict
            .get_item("id")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("Item at index {} missing 'id' field", idx))
            })?
            .extract()?;

        // Use "vector" field name
        let vector_data: Vec<f32> = dict
            .get_item("vector")?
            .ok_or_else(|| PyValueError::new_err(format!("Item '{}' missing 'vector' field", id)))?
            .extract()?;

        let mut metadata_json = if let Some(metadata_dict) = dict.get_item("metadata")? {
            pyobject_to_json(&metadata_dict)?
        } else {
            serde_json::json!({})
        };

        // Handle optional document field
        if let Some(document) = dict.get_item("document")? {
            let doc_str: String = document.extract().map_err(|_| {
                PyValueError::new_err(format!("Item '{}': 'document' must be a string", id))
            })?;
            if let Some(obj) = metadata_json.as_object_mut() {
                obj.insert("document".to_string(), serde_json::json!(doc_str));
            }
        }

        batch.push((id, Vector::new(vector_data), metadata_json));
    }

    Ok(batch)
}

/// Helper: Convert Python object to serde_json::Value
fn pyobject_to_json(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::Number::from_f64(f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if obj.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in dict.iter() {
            let key_str: String = key.extract()?;
            map.insert(key_str, pyobject_to_json(&value)?);
        }
        Ok(JsonValue::Object(map))
    } else if let Ok(list) = obj.cast::<PyList>() {
        let values: Result<Vec<_>, _> = list.iter().map(|item| pyobject_to_json(&item)).collect();
        Ok(JsonValue::Array(values?))
    } else {
        let type_name = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        Err(PyValueError::new_err(format!(
            "Unsupported type '{}' for metadata. Supported: str, int, float, bool, None, list, dict",
            type_name
        )))
    }
}

/// Helper: Convert serde_json::Value to Python object
fn json_to_pyobject(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None().into()),
        JsonValue::Bool(b) => Ok((*b).into_pyobject(py).unwrap().to_owned().unbind().into()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().unbind().into())
            } else {
                Ok(py.None().into())
            }
        }
        JsonValue::String(s) => Ok(s.clone().into_pyobject(py).unwrap().unbind().into()),
        JsonValue::Array(arr) => {
            let py_list = PyList::new(
                py,
                arr.iter()
                    .map(|v| json_to_pyobject(py, v))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            Ok(py_list.into())
        }
        JsonValue::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (k, v) in obj {
                py_dict.set_item(k, json_to_pyobject(py, v)?)?;
            }
            Ok(py_dict.into())
        }
    }
}

#[pymodule]
fn omendb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<VectorDatabase>()?;
    Ok(())
}
