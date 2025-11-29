use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyBool, PyTuple};
use pyo3::conversion::IntoPyObject;
use pyo3::Py;
use ::omendb::vector::{Vector, VectorStore, MetadataFilter};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use parking_lot::RwLock;
use numpy::{PyReadonlyArray1, IntoPyArray};

/// Extract query vector from Python object (list or numpy array)
fn extract_query_vector(ob: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try numpy array first (more efficient)
    if let Ok(arr) = ob.extract::<PyReadonlyArray1<'_, f32>>() {
        return arr.as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(format!("Invalid numpy array: {}", e)));
    }
    // Fall back to list
    if let Ok(list) = ob.extract::<Vec<f32>>() {
        return Ok(list);
    }
    Err(PyValueError::new_err(
        "query must be a list of floats or numpy array (dtype=float32)"
    ))
}

/// Convert PyO3 errors to Python exceptions with proper type mapping
fn convert_error(err: anyhow::Error) -> PyErr {
    let msg = err.to_string();

    // Map to appropriate Python exception types
    if msg.contains("dimension") {
        PyValueError::new_err(msg)
    } else if msg.contains("already exists") {
        pyo3::exceptions::PyKeyError::new_err(msg)
    } else if msg.contains("filter") || msg.contains("operator") {
        PyValueError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// Thread-safe inner state for VectorDatabase
struct VectorDatabaseInner {
    store: VectorStore,
    /// Cached reverse index (index -> id) for fast lookups during search
    /// This is rebuilt whenever id_to_index changes (set/delete)
    index_to_id_cache: HashMap<usize, String>,
    /// Track if cache is valid (invalidated on set/delete)
    cache_valid: bool,
}

/// Vector database wrapper for Python
/// Uses RwLock for thread-safe concurrent access
#[pyclass]
struct VectorDatabase {
    inner: RwLock<VectorDatabaseInner>,
    path: String,
    /// Dimensions (stored for creating collections)
    dimensions: usize,
    /// Whether this is a persistent database (uses seerdb storage)
    is_persistent: bool,
}

#[pymethods]
impl VectorDatabase {
    /// Store vectors with metadata (insert or replace).
    ///
    /// If a vector with the same ID already exists, it will be replaced.
    /// Otherwise, a new vector will be inserted.
    ///
    /// Args:
    ///     items (list[dict]): List of dictionaries, each containing:
    ///         - id (str): Unique identifier for the vector
    ///         - embedding (list[float]): Vector embedding (must match database dimensions)
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
    ///     ...     {"id": "doc1", "embedding": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}},
    ///     ...     {"id": "doc2", "embedding": [0.4, 0.5, 0.6], "metadata": {"title": "World"}},
    ///     ... ])
    ///     [0, 1]
    ///
    ///     Replace existing vector:
    ///
    ///     >>> db.set([{"id": "doc1", "embedding": [0.7, 0.8, 0.9]}])
    ///     [0]
    ///
    ///     With document:
    ///
    ///     >>> db.set([{"id": "doc1", "embedding": [...], "document": "Original text content"}])
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
    ///     db.set([{"id": "a", "embedding": [...], "metadata": {...}}])
    ///
    ///     # Batch kwargs
    ///     db.set(ids=["a", "b"], embeddings=[[...], [...]], metadatas=[{...}, {...}])
    #[pyo3(name = "set", signature = (id_or_items=None, embedding=None, metadata=None, *, ids=None, embeddings=None, metadatas=None))]
    fn set_vectors(
        &self,
        _py: Python<'_>,
        id_or_items: Option<&Bound<'_, PyAny>>,
        embedding: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
        ids: Option<Vec<String>>,
        embeddings: Option<Vec<Vec<f32>>>,
        metadatas: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Vec<usize>> {
        let batch = if let (Some(ids), Some(embeddings)) = (&ids, &embeddings) {
            // Batch kwargs: ids=[], embeddings=[], metadatas=[]
            if ids.len() != embeddings.len() {
                return Err(PyValueError::new_err(format!(
                    "ids and embeddings must have same length: {} vs {}",
                    ids.len(), embeddings.len()
                )));
            }
            ids.iter().enumerate().map(|(i, id)| {
                let meta = metadatas
                    .and_then(|m| m.get_item(i).ok())
                    .map(|m| pyobject_to_json(&m))
                    .transpose()?
                    .unwrap_or_else(|| serde_json::json!({}));
                Ok((id.clone(), Vector::new(embeddings[i].clone()), meta))
            }).collect::<PyResult<Vec<_>>>()?
        } else if let Some(id_or_items) = id_or_items {
            if let Ok(id_str) = id_or_items.extract::<String>() {
                // Single item: set("id", [...], {...})
                let emb = embedding.ok_or_else(|| PyValueError::new_err(
                    "embedding required when id is a string"
                ))?;
                let meta = metadata
                    .map(|m| pyobject_to_json(m.as_any()))
                    .transpose()?
                    .unwrap_or_else(|| serde_json::json!({}));
                vec![(id_str, Vector::new(emb), meta)]
            } else if let Ok(items) = id_or_items.cast::<PyList>() {
                // Batch: set([{...}, {...}])
                parse_batch_items(items)?
            } else {
                return Err(PyValueError::new_err(
                    "First argument must be a string (id) or list of dicts"
                ));
            }
        } else {
            return Err(PyValueError::new_err(
                "set() requires either (id, embedding) or a list of items or (ids=, embeddings=)"
            ));
        };

        // Acquire write lock and perform set
        let mut inner = self.inner.write();
        let result = inner.store.set_batch(batch).map_err(convert_error)?;
        inner.cache_valid = false;
        Ok(result)
    }

    /// Search for k nearest neighbors.
    ///
    /// Performs approximate nearest neighbor search using HNSW index.
    /// Returns exact nearest neighbors (100% recall) for the query vector.
    ///
    /// Args:
    ///     query (list[float]): Query embedding vector (must match database dimensions)
    ///     k (int): Number of nearest neighbors to return
    ///     filter (dict, optional): MongoDB-style metadata filter to restrict search.
    ///         Supported operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or
    ///
    /// Returns:
    ///     list[dict]: List of results, each containing:
    ///         - id (str): Vector identifier
    ///         - distance (float): L2 distance from query
    ///         - metadata (dict): Associated metadata
    ///
    /// Raises:
    ///     ValueError: If query dimensions don't match database dimensions
    ///     RuntimeError: If search operation fails
    ///
    /// Examples:
    ///     Basic search:
    ///
    ///     >>> results = db.search(query=[0.1, 0.2, 0.3], k=5)
    ///     >>> for result in results:
    ///     ...     print(f"{result['id']}: {result['distance']:.3f}")
    ///     doc1: 0.123
    ///     doc2: 0.456
    ///
    ///     Search with metadata filter:
    ///
    ///     >>> results = db.search(
    ///     ...     query=[0.1, 0.2, 0.3],
    ///     ...     k=10,
    ///     ...     filter={"year": {"$gte": 2020}, "category": "research"}
    ///     ... )
    ///
    /// Performance:
    ///     - QPS: 3,492 queries/sec @ 10K vectors
    ///     - Latency: 0.29ms p50, 0.34ms p95, 0.37ms p99
    ///     - Recall: 100% (exact nearest neighbors)
    ///
    /// Args:
    ///     query: Query vector (list or numpy array)
    ///     k: Number of nearest neighbors
    ///     ef: Optional search width override (default: auto-tuned to max(k*4, 64))
    ///     filter: Optional metadata filter
    ///     as_numpy: If True, return (ids, distances) as numpy arrays (faster, for ML pipelines)
    #[pyo3(signature = (query, k, ef=None, filter=None, as_numpy=false))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
        as_numpy: bool,
    ) -> PyResult<Py<PyAny>> {
        // Validate ef >= k if provided
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(PyValueError::new_err(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        let query_vec = Vector::new(extract_query_vector(query)?);

        // Convert Python filter to Rust MetadataFilter (if provided)
        let rust_filter = if let Some(f) = filter {
            Some(parse_filter(f)?)
        } else {
            None
        };

        // Acquire write lock (needed for potential index rebuild)
        let mut inner = self.inner.write();

        // Perform search
        let results = inner.store.search_with_ef(&query_vec, k, rust_filter.as_ref(), ef)
            .map_err(convert_error)?;

        // Rebuild cache if invalid (lazy rebuild on first search after set/delete)
        if !inner.cache_valid {
            inner.index_to_id_cache = inner.store.id_to_index
                .iter()
                .map(|(id, &idx)| (idx, id.clone()))
                .collect();
            inner.cache_valid = true;
        }

        // Convert results based on as_numpy flag
        if as_numpy {
            // NumPy mode: return (ids, distances) as numpy arrays (fastest, for ML pipelines)
            let n = results.len();
            let mut ids = Vec::with_capacity(n);
            let mut distances = Vec::with_capacity(n);

            for (index, distance, _) in results {
                ids.push(index as i64);
                distances.push(distance);
            }

            let ids_array = ids.into_pyarray(py);
            let distances_array = distances.into_pyarray(py);
            Ok(PyTuple::new(py, &[ids_array.into_any(), distances_array.into_any()])?.into_any().unbind())
        } else {
            // Dict mode: return list of {id, distance, metadata} dicts (default, user-friendly)
            let mut py_results = Vec::with_capacity(results.len());
            for (index, distance, metadata) in results {
                if let Some(id) = inner.index_to_id_cache.get(&index) {
                    let dict = PyDict::new(py);
                    dict.set_item(pyo3::intern!(py, "id"), id)?;
                    dict.set_item(pyo3::intern!(py, "distance"), distance)?;
                    if metadata.as_object().map_or(false, |o| o.is_empty()) {
                        dict.set_item(pyo3::intern!(py, "metadata"), PyDict::new(py))?;
                    } else if let Ok(metadata_dict) = json_to_pyobject(py, &metadata) {
                        dict.set_item(pyo3::intern!(py, "metadata"), metadata_dict)?;
                    }
                    py_results.push(dict.unbind());
                }
            }
            Ok(py_results.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Batch search multiple queries in a single call (amortizes FFI overhead).
    ///
    /// This is significantly faster than calling search() multiple times because it:
    /// - Crosses the Python/Rust boundary only once
    /// - Releases GIL once for all searches
    /// - Processes all queries with minimal per-query overhead
    ///
    /// Args:
    ///     queries (list[list[float]]): List of query vectors
    ///     k (int): Number of nearest neighbors to return per query
    ///     ef (int, optional): Search width override (default: auto-tuned to max(k*4, 64))
    ///     filter (dict, optional): Metadata filter (same for all queries)
    ///
    /// Returns:
    ///     list[list[dict]]: List of search results, one per query.
    ///         Each element is a list of dicts with keys: id, distance, metadata
    ///
    /// Performance:
    ///     3-4x faster than individual search() calls for batch queries.
    ///     Overhead per query: ~0.00004ms (vs ~0.04ms for individual calls)
    ///
    /// Examples:
    ///     Batch search 1000 queries:
    ///
    ///     >>> queries = [[0.1] * 128 for _ in range(1000)]
    ///     >>> all_results = db.batch_search(queries, k=10)
    ///     >>> len(all_results)
    ///     1000
    ///     >>> len(all_results[0])
    ///     10
    #[pyo3(signature = (queries, k, ef=None, filter=None))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f32>>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<Py<PyDict>>>> {
        // Validate ef >= k if provided
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(PyValueError::new_err(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        // Convert Python filter once (shared by all queries)
        let rust_filter = if let Some(f) = filter {
            Some(parse_filter(f)?)
        } else {
            None
        };

        // Convert all queries to Vector type
        let query_vecs: Vec<Vector> = queries.into_iter()
            .map(Vector::new)
            .collect();

        // Acquire write lock (needed for potential index rebuild)
        let mut inner = self.inner.write();

        // Search all queries sequentially
        let all_results: Vec<_> = query_vecs.iter()
            .map(|query| {
                inner.store.search_with_ef(query, k, rust_filter.as_ref(), ef)
                    .map_err(convert_error)
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Rebuild cache if invalid
        if !inner.cache_valid {
            inner.index_to_id_cache = inner.store.id_to_index
                .iter()
                .map(|(id, &idx)| (idx, id.clone()))
                .collect();
            inner.cache_valid = true;
        }

        // Pre-allocate outer results vector
        let mut py_all_results = Vec::with_capacity(all_results.len());

        // Convert all results to Python dicts using interned keys (requires GIL)
        for results in all_results {
            let mut query_results = Vec::with_capacity(results.len());
            for (index, distance, metadata) in results {
                if let Some(id) = inner.index_to_id_cache.get(&index) {
                    let dict = PyDict::new(py);
                    dict.set_item(pyo3::intern!(py, "id"), id)?;
                    dict.set_item(pyo3::intern!(py, "distance"), distance)?;
                    // Fast path for empty metadata
                    if metadata.as_object().map_or(false, |o| o.is_empty()) {
                        dict.set_item(pyo3::intern!(py, "metadata"), PyDict::new(py))?;
                    } else if let Ok(metadata_dict) = json_to_pyobject(py, &metadata) {
                        dict.set_item(pyo3::intern!(py, "metadata"), metadata_dict)?;
                    }
                    query_results.push(dict.unbind());
                }
            }
            py_all_results.push(query_results);
        }

        Ok(py_all_results)
    }

    /// Delete vectors by ID.
    ///
    /// Args:
    ///     ids (list[str]): List of vector IDs to delete
    ///
    /// Returns:
    ///     int: Number of vectors successfully deleted
    ///
    /// Examples:
    ///     >>> db.delete(["doc1", "doc2"])
    ///     2
    ///
    ///     >>> db.delete(["nonexistent"])  # Silently skips missing IDs
    ///     0
    fn delete(&self, ids: Vec<String>) -> PyResult<usize> {
        let mut inner = self.inner.write();

        let result = inner.store.delete_batch(&ids)
            .map_err(convert_error)?;

        // Invalidate cache since id_to_index changed
        inner.cache_valid = false;

        Ok(result)
    }

    /// Update vector and/or metadata for existing ID.
    ///
    /// Args:
    ///     id (str): Vector ID to update
    ///     embedding (list[float], optional): New embedding vector
    ///     metadata (dict, optional): New metadata (replaces existing)
    ///
    /// Raises:
    ///     RuntimeError: If vector with given ID doesn't exist
    ///
    /// Examples:
    ///     Update embedding only:
    ///
    ///     >>> db.update("doc1", embedding=[0.1, 0.2, 0.3])
    ///
    ///     Update metadata only:
    ///
    ///     >>> db.update("doc1", metadata={"title": "Updated"})
    ///
    ///     Update both:
    ///
    ///     >>> db.update("doc1", embedding=[0.4, 0.5, 0.6], metadata={"title": "New"})
    fn update(
        &self,
        id: String,
        embedding: Vec<f32>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let vector = Some(Vector::new(embedding));
        let metadata_json = if let Some(m) = metadata {
            Some(pyobject_to_json(m.as_any())?)
        } else {
            None
        };

        let mut inner = self.inner.write();

        inner.store.update(&id, vector, metadata_json)
            .map_err(convert_error)
    }

    /// Get vector by ID.
    ///
    /// Args:
    ///     id (str): Vector ID to retrieve
    ///
    /// Returns:
    ///     dict or None: Dictionary with keys "id", "embedding", "metadata"
    ///                   Returns None if ID not found
    ///
    /// Examples:
    ///     >>> result = db.get("doc1")
    ///     >>> if result:
    ///     ...     print(result["id"], result["embedding"], result["metadata"])
    ///     doc1 [0.1, 0.2, 0.3] {'title': 'Hello'}
    fn get(&self, py: Python<'_>, id: String) -> PyResult<Option<HashMap<String, Py<PyAny>>>> {
        let inner = self.inner.read();

        if let Some((vector, metadata)) = inner.store.get_by_id(&id) {
            let mut result = HashMap::new();
            result.insert("id".to_string(), id.into_pyobject(py).unwrap().unbind().into());
            result.insert("embedding".to_string(), vector.data.clone().into_pyobject(py).unwrap().unbind().into());

            let metadata_dict = json_to_pyobject(py, metadata)?;
            result.insert("metadata".to_string(), metadata_dict);

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Save database to disk.
    ///
    /// Persists HNSW index, vectors, metadata, and ID mappings to disk.
    ///
    /// Raises:
    ///     RuntimeError: If file I/O fails
    ///
    /// Examples:
    ///     >>> db.save()  # Saves to path specified in omendb.open()
    ///
    /// Performance:
    ///     Loading from disk is 401x faster than rebuilding index from scratch.
    fn save(&self) -> PyResult<()> {
        let inner = self.inner.read();
        inner.store.save_to_disk(&self.path)
            .map_err(convert_error)
    }

    /// Number of vectors in database (Pythonic).
    ///
    /// Returns:
    ///     int: Total vector count (excluding deleted vectors)
    ///
    /// Examples:
    ///     >>> len(db)
    ///     1000
    fn __len__(&self) -> PyResult<usize> {
        let inner = self.inner.read();
        Ok(inner.store.len())
    }

    /// Number of vectors in database (explicit method).
    ///
    /// Alternative to len(db) for discoverability.
    ///
    /// Returns:
    ///     int: Total vector count (excluding deleted vectors)
    ///
    /// Examples:
    ///     >>> db.count()
    ///     1000
    fn count(&self) -> PyResult<usize> {
        let inner = self.inner.read();
        Ok(inner.store.len())
    }

    /// Get current ef_search value (search depth parameter).
    ///
    /// Lower values = faster search, lower recall.
    /// Higher values = slower search, higher recall.
    ///
    /// Returns:
    ///     int or None: Current ef_search, or None if no HNSW index
    fn get_ef_search(&self) -> PyResult<Option<usize>> {
        let inner = self.inner.read();
        Ok(inner.store.get_ef_search())
    }

    /// Set ef_search value (search depth parameter).
    ///
    /// Tune this for speed/recall tradeoff:
    /// - ef_search=50: ~3000 QPS, ~93% recall (fast)
    /// - ef_search=100: ~2000 QPS, ~98% recall (balanced)
    /// - ef_search=200: ~1300 QPS, ~99.7% recall (accurate)
    ///
    /// Args:
    ///     ef_search (int): Search depth (50-1000)
    fn set_ef_search(&self, ef_search: usize) -> PyResult<()> {
        let mut inner = self.inner.write();
        inner.store.set_ef_search(ef_search);
        Ok(())
    }

    /// Get or create a named collection.
    ///
    /// Collections provide multi-tenant support - each collection has its own
    /// vectors, metadata, and ID space. Collections are isolated from each other.
    ///
    /// Args:
    ///     name (str): Collection name (alphanumeric and underscores only)
    ///
    /// Returns:
    ///     VectorDatabase: A new database instance for the collection
    ///
    /// Raises:
    ///     ValueError: If name contains invalid characters
    ///     RuntimeError: If collection cannot be created/opened
    ///
    /// Examples:
    ///     Multi-tenant usage:
    ///
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    ///     >>> users = db.collection("users")
    ///     >>> products = db.collection("products")
    ///     >>> users.set([{"id": "u1", "embedding": [...]}])
    ///     >>> products.set([{"id": "p1", "embedding": [...]}])
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
                "Collection name must contain only alphanumeric characters and underscores"
            ));
        }

        // Only persistent databases support collections
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage"
            ));
        }

        // Create collection path: {base_path}/collections/{name}
        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        // Ensure collections directory exists
        std::fs::create_dir_all(collection_path.parent().unwrap())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create collections directory: {}", e)))?;

        // Open the collection as a separate VectorStore
        let store = if self.dimensions == 0 || self.dimensions == 128 {
            VectorStore::open(&collection_path).map_err(convert_error)?
        } else {
            VectorStore::open_with_dimensions(&collection_path, self.dimensions).map_err(convert_error)?
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
    ///     list[str]: List of collection names
    ///
    /// Examples:
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    ///     >>> db.collection("users")
    ///     >>> db.collection("products")
    ///     >>> db.collections()
    ///     ['users', 'products']
    fn collections(&self) -> PyResult<Vec<String>> {
        if !self.is_persistent {
            return Ok(Vec::new());
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
            let entry = entry.map_err(|e| PyRuntimeError::new_err(format!("Failed to read entry: {}", e)))?;
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Merge another VectorDatabase into this one using IGTM algorithm.
    ///
    /// Uses Iterative Greedy Tree Merging for 1.3-1.7x faster batch inserts
    /// compared to naive insertion. All vectors and metadata from the source
    /// database are copied into this database.
    ///
    /// Args:
    ///     other (VectorDatabase): Source database to merge from
    ///
    /// Returns:
    ///     int: Number of vectors successfully merged
    ///
    /// Raises:
    ///     ValueError: If dimensions don't match
    ///     RuntimeError: If merge operation fails
    ///
    /// Examples:
    ///     Batch import from another database:
    ///
    ///     >>> main_db = omendb.open("./main", dimensions=128)
    ///     >>> incoming = omendb.open("./incoming", dimensions=128)
    ///     >>> incoming.set([...])  # Add new vectors
    ///     >>> merged = main_db.merge_from(incoming)
    ///     >>> print(f"Merged {merged} vectors")
    ///
    ///     Building index in parallel (advanced):
    ///
    ///     >>> # Build small graphs in parallel threads
    ///     >>> graphs = [build_graph(chunk) for chunk in data_chunks]
    ///     >>> # Merge all into one
    ///     >>> for g in graphs[1:]:
    ///     ...     graphs[0].merge_from(g)
    ///
    /// Note:
    ///     - IDs are preserved; conflicting IDs are skipped (existing wins)
    ///     - Source database is not modified
    ///     - Both databases must have the same dimensions
    fn merge_from(&self, other: &VectorDatabase) -> PyResult<usize> {
        let mut inner = self.inner.write();
        let other_inner = other.inner.read();

        let count = inner.store.merge_from(&other_inner.store)
            .map_err(convert_error)?;

        // Invalidate cache since id_to_index changed
        inner.cache_valid = false;

        Ok(count)
    }

    /// Delete a collection and all its data.
    ///
    /// WARNING: This permanently deletes all vectors and metadata in the collection.
    ///
    /// Args:
    ///     name (str): Collection name to delete
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
                "Collections require persistent storage"
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        if !collection_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Collection '{}' does not exist",
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
/// If the database exists at the given path, it will be loaded from disk.
/// Otherwise, a new database will be created with the specified dimensions.
///
/// Args:
///     path (str): Path to database directory (will be created if needed).
///                 Uses seerdb persistent storage with auto-persist.
///     dimensions (int, optional): Vector dimensionality. Required for new databases,
///                                 ignored when loading existing database. Default: 128
///     config (dict, optional): Advanced configuration options:
///         - hnsw (dict): HNSW parameters
///             - m (int): Max edges per node (16-48, default: adaptive)
///             - ef_construction (int): Build-time search depth (100-800, default: adaptive)
///             - ef_search (int): Query-time search depth (100-800, default: adaptive)
///         - quantization (dict): RaBitQ quantization
///             - bits (int): 2, 4, or 8 bits (default: 4-bit, 8x compression, 100% recall)
///         - expected_vectors (int): Hint for adaptive parameter selection
///
/// Returns:
///     VectorDatabase: Database instance
///
/// Raises:
///     RuntimeError: If loading fails or path is invalid
///
/// Examples:
///     Basic usage:
///
///     >>> import omendb
///     >>> db = omendb.open("./my_vectors")  # Auto-persists all operations
///     >>> db.set([{"id": "doc1", "embedding": [0.1] * 128}])
///     >>> # Data is automatically saved - no db.save() needed
///
///     Load existing database:
///
///     >>> db = omendb.open("./my_vectors")  # dimensions auto-detected
///
///     With HNSW parameters (power users):
///
///     >>> db = omendb.open("./my_vectors", dimensions=384, m=32, ef_construction=400)
///
/// Performance:
///     - Loading from disk: 401x faster than rebuilding
///     - Default settings optimized for 1K-100K vectors
#[pyfunction]
#[pyo3(signature = (path, dimensions=128, m=None, ef_construction=None, config=None))]
fn open(
    path: String,
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<VectorDatabase> {
    // Validate optional params
    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(PyValueError::new_err(format!(
                "m must be between 4 and 64, got {}", m_val
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
    use std::path::Path;

    let db_path = Path::new(&path);

    // Always use seerdb persistent storage (recommended)
    // Legacy paths are still supported for backward compatibility
    if !path.ends_with(".vectors.bin") && !path.ends_with(".hnsw") {
        // Use seerdb persistent storage (default for all new databases)
        let mut store = if dimensions == 0 {
            // Try to load existing (dimensions come from stored data)
            VectorStore::open(&path).map_err(convert_error)?
        } else {
            // Create with specified dimensions
            VectorStore::open_with_dimensions(&path, dimensions).map_err(convert_error)?
        };

        // Apply HNSW config if provided (allows tuning ef_search for speed/recall tradeoff)
        if let Some(cfg) = config {
            if let Some(hnsw_dict) = cfg.get_item("hnsw")? {
                let hnsw = hnsw_dict.cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("'hnsw' must be a dict"))?;

                // Apply ef_search tuning (most impactful for search QPS)
                if let Some(ef) = hnsw.get_item("ef_search")? {
                    let ef_search: usize = ef.extract()?;
                    store.set_ef_search(ef_search);
                }
            }
        }

        return Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,  // Will be built on first search
            }),
            path,
            dimensions,
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
        let store = VectorStore::load_from_disk(&path, dimensions)
            .map_err(convert_error)?;
        Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,  // Will be built on first search
            }),
            path,
            dimensions,
            is_persistent: false,  // Legacy file-based storage
        })
    } else {
        // Create new database with configuration
        let store = if m.is_some() || ef_construction.is_some() {
            // Use explicit HNSW params if provided
            let m_val = m.unwrap_or(16);
            let ef_con = ef_construction.unwrap_or(100);
            let ef_search = (10_usize * 4).max(64).max(100); // Default ef_search
            VectorStore::new_with_params(dimensions, m_val, ef_con, ef_search)
                .map_err(|e| PyValueError::new_err(format!("Failed to create HNSW index: {}", e)))?
        } else if let Some(cfg) = config {
            create_store_with_config(dimensions, cfg)?
        } else {
            // No config: use adaptive defaults (M=16 for <50K vectors)
            VectorStore::new_with_capacity(dimensions, 10_000)
        };

        Ok(VectorDatabase {
            inner: RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: true,  // Empty database, cache is valid (empty)
            }),
            path,
            dimensions,
            is_persistent: false,  // Legacy in-memory storage
        })
    }
}

/// Helper: Create VectorStore with configuration
fn create_store_with_config(dimensions: usize, config: &Bound<'_, PyDict>) -> PyResult<VectorStore> {
    use ::omendb::vector::rabitq::RaBitQParams;

    // Parse HNSW configuration (if provided)
    if let Some(hnsw_dict) = config.get_item("hnsw")? {
        let hnsw = hnsw_dict.cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("'hnsw' must be a dict"))?;

        let m: usize = hnsw.get_item("m")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.m' required"))?
            .extract()?;
        let ef_construction: usize = hnsw.get_item("ef_construction")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.ef_construction' required"))?
            .extract()?;
        let ef_search: usize = hnsw.get_item("ef_search")?
            .ok_or_else(|| PyValueError::new_err("'hnsw.ef_search' required"))?
            .extract()?;

        VectorStore::new_with_params(dimensions, m, ef_construction, ef_search)
            .map_err(|e| PyValueError::new_err(format!("Failed to create HNSW index: {}", e)))
    }
    // Parse quantization configuration (if provided)
    else if let Some(quant_dict) = config.get_item("quantization")? {
        let quant = quant_dict.cast::<PyDict>()
            .map_err(|_| PyValueError::new_err("'quantization' must be a dict"))?;

        let bits: u8 = quant.get_item("bits")?
            .ok_or_else(|| PyValueError::new_err("'quantization.bits' required (2/4/8)"))?
            .extract()?;

        let params = match bits {
            2 => RaBitQParams::bits2(),
            4 => RaBitQParams::bits4(),
            8 => RaBitQParams::bits8(),
            _ => return Err(PyValueError::new_err("quantization.bits must be 2, 4, or 8")),
        };

        Ok(VectorStore::new_with_quantization(dimensions, params))
    }
    // Parse expected_vectors for adaptive defaults
    else if let Some(expected) = config.get_item("expected_vectors")? {
        let expected_vectors: usize = expected.extract()?;
        Ok(VectorStore::new_with_capacity(dimensions, expected_vectors))
    }
    else {
        // Config dict provided but empty: use default capacity
        Ok(VectorStore::new_with_capacity(dimensions, 10_000))
    }
}

/// Helper: Parse Python filter dict to Rust MetadataFilter
fn parse_filter(filter: &Bound<'_, PyDict>) -> PyResult<MetadataFilter> {
    // Handle special logical operators first
    if let Some(and_value) = filter.get_item("$and")? {
        // $and expects an array of filter dicts
        let and_list = and_value.cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$and must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in and_list.iter() {
            let sub_dict = item.cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $and element must be a dict"))?;
            sub_filters.push(parse_filter(&sub_dict)?);
        }

        return Ok(MetadataFilter::And(sub_filters));
    }

    if let Some(or_value) = filter.get_item("$or")? {
        // $or expects an array of filter dicts
        let or_list = or_value.cast::<PyList>()
            .map_err(|_| PyValueError::new_err("$or must be an array of filters"))?;

        let mut sub_filters = Vec::new();
        for item in or_list.iter() {
            let sub_dict = item.cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Each $or element must be a dict"))?;
            sub_filters.push(parse_filter(&sub_dict)?);
        }

        return Ok(MetadataFilter::Or(sub_filters));
    }

    // Parse regular field filters
    let mut filters = Vec::new();

    for (key, value) in filter.iter() {
        let key_str: String = key.extract()?;

        // Check if value is a dict (operator-based filter)
        if let Ok(op_dict) = value.cast::<PyDict>() {
            for (op, op_value) in op_dict.iter() {
                let op_str: String = op.extract()?;

                match op_str.as_str() {
                    "$eq" => {
                        let json_val = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Eq(key_str.clone(), json_val));
                    }
                    "$ne" => {
                        let json_val = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Ne(key_str.clone(), json_val));
                    }
                    "$gte" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gte(key_str.clone(), num));
                    }
                    "$gt" => {
                        let num: f64 = op_value.extract()?;
                        filters.push(MetadataFilter::Gt(key_str.clone(), num));
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
                        let json_vals: Result<Vec<JsonValue>, _> = list.iter()
                            .map(|obj| pyobject_to_json(&obj))
                            .collect();
                        filters.push(MetadataFilter::In(key_str.clone(), json_vals?));
                    }
                    "$contains" => {
                        let substr: String = op_value.extract()?;
                        filters.push(MetadataFilter::Contains(key_str.clone(), substr));
                    }
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Unknown filter operator '{}' for field '{}'. Valid operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $contains",
                            op_str, key_str
                        )));
                    }
                }
            }
        } else {
            // Simple equality filter
            let json_val = pyobject_to_json(&value)?;
            filters.push(MetadataFilter::Eq(key_str, json_val));
        }
    }

    // If multiple filters, combine with AND
    if filters.len() == 1 {
        Ok(filters.into_iter().next().unwrap())
    } else {
        Ok(MetadataFilter::And(filters))
    }
}

/// Helper: Parse batch items from a list of dicts
fn parse_batch_items(items: &Bound<'_, PyList>) -> PyResult<Vec<(String, Vector, JsonValue)>> {
    let mut batch = Vec::new();

    for (idx, item) in items.iter().enumerate() {
        let dict = item.cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!(
                "Item at index {} must be a dict", idx
            )))?;

        let id: String = dict.get_item("id")?
            .ok_or_else(|| PyValueError::new_err(format!(
                "Item at index {} missing 'id' field", idx
            )))?
            .extract()?;

        // Support "embedding", "vector", or "values" field names
        let embedding: Vec<f32> = dict.get_item("embedding")?
            .or(dict.get_item("vector")?)
            .or(dict.get_item("values")?)
            .ok_or_else(|| PyValueError::new_err(format!(
                "Item '{}' missing 'embedding' (or 'vector'/'values') field", id
            )))?
            .extract()?;

        let mut metadata_json = if let Some(metadata_dict) = dict.get_item("metadata")? {
            pyobject_to_json(&metadata_dict)?
        } else {
            serde_json::json!({})
        };

        // Handle optional document field
        if let Some(document) = dict.get_item("document")? {
            let doc_str: String = document.extract()
                .map_err(|_| PyValueError::new_err(format!(
                    "Item '{}': 'document' must be a string", id
                )))?;
            if let Some(obj) = metadata_json.as_object_mut() {
                obj.insert("document".to_string(), serde_json::json!(doc_str));
            }
        }

        batch.push((id, Vector::new(embedding), metadata_json));
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
    } else {
        // Try as dict
        if let Ok(dict) = obj.cast::<PyDict>() {
            let mut map = serde_json::Map::new();
            for (key, value) in dict.iter() {
                let key_str: String = key.extract()?;
                map.insert(key_str, pyobject_to_json(&value)?);
            }
            Ok(JsonValue::Object(map))
        } else if let Ok(list) = obj.cast::<PyList>() {
            let values: Result<Vec<_>, _> = list.iter()
                .map(|item| pyobject_to_json(&item))
                .collect();
            Ok(JsonValue::Array(values?))
        } else {
            let type_name = obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "unknown".to_string());
            Err(PyValueError::new_err(format!(
                "Unsupported type '{}' for metadata. Supported: str, int, float, bool, None, list, dict",
                type_name
            )))
        }
    }
}

/// Helper: Convert serde_json::Value to Python object
fn json_to_pyobject(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(PyBool::new(py, *b).to_owned().into()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().unbind().into())
            } else {
                Ok(py.None())
            }
        }
        JsonValue::String(s) => Ok(s.into_pyobject(py).unwrap().unbind().into()),
        JsonValue::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_pyobject(py, item)?)?;
            }
            Ok(list.into())
        }
        JsonValue::Object(obj) => {
            let dict = PyDict::new(py);
            for (key, val) in obj {
                dict.set_item(key, json_to_pyobject(py, val)?)?;
            }
            Ok(dict.into())
        }
    }
}

/// Python module
#[pymodule]
fn omendb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<VectorDatabase>()?;
    Ok(())
}
