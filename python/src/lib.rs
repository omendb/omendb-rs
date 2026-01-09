// Allow these lints - they're structural API design choices that work correctly
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use omendb_core::text::TextSearchConfig;
use omendb_core::vector::{
    MetadataFilter, QuantizationMode, RaBitQParams, Vector, VectorStore, VectorStoreOptions,
};
use parking_lot::RwLock;
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Parse quantization parameter and return QuantizationMode if enabled
///
/// Accepts:
/// - True → SQ8 (4x compression, ~99% recall) - RECOMMENDED
/// - "sq8" → SQ8 (explicit)
/// - "rabitq" → RaBitQ 4-bit (8x compression, ~98% recall)
/// - "binary" → Binary (32x compression, ~70-85% recall, fastest search)
/// - None/False → no quantization (full precision)
///
/// Returns Ok(Some(mode)) if quantization enabled, Ok(None) if disabled
fn parse_quantization(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Option<QuantizationMode>> {
    let Some(value) = ob else {
        return Ok(None);
    };

    // Handle boolean: True enables SQ8 (default), False disables
    if let Ok(b) = value.extract::<bool>() {
        return if b {
            Ok(Some(QuantizationMode::SQ8)) // SQ8 is the new default
        } else {
            Ok(None)
        };
    }

    // Handle string quantization modes
    if let Ok(mode) = value.extract::<String>() {
        return match mode.to_lowercase().as_str() {
            "sq8" => Ok(Some(QuantizationMode::SQ8)),
            "binary" => Ok(Some(QuantizationMode::Binary)), // 32x compression, fastest
            "rabitq" | "rabitq-4" | "rabitq_4" => {
                Ok(Some(QuantizationMode::RaBitQ(RaBitQParams::bits4()))) // 8x compression
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown quantization mode: '{}'\n\
                  Valid modes:\n\
                  - True or 'sq8':  4x smaller, ~99% recall (RECOMMENDED)\n\
                  - 'rabitq':       8x smaller, ~98% recall (balanced)\n\
                  - 'binary':      32x smaller, ~70-85% recall (fastest)",
                mode
            ))),
        };
    }

    Err(PyValueError::new_err(
        "quantization must be True, False, or a string:\n\
          - True or 'sq8':  4x smaller, ~99% recall (RECOMMENDED)\n\
          - 'rabitq':       8x smaller, ~98% recall (balanced)\n\
          - 'binary':      32x smaller, ~70-85% recall (fastest)",
    ))
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
    if msg.contains("dimension") || msg.contains("not found") || msg.contains("does not exist") {
        PyValueError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

/// Build VectorStoreOptions from open() parameters
fn build_store_options(
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    quant_mode: Option<QuantizationMode>,
    rescore: Option<bool>,
    oversample: Option<f32>,
    metric: Option<&str>,
) -> PyResult<VectorStoreOptions> {
    let mut options = VectorStoreOptions::default().dimensions(dimensions);

    if let Some(m_val) = m {
        options = options.m(m_val);
    }
    if let Some(ef_con) = ef_construction {
        options = options.ef_construction(ef_con);
    }
    if let Some(ef_s) = ef_search {
        options = options.ef_search(ef_s);
    }
    if let Some(mode) = quant_mode {
        options = options.quantization(mode);
    }
    if let Some(rescore_val) = rescore {
        options = options.rescore(rescore_val);
    }
    if let Some(oversample_val) = oversample {
        options = options.oversample(oversample_val);
    }
    if let Some(metric_str) = metric {
        options = options.metric(metric_str).map_err(PyValueError::new_err)?;
    }

    Ok(options)
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
///
/// Supports context manager protocol for automatic cleanup:
///
/// ```python
/// with omendb.open("./db", dimensions=768) as db:
///     db.set([...])
/// # Automatically flushed on exit
/// ```
#[pyclass]
pub struct VectorDatabase {
    inner: Arc<RwLock<VectorDatabaseInner>>,
    path: String,
    dimensions: usize,
    is_persistent: bool,
    /// Cache of open collection handles (same name = same object)
    collections_cache: RwLock<HashMap<String, Py<VectorDatabase>>>,
}

/// Lazy iterator for VectorDatabase IDs.
///
/// Memory efficient: iterates over IDs one at a time from a snapshot.
/// Handles deletions during iteration gracefully (skips deleted IDs).
#[pyclass]
pub struct VectorDatabaseIdIterator {
    /// Reference to the database inner state
    inner: Arc<RwLock<VectorDatabaseInner>>,
    /// IDs to iterate over
    ids: Vec<String>,
    /// Current position
    index: usize,
}

#[pymethods]
impl VectorDatabaseIdIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        // Loop to skip items deleted during iteration
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            // Check if ID still exists
            let inner = self.inner.read();
            if inner.store.contains(id) {
                return Some(id.clone());
            }
            // Item was deleted during iteration, continue to next
        }
        None
    }
}

/// Lazy iterator for VectorDatabase items.
///
/// Enables `for item in db:` syntax with true lazy evaluation.
/// Memory efficient: stores only IDs (~20MB for 1M items), fetches vectors one at a time.
/// Handles items deleted during iteration gracefully (skips them).
#[pyclass]
pub struct VectorDatabaseIterator {
    /// Reference to the database inner state
    inner: Arc<RwLock<VectorDatabaseInner>>,
    /// IDs to iterate over (lightweight - just strings)
    ids: Vec<String>,
    /// Current position
    index: usize,
}

#[pymethods]
impl VectorDatabaseIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<HashMap<String, Py<PyAny>>>> {
        // Loop to skip items deleted during iteration
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            // Fetch item lazily - only loads one vector at a time
            let inner = self.inner.read();
            if let Some((vec, meta)) = inner.store.get(id) {
                let mut result = HashMap::new();
                result.insert(
                    "id".to_string(),
                    id.clone().into_pyobject(py).unwrap().unbind().into(),
                );
                result.insert(
                    "vector".to_string(),
                    vec.data.clone().into_pyobject(py).unwrap().unbind(),
                );
                result.insert("metadata".to_string(), json_to_pyobject(py, &meta)?);
                return Ok(Some(result));
            }
            // Item was deleted during iteration, continue to next
        }
        Ok(None)
    }
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

        // Use interned strings for dict keys (hot path optimization)
        dict.set_item(pyo3::intern!(py, "id"), id)?;
        dict.set_item(pyo3::intern!(py, "distance"), *distance)?;

        // Convert metadata to Python dict
        let metadata_dict = json_to_pyobject(py, metadata)?;
        dict.set_item(pyo3::intern!(py, "metadata"), metadata_dict)?;

        py_results.push(dict.unbind());
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
    /// When any item includes a `text` field, text search is automatically enabled.
    /// This allows immediate use of search_hybrid() without calling enable_text_search().
    ///
    /// Args:
    ///     items (list[dict]): List of dictionaries, each containing:
    ///         - id (str): Unique identifier for the vector
    ///         - vector (list[float]): Vector data (must match database dimensions)
    ///         - metadata (dict, optional): Arbitrary metadata as JSON-compatible dict
    ///         - text (str, optional): Text for hybrid search - indexed for BM25 AND
    ///           auto-stored in metadata["text"] for retrieval
    ///
    /// Returns:
    ///     int: Number of vectors inserted/updated
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
    ///     2
    ///
    ///     With text for hybrid search (auto-enables text search):
    ///
    ///     >>> db.set([{"id": "doc1", "vector": [...], "text": "Machine learning intro"}])
    ///     >>> db.get("doc1")["metadata"]["text"]  # Text is auto-stored
    ///     'Machine learning intro'
    ///     >>> results = db.search_hybrid([...], "machine learning", k=10)
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
        py: Python<'_>,
        id_or_items: Option<&Bound<'_, PyAny>>,
        vector: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
        ids: Option<Vec<String>>,
        vectors: Option<Vec<Vec<f32>>>,
        metadatas: Option<&Bound<'_, PyList>>,
    ) -> PyResult<usize> {
        // Handle kwargs batch format (no text support in this path)
        if let (Some(ids), Some(vectors)) = (&ids, &vectors) {
            if ids.len() != vectors.len() {
                return Err(PyValueError::new_err(format!(
                    "ids and vectors must have same length: {} vs {}",
                    ids.len(),
                    vectors.len()
                )));
            }
            let batch: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let meta = metadatas
                        .and_then(|m| m.get_item(i).ok())
                        .map(|m| pyobject_to_json(&m))
                        .transpose()?
                        .unwrap_or_else(|| serde_json::json!({}));
                    Ok((id.clone(), Vector::new(vectors[i].clone()), meta))
                })
                .collect::<PyResult<Vec<_>>>()?;

            // Release GIL during batch insert
            let inner_arc = Arc::clone(&self.inner);
            let result = py.detach(|| {
                let mut inner = inner_arc.write();
                let result = inner.store.set_batch(batch).map_err(convert_error);
                inner.cache_valid = false;
                result
            })?;
            return Ok(result.len());
        }

        // Handle single item: set("id", [...], {...})
        if let Some(id_or_items) = id_or_items {
            if let Ok(id_str) = id_or_items.extract::<String>() {
                let vec_data = vector
                    .ok_or_else(|| PyValueError::new_err("vector required when id is a string"))?;
                let meta = metadata
                    .map(|m| pyobject_to_json(m.as_any()))
                    .transpose()?
                    .unwrap_or_else(|| serde_json::json!({}));

                let mut inner = self.inner.write();
                inner
                    .store
                    .set(id_str, Vector::new(vec_data), meta)
                    .map_err(convert_error)?;
                inner.cache_valid = false;
                return Ok(1);
            }

            // Handle batch: set([{...}, {...}])
            if let Ok(items) = id_or_items.cast::<PyList>() {
                let parsed = parse_batch_items_with_text(items)?;

                // Check if any items have text
                let has_text = parsed.iter().any(|item| item.text.is_some());

                // Release GIL during batch insert
                let inner_arc = Arc::clone(&self.inner);
                let count = py.detach(move || -> PyResult<usize> {
                    let mut inner = inner_arc.write();

                    // Auto-enable text search if text field is present
                    if has_text && !inner.store.has_text_search() {
                        inner.store.enable_text_search().map_err(convert_error)?;
                    }

                    // Insert items - use batch path when no text for performance
                    let results = if has_text {
                        // Slow path: items with text must be inserted individually
                        let mut results = Vec::with_capacity(parsed.len());
                        for item in parsed {
                            let result = if let Some(text) = item.text {
                                inner
                                    .store
                                    .set_with_text(item.id, item.vector, &text, item.metadata)
                                    .map_err(convert_error)?
                            } else {
                                inner
                                    .store
                                    .set(item.id, item.vector, item.metadata)
                                    .map_err(convert_error)?
                            };
                            results.push(result);
                        }
                        results
                    } else {
                        // Fast path: use set_batch for items without text
                        let batch: Vec<_> = parsed
                            .into_iter()
                            .map(|item| (item.id, item.vector, item.metadata))
                            .collect();
                        inner.store.set_batch(batch).map_err(convert_error)?
                    };

                    inner.cache_valid = false;
                    Ok(results.len())
                })?;
                return Ok(count);
            }

            return Err(PyValueError::new_err(
                "First argument must be a string (id) or list of dicts",
            ));
        }

        Err(PyValueError::new_err(
            "set() requires either (id, vector) or a list of items or (ids=, vectors=)",
        ))
    }

    /// Search for k nearest neighbors (single query).
    ///
    /// Releases the GIL during search for better concurrency with Python threads.
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
    ///
    ///     With max_distance (filter out distant results):
    ///     >>> db.search([...], k=10, max_distance=0.5)
    #[pyo3(name = "search", signature = (query, k, ef=None, filter=None, max_distance=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
        max_distance: Option<f32>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(PyValueError::new_err(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }
        if let Some(max_dist) = max_distance {
            if max_dist < 0.0 {
                return Err(PyValueError::new_err("max_distance must be non-negative"));
            }
        }

        // Extract Python objects before releasing GIL
        let query_vec = Vector::new(extract_query_vector(query)?);
        let rust_filter = filter.map(parse_filter).transpose()?;

        // Ensure index and cache are ready before releasing GIL
        {
            let needs_rebuild = {
                let inner = self.inner.read();
                !inner.cache_valid || inner.store.needs_index_rebuild()
            };

            if needs_rebuild {
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
        }

        // Clone Arc for use inside detach
        let inner_arc = Arc::clone(&self.inner);

        // Release GIL during compute-intensive search
        let results = py.detach(|| {
            let inner = inner_arc.read();
            inner.store.search_with_options_readonly(
                &query_vec,
                k,
                rust_filter.as_ref(),
                ef,
                max_distance,
            )
        });

        let results = results.map_err(convert_error)?;

        // Convert to Python (needs GIL)
        let inner = self.inner.read();
        results_to_py(py, &results, &inner.index_to_id_cache)
    }

    /// Debug timing search - returns timing breakdown in microseconds
    #[pyo3(name = "_debug_search_timing", signature = (query, k, n_iterations=100))]
    fn debug_search_timing(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        n_iterations: usize,
    ) -> PyResult<HashMap<String, f64>> {
        let query_vec = Vector::new(extract_query_vector(query)?);

        // Ensure index ready (not timed - one-time cost)
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

        let inner_arc = Arc::clone(&self.inner);

        // Time just the Rust search (no result conversion)
        let t0 = Instant::now();
        for _ in 0..n_iterations {
            py.detach(|| {
                let inner = inner_arc.read();
                let _ = inner
                    .store
                    .search_with_options_readonly(&query_vec, k, None, None, None);
            });
        }
        let t_search_total = t0.elapsed().as_micros() as f64;

        // Time knn_search_ef directly (bypass VectorStore wrapper)
        let t0 = Instant::now();
        for _ in 0..n_iterations {
            py.detach(|| {
                let inner = inner_arc.read();
                if let Some(ref idx) = inner.store.hnsw_index {
                    let _ = idx.search_ef(&query_vec.data, k, 100);
                }
            });
        }
        let t_hnsw_total = t0.elapsed().as_micros() as f64;

        // Time HNSW search in tight Rust loop (no GIL release per iteration)
        let t0 = Instant::now();
        py.detach(|| {
            let inner = inner_arc.read();
            if let Some(ref idx) = inner.store.hnsw_index {
                for _ in 0..n_iterations {
                    let _ = idx.search_ef(&query_vec.data, k, 100);
                }
            }
        });
        let t_hnsw_tight_total = t0.elapsed().as_micros() as f64;

        // Check storage properties
        let is_sq8 = {
            let inner = inner_arc.read();
            if let Some(ref idx) = inner.store.hnsw_index {
                idx.is_sq8()
            } else {
                false
            }
        };

        let mut result = HashMap::new();
        result.insert(
            "search_per_call_us".to_string(),
            t_search_total / n_iterations as f64,
        );
        result.insert(
            "hnsw_per_call_us".to_string(),
            t_hnsw_total / n_iterations as f64,
        );
        result.insert(
            "hnsw_tight_per_call_us".to_string(),
            t_hnsw_tight_total / n_iterations as f64,
        );
        result.insert(
            "overhead_per_call_us".to_string(),
            (t_search_total - t_hnsw_total) / n_iterations as f64,
        );
        result.insert(
            "gil_overhead_per_call_us".to_string(),
            (t_hnsw_total - t_hnsw_tight_total) / n_iterations as f64,
        );
        result.insert("n_iterations".to_string(), n_iterations as f64);
        result.insert("is_sq8".to_string(), if is_sq8 { 1.0 } else { 0.0 });

        Ok(result)
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
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
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
        let all_results: Vec<Result<Vec<(usize, f32, JsonValue)>, _>> = py.detach(|| {
            let inner = self.inner.read();
            inner.store.search_batch_with_metadata(&query_vecs, k, ef)
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

    /// Delete vectors matching a metadata filter.
    ///
    /// Evaluates the filter against all vectors and deletes those that match.
    /// Uses the same MongoDB-style filter syntax as search().
    ///
    /// Args:
    ///     filter (dict): MongoDB-style metadata filter
    ///
    /// Returns:
    ///     int: Number of vectors deleted
    ///
    /// Examples:
    ///     Delete by equality:
    ///
    ///     >>> db.delete_by_filter({"status": "archived"})
    ///     5
    ///
    ///     Delete with comparison operators:
    ///
    ///     >>> db.delete_by_filter({"score": {"$lt": 0.5}})
    ///     3
    ///
    ///     Delete with complex filter:
    ///
    ///     >>> db.delete_by_filter({"$and": [{"type": "draft"}, {"age": {"$gt": 30}}]})
    ///     2
    #[pyo3(signature = (filter))]
    fn delete_by_filter(&self, filter: &Bound<'_, PyDict>) -> PyResult<usize> {
        let parsed_filter = parse_filter(filter)?;

        let mut inner = self.inner.write();

        let result = inner
            .store
            .delete_by_filter(&parsed_filter)
            .map_err(convert_error)?;

        // Invalidate cache since id_to_index changed
        if result > 0 {
            inner.cache_valid = false;
        }

        Ok(result)
    }

    /// Count vectors, optionally filtered by metadata.
    ///
    /// Without a filter, returns total count (same as len(db)).
    /// With a filter, returns count of vectors matching the filter.
    ///
    /// Args:
    ///     filter (dict, optional): MongoDB-style metadata filter
    ///
    /// Returns:
    ///     int: Number of vectors (matching filter if provided)
    ///
    /// Examples:
    ///     Total count:
    ///
    ///     >>> db.count()
    ///     1000
    ///
    ///     Filtered count:
    ///
    ///     >>> db.count(filter={"status": "active"})
    ///     750
    ///
    ///     With comparison operators:
    ///
    ///     >>> db.count(filter={"score": {"$gte": 0.8}})
    ///     250
    #[pyo3(signature = (filter=None))]
    fn count(&self, filter: Option<&Bound<'_, PyDict>>) -> PyResult<usize> {
        let inner = self.inner.read();

        match filter {
            Some(f) => {
                let parsed_filter = parse_filter(f)?;
                Ok(inner.store.count_by_filter(&parsed_filter))
            }
            None => Ok(inner.store.len()),
        }
    }

    /// Update vector, metadata, and/or text for existing ID.
    ///
    /// At least one of vector, metadata, or text must be provided.
    ///
    /// Args:
    ///     id (str): Vector ID to update
    ///     vector (list[float], optional): New vector data
    ///     metadata (dict, optional): New metadata (replaces existing)
    ///     text (str, optional): New text for hybrid search (re-indexed for BM25)
    ///
    /// Raises:
    ///     ValueError: If no update parameters provided
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
    ///     Update text (re-indexes for BM25 search):
    ///
    ///     >>> db.update("doc1", text="New searchable content")
    #[pyo3(signature = (id, vector=None, metadata=None, text=None))]
    fn update(
        &self,
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
        text: Option<String>,
    ) -> PyResult<()> {
        if vector.is_none() && metadata.is_none() && text.is_none() {
            return Err(PyValueError::new_err(
                "update() requires at least one of vector, metadata, or text",
            ));
        }

        let mut inner = self.inner.write();

        // Handle text update - requires re-indexing
        if let Some(ref new_text) = text {
            // Get existing data
            let (existing_vec, existing_meta) = inner.store.get(&id).ok_or_else(|| {
                PyRuntimeError::new_err(format!("Vector with ID '{}' not found", id))
            })?;

            // Determine final vector
            let final_vec = vector.map(Vector::new).unwrap_or(existing_vec);

            // Determine final metadata, incorporating new text
            let mut final_meta = if let Some(m) = metadata {
                pyobject_to_json(m.as_any())?
            } else {
                existing_meta
            };

            // Check for conflict
            if let Some(obj) = final_meta.as_object_mut() {
                if metadata.is_some() && obj.contains_key("text") {
                    return Err(PyValueError::new_err(
                        "Cannot provide both 'text' parameter and 'metadata.text' - use one or the other",
                    ));
                }
                obj.insert("text".to_string(), serde_json::json!(new_text));
            }

            // Re-index text and update vector/metadata
            if inner.store.has_text_search() {
                inner
                    .store
                    .set_with_text(id, final_vec, new_text, final_meta)
                    .map_err(convert_error)?;
            } else {
                // Text search not enabled, just update metadata
                inner
                    .store
                    .set(id, final_vec, final_meta)
                    .map_err(convert_error)?;
            }
            inner.cache_valid = false;
            return Ok(());
        }

        // No text update - use standard update path
        let vector = vector.map(Vector::new);
        let metadata_json = if let Some(m) = metadata {
            Some(pyobject_to_json(m.as_any())?)
        } else {
            None
        };

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

        if let Some((vector, metadata)) = inner.store.get(&id) {
            let mut result = HashMap::new();
            result.insert(
                "id".to_string(),
                id.into_pyobject(py).unwrap().unbind().into(),
            );
            result.insert(
                "vector".to_string(),
                vector.data.into_pyobject(py).unwrap().unbind(),
            );

            let metadata_dict = json_to_pyobject(py, &metadata)?;
            result.insert("metadata".to_string(), metadata_dict);

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Get multiple vectors by ID.
    ///
    /// Batch version of get(). More efficient than calling get() in a loop.
    ///
    /// Args:
    ///     ids (list[str]): List of vector IDs to retrieve
    ///
    /// Returns:
    ///     list[dict | None]: List of results in same order as input.
    ///                        None for IDs that don't exist.
    ///
    /// Examples:
    ///     >>> results = db.get_batch(["doc1", "doc2", "missing"])
    ///     >>> results[0]  # doc1
    ///     {'id': 'doc1', 'vector': [...], 'metadata': {...}}
    ///     >>> results[2]  # missing
    ///     None
    fn get_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
    ) -> PyResult<Vec<Option<HashMap<String, Py<PyAny>>>>> {
        // Release GIL during data loading
        let inner_arc = Arc::clone(&self.inner);
        let fetched: Vec<_> = py.detach(|| {
            let inner = inner_arc.read();
            ids.into_iter()
                .map(|id| {
                    let data = inner.store.get(&id);
                    (id, data)
                })
                .collect()
        });

        // Convert to Python with GIL held
        fetched
            .into_iter()
            .map(|(id, data)| {
                if let Some((vector, metadata)) = data {
                    let mut result = HashMap::new();
                    result.insert(
                        "id".to_string(),
                        id.into_pyobject(py).unwrap().unbind().into(),
                    );
                    result.insert(
                        "vector".to_string(),
                        vector.data.into_pyobject(py).unwrap().unbind(),
                    );
                    result.insert("metadata".to_string(), json_to_pyobject(py, &metadata)?);
                    Ok(Some(result))
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    /// Context manager entry - returns self for `with` statement.
    ///
    /// Examples:
    ///     >>> with omendb.open("./db", dimensions=768) as db:
    ///     ...     db.set([...])
    ///     # Automatically flushed on exit
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit - flushes changes on exit.
    ///
    /// Called automatically when exiting a `with` block.
    /// Flushes pending changes to disk for persistent databases.
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_val: Option<Py<PyAny>>,
        _exc_tb: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)?;
        Ok(false) // Don't suppress exceptions
    }

    /// Get current ef_search value.
    ///
    /// ef_search controls the search quality/speed tradeoff. Higher values
    /// give better recall but slower search.
    ///
    /// Returns:
    ///     int: Current ef_search value (default: 100)
    ///
    /// Examples:
    ///     >>> db.get_ef_search()
    ///     100
    fn get_ef_search(&self) -> usize {
        let inner = self.inner.read();
        inner.store.get_ef_search().unwrap_or(100)
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

    /// Optimize index for cache-efficient search.
    ///
    /// Reorders graph nodes and vectors using BFS traversal to improve memory locality.
    /// Nodes frequently accessed together during search will be stored adjacently,
    /// reducing cache misses and improving QPS by 6-40%.
    ///
    /// Call this after loading data and before querying for best results.
    ///
    /// Returns:
    ///     int: Number of nodes reordered (0 if index empty/not initialized)
    ///
    /// Examples:
    ///     >>> db.set([...])  # Load data
    ///     >>> db.optimize()  # Optimize for search
    ///     >>> db.search(...)  # Faster queries
    fn optimize(&mut self) -> PyResult<usize> {
        let mut inner = self.inner.write();
        let result = inner.store.optimize().map_err(convert_error)?;
        // Invalidate cache since internal indices have been remapped
        inner.cache_valid = false;
        Ok(result)
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

    /// Iterate over all vector IDs (without loading vector data).
    ///
    /// Returns a lazy iterator that yields IDs one at a time.
    /// Memory efficient for large datasets. Use `list(db.ids())` if you need all IDs at once.
    ///
    /// Returns:
    ///     Iterator[str]: Iterator over all vector IDs
    ///
    /// Examples:
    ///     >>> for id in db.ids():
    ///     ...     print(id)
    ///
    ///     >>> # Get as list if needed
    ///     >>> all_ids = list(db.ids())
    ///     >>> len(all_ids)
    ///     1000
    fn ids(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<VectorDatabaseIdIterator>> {
        let borrowed = slf.borrow(py);
        let ids = borrowed.inner.read().store.ids();
        Py::new(
            py,
            VectorDatabaseIdIterator {
                inner: Arc::clone(&borrowed.inner),
                ids,
                index: 0,
            },
        )
    }

    /// Get all items as list of dicts.
    ///
    /// Returns all vectors with their IDs and metadata. Use for export,
    /// migration, or analytics. For large datasets, consider chunked processing.
    ///
    /// Returns:
    ///     list[dict]: List of {"id": str, "vector": list[float], "metadata": dict}
    ///
    /// Examples:
    ///     >>> items = db.items()
    ///     >>> len(items)
    ///     1000
    ///     >>> items[0]
    ///     {'id': 'doc1', 'vector': [0.1, 0.2, ...], 'metadata': {'title': 'Hello'}}
    ///
    ///     # Export to pandas
    ///     >>> import pandas as pd
    ///     >>> df = pd.DataFrame(db.items())
    fn items(&self, py: Python<'_>) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        // Release GIL during data loading
        let inner_arc = Arc::clone(&self.inner);
        let items: Vec<_> = py.detach(|| {
            let inner = inner_arc.read();
            inner.store.items()
        });

        // Convert to Python with GIL held
        items
            .into_iter()
            .map(|(id, vector, metadata)| {
                let mut result = HashMap::new();
                result.insert(
                    "id".to_string(),
                    id.into_pyobject(py).unwrap().unbind().into(),
                );
                result.insert(
                    "vector".to_string(),
                    vector.into_pyobject(py).unwrap().unbind(),
                );
                result.insert("metadata".to_string(), json_to_pyobject(py, &metadata)?);
                Ok(result)
            })
            .collect()
    }

    /// Check if an ID exists in the database.
    ///
    /// Args:
    ///     id (str): Vector ID to check
    ///
    /// Returns:
    ///     bool: True if ID exists and is not deleted
    ///
    /// Examples:
    ///     >>> db.exists("doc1")
    ///     True
    ///     >>> db.exists("nonexistent")
    ///     False
    fn exists(&self, id: String) -> bool {
        let inner = self.inner.read();
        inner.store.contains(&id)
    }

    /// Support `in` operator for checking ID existence.
    ///
    /// Examples:
    ///     >>> "doc1" in db
    ///     True
    fn __contains__(&self, id: String) -> bool {
        let inner = self.inner.read();
        inner.store.contains(&id)
    }

    /// Iteration support - returns list of items.
    ///
    /// Enables `for item in db:` syntax.
    ///
    /// Examples:
    ///     >>> for item in db:
    ///     ...     print(item["id"], item["vector"][:3])
    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<VectorDatabaseIterator>> {
        let borrowed = slf.borrow(py);
        // Get just the IDs (lightweight - ~20 bytes per ID vs ~3KB per 768D vector)
        let ids = borrowed.inner.read().store.ids();
        Py::new(
            py,
            VectorDatabaseIterator {
                inner: Arc::clone(&borrowed.inner),
                ids,
                index: 0,
            },
        )
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
    ///
    ///     Collection handles are cached - same name returns same object:
    ///
    ///     >>> col1 = db.collection("users")
    ///     >>> col2 = db.collection("users")
    ///     >>> col1 is col2  # True - same object
    fn collection(&self, py: Python<'_>, name: String) -> PyResult<Py<VectorDatabase>> {
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

        // Check cache first
        {
            let cache = self.collections_cache.read();
            if let Some(cached) = cache.get(&name) {
                return Ok(cached.clone_ref(py));
            }
        }

        // Not in cache - create new collection
        let mut cache = self.collections_cache.write();

        // Double-check after acquiring write lock
        if let Some(cached) = cache.get(&name) {
            return Ok(cached.clone_ref(py));
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

        let collection_db = VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,
            })),
            path: collection_path.to_string_lossy().to_string(),
            dimensions: self.dimensions,
            is_persistent: true,
            collections_cache: RwLock::new(HashMap::new()),
        };

        // Cache and return
        let py_db = Py::new(py, collection_db)?;
        cache.insert(name, py_db.clone_ref(py));
        Ok(py_db)
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

    // =========================================================================
    // Hybrid Search Methods
    // =========================================================================

    /// Enable text search for hybrid (vector + text) search.
    ///
    /// Note: This is called automatically when using set() with items that have
    /// a `text` field. Only call manually if you need custom buffer_mb config.
    ///
    /// Args:
    ///     buffer_mb (int, optional): Writer buffer size in MB (default: 50)
    ///
    /// Examples:
    ///     >>> db.enable_text_search(buffer_mb=100)  # For high-throughput
    #[pyo3(name = "enable_text_search", signature = (buffer_mb=None))]
    fn enable_text_search(&self, buffer_mb: Option<usize>) -> PyResult<()> {
        let mut inner = self.inner.write();

        let config = buffer_mb.map(|mb| TextSearchConfig {
            writer_buffer_mb: mb,
        });

        inner
            .store
            .enable_text_search_with_config(config)
            .map_err(convert_error)
    }

    /// Check if text search is enabled.
    ///
    /// Returns:
    ///     bool: True if text search is enabled
    fn has_text_search(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_text_search()
    }

    /// Search using text only (BM25 scoring).
    ///
    /// Args:
    ///     query (str): Text query
    ///     k (int): Number of results to return
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score, metadata} sorted by BM25 score descending
    ///
    /// Examples:
    ///     >>> results = db.search_text("machine learning", k=10)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: {r['score']:.4f}, text={r['metadata']['text']}")
    #[pyo3(name = "search_text")]
    fn search_text(&self, py: Python<'_>, query: &str, k: usize) -> PyResult<Vec<Py<PyDict>>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }

        let mut inner = self.inner.write();

        // Auto-flush text index to ensure search sees latest inserts
        if inner.store.has_text_search() {
            inner.store.flush().map_err(convert_error)?;
        }

        let results = inner.store.text_search(query, k).map_err(convert_error)?;

        let mut py_results = Vec::with_capacity(results.len());
        for (id, score) in results {
            let dict = PyDict::new(py);
            dict.set_item("id", id.clone())?;
            dict.set_item("score", score)?;

            // Include metadata for consistency with search_hybrid
            if let Some((_, meta)) = inner.store.get(&id) {
                dict.set_item("metadata", json_to_pyobject(py, &meta)?)?;
            } else {
                dict.set_item("metadata", PyDict::new(py))?;
            }

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
    ///     subscores (bool, optional): Return separate keyword_score and semantic_score (default: False)
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score, metadata} sorted by RRF score descending.
    ///                 When subscores=True, also includes keyword_score and semantic_score.
    ///
    /// Examples:
    ///     >>> results = db.search_hybrid([0.1, 0.2, ...], "machine learning", k=10)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: {r['score']:.4f}")
    ///
    ///     With filter:
    ///     >>> results = db.search_hybrid(vec, "ML", k=10, filter={"category": "tech"})
    ///
    ///     Favor vector similarity (70% vector, 30% text):
    ///     >>> results = db.search_hybrid(vec, "ML", k=10, alpha=0.7)
    ///
    ///     Get separate keyword and semantic scores:
    ///     >>> results = db.search_hybrid(vec, "ML", k=10, subscores=True)
    ///     >>> for r in results:
    ///     ...     print(f"{r['id']}: combined={r['score']:.3f}")
    ///     ...     print(f"  keyword={r.get('keyword_score')}, semantic={r.get('semantic_score')}")
    #[pyo3(name = "search_hybrid", signature = (query_vector, query_text, k, filter=None, alpha=None, rrf_k=None, subscores=None))]
    fn search_hybrid(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        query_text: &str,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
        subscores: Option<bool>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        // Validate inputs
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        if let Some(a) = alpha {
            if !(0.0..=1.0).contains(&a) {
                return Err(PyValueError::new_err(format!(
                    "alpha must be between 0.0 and 1.0, got {}",
                    a
                )));
            }
        }
        if let Some(rrf) = rrf_k {
            if rrf == 0 {
                return Err(PyValueError::new_err("rrf_k must be greater than 0"));
            }
        }

        let query_vec = Vector::new(extract_query_vector(query_vector)?);
        let rust_filter = filter.map(parse_filter).transpose()?;

        let mut inner = self.inner.write();

        // Auto-flush text index to ensure search sees latest inserts
        if inner.store.has_text_search() {
            inner.store.flush().map_err(convert_error)?;
        }

        // Use subscores path when requested
        if subscores.unwrap_or(false) {
            let results = if let Some(f) = rust_filter {
                inner
                    .store
                    .hybrid_search_with_filter_subscores(
                        &query_vec, query_text, k, &f, alpha, rrf_k,
                    )
                    .map_err(convert_error)?
            } else {
                inner
                    .store
                    .hybrid_search_with_subscores(&query_vec, query_text, k, alpha, rrf_k)
                    .map_err(convert_error)?
            };

            let mut py_results = Vec::with_capacity(results.len());
            for (hybrid_result, metadata) in results {
                let dict = PyDict::new(py);
                dict.set_item("id", &hybrid_result.id)?;
                dict.set_item("score", hybrid_result.score)?;
                dict.set_item("metadata", json_to_pyobject(py, &metadata)?)?;

                // Add subscores (None if document only appeared in one search)
                match hybrid_result.keyword_score {
                    Some(score) => dict.set_item("keyword_score", score)?,
                    None => dict.set_item("keyword_score", py.None())?,
                }
                match hybrid_result.semantic_score {
                    Some(score) => dict.set_item("semantic_score", score)?,
                    None => dict.set_item("semantic_score", py.None())?,
                }

                py_results.push(dict.into());
            }
            return Ok(py_results);
        }

        // Standard path without subscores
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
        let collections_dir = base_path.join("collections");
        let omen_path = collections_dir.join(format!("{}.omen", name));
        let wal_path = collections_dir.join(format!("{}.wal", name));

        if !omen_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Collection '{}' not found",
                name
            )));
        }

        // Remove from cache first
        {
            let mut cache = self.collections_cache.write();
            cache.remove(&name);
        }

        // Remove .omen file
        std::fs::remove_file(&omen_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete collection: {}", e)))?;

        // Remove .wal file if it exists
        let _ = std::fs::remove_file(&wal_path);

        Ok(())
    }
}

/// Open or create a vector database.
///
/// All parameters except `path` are optional with sensible defaults.
///
/// Args:
///     path (str): Database directory path, or ":memory:" for in-memory
///     dimensions (int): Vector dimensionality (default: 128, auto-detected on first insert)
///     m (int): HNSW neighbors per node (default: 16, range: 4-64)
///     ef_construction (int): Build quality (default: 100, higher = better graph)
///     ef_search (int): Search quality (default: 100, higher = better recall)
///     quantization (bool|str): Enable quantization (default: None = full precision)
///         - True or "sq8": SQ8 ~4x smaller, ~99% recall (RECOMMENDED)
///         - "rabitq": RaBitQ 4-bit, ~8x smaller, ~98% recall
///         - False/None: Full precision (no quantization)
///     rescore (bool): Rerank with full precision (default: True when quantized)
///     oversample (float): Candidate multiplier for rescoring (default: 3.0)
///     metric (str): Distance metric for similarity search (default: "l2")
///         - "l2" or "euclidean": Euclidean distance (default)
///         - "cosine": Cosine distance (1 - cosine similarity)
///         - "dot" or "ip": Inner product (for MIPS)
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
///     >>> db = omendb.open("./my_vectors", dimensions=768)
///
///     # With SQ8 quantization (4x smaller, similar speed, ~99% recall)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization=True)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization="sq8")
///
///     # With RaBitQ for higher compression (8x smaller, ~98% recall)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization="rabitq")
///
///     # Disable rescore for max speed (~1-3% recall loss)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization=True, rescore=False)
///
///     # Custom oversample factor (default 3.0)
///     >>> db = omendb.open("./vectors", dimensions=768, quantization=True, oversample=5.0)
///
///     # With cosine distance metric
///     >>> db = omendb.open("./vectors", dimensions=768, metric="cosine")
#[pyfunction]
#[pyo3(signature = (path, dimensions=0, m=None, ef_construction=None, ef_search=None, quantization=None, rescore=None, oversample=None, metric=None, config=None))]
fn open(
    path: String,
    dimensions: usize,
    m: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    quantization: Option<&Bound<'_, PyAny>>,
    rescore: Option<bool>,
    oversample: Option<f32>,
    metric: Option<String>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<VectorDatabase> {
    use std::path::{Path, PathBuf};

    // Validate optional params
    if let Some(m_val) = m {
        if !(4..=64).contains(&m_val) {
            return Err(PyValueError::new_err(format!(
                "m must be between 4 and 64, got {}",
                m_val
            )));
        }
    }

    // Parse quantization mode
    let quant_mode = parse_quantization(quantization)?;

    if let (Some(ef_val), Some(m_val)) = (ef_construction, m) {
        if ef_val < m_val {
            return Err(PyValueError::new_err(format!(
                "ef_construction ({}) must be >= m ({})",
                ef_val, m_val
            )));
        }
    }

    // Validate oversample
    if let Some(factor) = oversample {
        if factor < 1.0 {
            return Err(PyValueError::new_err(format!(
                "oversample must be >= 1.0, got {}",
                factor
            )));
        }
    }

    // Validate metric
    if let Some(ref m) = metric {
        match m.to_lowercase().as_str() {
            "l2" | "euclidean" | "cosine" | "dot" | "ip" => {}
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric: '{}'. Valid: l2, euclidean, cosine, dot, ip",
                    m
                )));
            }
        }
    }

    // Resolve effective dimensions (use 128 as default if not specified)
    let effective_dims = if dimensions == 0 { 128 } else { dimensions };

    // Handle :memory: for in-memory database (must check BEFORE path existence checks)
    if path == ":memory:" {
        let options = build_store_options(
            effective_dims,
            m,
            ef_construction,
            ef_search,
            quant_mode.clone(),
            rescore,
            oversample,
            metric.as_deref(),
        )?;

        let store = options
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: true,
            })),
            path,
            dimensions: effective_dims,
            is_persistent: false,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    let db_path = Path::new(&path);
    // Compute .omen path by appending extension (preserves full filename)
    let omen_path = if db_path.extension().is_some_and(|ext| ext == "omen") {
        db_path.to_path_buf()
    } else {
        let mut omen = db_path.as_os_str().to_os_string();
        omen.push(".omen");
        PathBuf::from(omen)
    };

    // Check if this is a directory (persistent storage) or .omen file exists
    if db_path.is_dir() || omen_path.exists() || !db_path.exists() {
        let mut options = build_store_options(
            effective_dims,
            m,
            ef_construction,
            ef_search,
            quant_mode.clone(),
            rescore,
            oversample,
            metric.as_deref(),
        )?;

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
        }

        // Check if enabling quantization on existing non-empty database
        if db_path.exists() && quant_mode.is_some() {
            let existing = VectorStore::open(&path).map_err(convert_error)?;
            if !existing.is_empty() {
                return Err(PyValueError::new_err(
                    "Cannot enable quantization on existing database. Create a new database with quantization.",
                ));
            }
        }

        // Open with options
        let store = options.open(&path).map_err(convert_error)?;

        return Ok(VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner {
                store,
                index_to_id_cache: HashMap::new(),
                cache_valid: false,
            })),
            path,
            dimensions: effective_dims,
            is_persistent: true,
            collections_cache: RwLock::new(HashMap::new()),
        });
    }

    // Fallback: create new in-memory database with configuration
    let options = build_store_options(
        effective_dims,
        m,
        ef_construction,
        ef_search,
        quant_mode,
        rescore,
        oversample,
        metric.as_deref(),
    )?;

    let store = options
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to create store: {}", e)))?;

    Ok(VectorDatabase {
        inner: Arc::new(RwLock::new(VectorDatabaseInner {
            store,
            index_to_id_cache: HashMap::new(),
            cache_valid: true,
        })),
        path,
        dimensions: effective_dims,
        is_persistent: false,
        collections_cache: RwLock::new(HashMap::new()),
    })
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
            sub_filters.push(parse_filter(sub_dict)?);
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
            sub_filters.push(parse_filter(sub_dict)?);
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
                    "$eq" => {
                        let json_value = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Eq(key_str.clone(), json_value));
                    }
                    "$ne" => {
                        let json_value = pyobject_to_json(&op_value)?;
                        filters.push(MetadataFilter::Ne(key_str.clone(), json_value));
                    }
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

/// Parsed batch item with optional text for hybrid search
struct ParsedItem {
    id: String,
    vector: Vector,
    metadata: JsonValue,
    text: Option<String>,
}

// Helper: Parse batch items from a list of dicts, including optional text field
fn parse_batch_items_with_text(items: &Bound<'_, PyList>) -> PyResult<Vec<ParsedItem>> {
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

        // Handle optional text field for hybrid search
        // Text is both indexed for BM25 AND stored in metadata["text"]
        let text: Option<String> = dict
            .get_item("text")?
            .map(|t| t.extract())
            .transpose()
            .map_err(|_| {
                PyValueError::new_err(format!("Item '{}': 'text' must be a string", id))
            })?;

        // Auto-store text in metadata for retrieval
        if let Some(ref text_str) = text {
            if let Some(obj) = metadata_json.as_object_mut() {
                // Check for conflict
                if obj.contains_key("text") {
                    return Err(PyValueError::new_err(format!(
                        "Item '{}': cannot have both 'text' field and 'metadata.text' - use one or the other",
                        id
                    )));
                }
                obj.insert("text".to_string(), serde_json::json!(text_str));
            }
        }

        batch.push(ParsedItem {
            id,
            vector: Vector::new(vector_data),
            metadata: metadata_json,
            text,
        });
    }

    Ok(batch)
}

/// Helper: Convert Python object to serde_json::Value
fn pyobject_to_json(obj: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    // Check None first (fast path)
    if obj.is_none() {
        Ok(JsonValue::Null)
    // Check bool BEFORE int/float - Python bool is subclass of int (True == 1, False == 0)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::Number::from_f64(f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null))
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
#[allow(clippy::useless_conversion)]
fn json_to_pyobject(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok((*b).into_pyobject(py).unwrap().to_owned().unbind().into()),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().unbind().into())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().unbind().into())
            } else {
                Ok(py.None())
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

/// Benchmark SQ8 distance via VectorStorage (to test enum matching overhead)
#[pyfunction]
#[pyo3(name = "_bench_sq8_via_storage")]
fn bench_sq8_via_storage(
    py: Python<'_>,
    dim: usize,
    n_vectors: usize,
    n_iterations: usize,
) -> PyResult<f64> {
    use omendb_core::vector::hnsw::VectorStorage;
    use rand::Rng;
    use std::hint::black_box;

    let elapsed_ns = py.detach(|| {
        let mut rng = ::rand::thread_rng();

        // Create VectorStorage with SQ8
        let mut storage = VectorStorage::new_sq8_quantized(dim);

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        for v in &vectors {
            storage.insert(v.clone()).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();

        // Warmup
        for id in 0..n_vectors.min(100) {
            black_box(storage.distance_asymmetric_l2(&query, id as u32));
        }

        // Benchmark - use black_box to prevent dead code elimination
        let start = Instant::now();
        let mut total = 0.0f32;
        for _ in 0..n_iterations {
            for id in 0..n_vectors {
                if let Some(d) = storage.distance_asymmetric_l2(&query, id as u32) {
                    total += black_box(d);
                }
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64 / (n_iterations * n_vectors) as f64;

        // Ensure total is used
        black_box(total);
        elapsed_ns
    });
    Ok(elapsed_ns)
}

/// Benchmark SQ8 L2 decomposed distance (the actual path used in search)
#[pyfunction]
#[pyo3(name = "_bench_sq8_decomposed")]
fn bench_sq8_decomposed(
    py: Python<'_>,
    dim: usize,
    n_vectors: usize,
    n_iterations: usize,
) -> PyResult<f64> {
    use omendb_core::distance::norm_squared;
    use omendb_core::vector::hnsw::VectorStorage;
    use rand::Rng;
    use std::hint::black_box;

    let elapsed_ns = py.detach(|| {
        let mut rng = ::rand::thread_rng();

        // Create VectorStorage with SQ8
        let mut storage = VectorStorage::new_sq8_quantized(dim);

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        for v in &vectors {
            storage.insert(v.clone()).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let query_norm = norm_squared(&query);

        // Warmup
        for id in 0..n_vectors.min(100) {
            black_box(storage.distance_l2_decomposed(&query, query_norm, id as u32));
        }

        // Benchmark L2 decomposed path
        let start = Instant::now();
        let mut total = 0.0f32;
        for _ in 0..n_iterations {
            for id in 0..n_vectors {
                if let Some(d) = storage.distance_l2_decomposed(&query, query_norm, id as u32) {
                    total += black_box(d);
                }
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64 / (n_iterations * n_vectors) as f64;

        // Ensure total is used
        black_box(total);
        elapsed_ns
    });
    Ok(elapsed_ns)
}

/// Benchmark SQ8 distance through a closure (simulates HNSW search pattern)
#[pyfunction]
#[pyo3(name = "_bench_sq8_via_closure")]
fn bench_sq8_via_closure(
    py: Python<'_>,
    dim: usize,
    n_vectors: usize,
    n_iterations: usize,
) -> PyResult<f64> {
    use omendb_core::distance::norm_squared;
    use omendb_core::vector::hnsw::VectorStorage;
    use rand::Rng;
    use std::hint::black_box;

    let elapsed_ns = py.detach(|| {
        let mut rng = ::rand::thread_rng();

        // Create SQ8 storage
        let mut storage = VectorStorage::new_sq8_quantized(dim);

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        for v in vectors {
            storage.insert(v).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let query_norm = norm_squared(&query);

        // Warmup
        for id in 0..n_vectors.min(100) {
            black_box(storage.distance_l2_decomposed(&query, query_norm, id as u32));
        }

        // Benchmark - call through closure like HNSW search does
        let storage_ref = &storage;
        let query_ref = &query;
        let distance_fn = |id: u32| -> Option<f32> {
            storage_ref.distance_l2_decomposed(query_ref, query_norm, id)
        };

        let start = Instant::now();
        let mut total = 0.0f32;
        for _ in 0..n_iterations {
            for id in 0..n_vectors {
                if let Some(d) = distance_fn(id as u32) {
                    total += black_box(d);
                }
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64 / (n_iterations * n_vectors) as f64;

        black_box(total);
        elapsed_ns
    });
    Ok(elapsed_ns)
}

/// Benchmark FP32 L2 decomposed distance for comparison
#[pyfunction]
#[pyo3(name = "_bench_fp32_decomposed")]
fn bench_fp32_decomposed(
    py: Python<'_>,
    dim: usize,
    n_vectors: usize,
    n_iterations: usize,
) -> PyResult<f64> {
    use omendb_core::distance::norm_squared;
    use omendb_core::vector::hnsw::VectorStorage;
    use rand::Rng;
    use std::hint::black_box;

    let elapsed_ns = py.detach(|| {
        let mut rng = ::rand::thread_rng();

        // Create VectorStorage with FP32
        let mut storage = VectorStorage::new_full_precision(dim);

        // Insert vectors
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect();

        for v in &vectors {
            storage.insert(v.clone()).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        let query_norm = norm_squared(&query);

        // Warmup
        for id in 0..n_vectors.min(100) {
            black_box(storage.distance_l2_decomposed(&query, query_norm, id as u32));
        }

        // Benchmark L2 decomposed path
        let start = Instant::now();
        let mut total = 0.0f32;
        for _ in 0..n_iterations {
            for id in 0..n_vectors {
                if let Some(d) = storage.distance_l2_decomposed(&query, query_norm, id as u32) {
                    total += black_box(d);
                }
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64 / (n_iterations * n_vectors) as f64;

        // Ensure total is used
        black_box(total);
        elapsed_ns
    });
    Ok(elapsed_ns)
}

#[pymodule]
fn omendb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(bench_sq8_via_storage, m)?)?;
    m.add_function(wrap_pyfunction!(bench_sq8_decomposed, m)?)?;
    m.add_function(wrap_pyfunction!(bench_sq8_via_closure, m)?)?;
    m.add_function(wrap_pyfunction!(bench_fp32_decomposed, m)?)?;
    m.add_class::<VectorDatabase>()?;
    m.add_class::<VectorDatabaseIdIterator>()?;
    Ok(())
}
