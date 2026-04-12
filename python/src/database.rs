use crate::conversions::{call_embedding_fn, convert_error, json_to_pyobject, pyobject_to_json};
use crate::iterators::{VectorDatabaseIdIterator, VectorDatabaseIterator};
use crate::parsing::parse_batch_items_with_text;
use crate::parsing::parse_multi_vec_items;
use omendb_lib::catalog::{
    CollectionSchema, DenseSchema, GraphSchema, GraphTemporalMode, MultiSchema, QuantizationMode,
    SparseSchema, TextSchema,
};
use omendb_lib::vector::{Vector, VectorStore};
use parking_lot::RwLock;
use pyo3::conversion::IntoPyObject;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Py;
use std::collections::HashMap;
use std::sync::Arc;

fn closed_database_error() -> pyo3::PyErr {
    PyRuntimeError::new_err("database is closed")
}

fn dense_schema_to_pydict(py: Python<'_>, dense: &DenseSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("dim", dense.dim)?;
    dict.set_item(
        "quantization",
        match dense.quantization {
            QuantizationMode::None => "none",
            QuantizationMode::Sq8 => "sq8",
        },
    )?;
    dict.set_item("mutable_index", "hnsw")?;
    dict.set_item("frozen_index", "hnsw")?;
    Ok(dict.into())
}

fn sparse_schema_to_pydict(py: Python<'_>, sparse: &SparseSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("index_kind", "inverted_exact")?;
    dict.set_item("max_nonzero", sparse.max_nonzero)?;
    Ok(dict.into())
}

fn multi_schema_to_pydict(py: Python<'_>, multi: &MultiSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("token_dim", multi.token_dim)?;
    dict.set_item("encoder", "muvera")?;
    dict.set_item("repetitions", multi.repetitions)?;
    dict.set_item("partition_bits", multi.partition_bits)?;
    dict.set_item("d_proj", multi.d_proj)?;
    dict.set_item("seed", multi.seed)?;
    dict.set_item("max_tokens", multi.max_tokens)?;
    dict.set_item("pool_factor", multi.pool_factor)?;
    Ok(dict.into())
}

fn text_schema_to_pydict(py: Python<'_>, text: &TextSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "tokenizer",
        format!("{:?}", text.tokenizer).to_lowercase(),
    )?;
    dict.set_item("writer_buffer_mb", text.writer_buffer_mb)?;
    Ok(dict.into())
}

fn graph_schema_to_pydict(py: Python<'_>, graph: &GraphSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("enabled", graph.enabled)?;
    dict.set_item(
        "temporal",
        match graph.temporal {
            GraphTemporalMode::None => "none",
            GraphTemporalMode::ValidAt => "valid_at",
            GraphTemporalMode::BiTemporal => "bi_temporal",
        },
    )?;
    dict.set_item("provenance", graph.provenance)?;
    Ok(dict.into())
}

fn schema_to_pydict(py: Python<'_>, schema: &CollectionSchema) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &schema.name)?;
    dict.set_item("metric", schema.metric.canonical_name())?;
    dict.set_item(
        "dense",
        schema
            .dense
            .as_ref()
            .map(|dense| dense_schema_to_pydict(py, dense))
            .transpose()?,
    )?;
    dict.set_item(
        "sparse",
        schema
            .sparse
            .as_ref()
            .map(|sparse| sparse_schema_to_pydict(py, sparse))
            .transpose()?,
    )?;
    dict.set_item(
        "multi",
        schema
            .multi
            .as_ref()
            .map(|multi| multi_schema_to_pydict(py, multi))
            .transpose()?,
    )?;
    dict.set_item(
        "text",
        schema
            .text
            .as_ref()
            .map(|text| text_schema_to_pydict(py, text))
            .transpose()?,
    )?;
    dict.set_item(
        "graph",
        schema
            .graph
            .as_ref()
            .map(|graph| graph_schema_to_pydict(py, graph))
            .transpose()?,
    )?;
    Ok(dict.into())
}

pub(crate) struct VectorDatabaseInner {
    pub(crate) store: Option<VectorStore>,
}

impl VectorDatabaseInner {
    pub(crate) fn store(&self) -> PyResult<&VectorStore> {
        self.store
            .as_ref()
            .ok_or_else(closed_database_error)
    }

    pub(crate) fn store_mut(&mut self) -> PyResult<&mut VectorStore> {
        self.store
            .as_mut()
            .ok_or_else(closed_database_error)
    }
}

impl VectorDatabase {
    pub(crate) fn live_is_multi_vector(&self) -> PyResult<bool> {
        let inner = self.inner.read();
        Ok(inner.store()?.is_multi_vector())
    }

    pub(crate) fn live_dimensions(&self) -> PyResult<usize> {
        let inner = self.inner.read();
        let store = inner.store()?;

        Ok(if store.is_multi_vector() {
            store
                .token_dimension()
                .unwrap_or(store.dimensions())
        } else {
            store.dimensions()
        })
    }

}

/// High-performance embedded vector database.
///
/// Provides fast similarity search using HNSW indexing with:
/// - ~19,000 QPS @ 10K vectors with 100% recall
/// - 20,000-28,000 vec/s insert throughput
/// - SQ8 quantization (4x compression, ~99% recall)
/// - ACORN-1 filtered search (37.79x speedup)
/// - Multi-vector support for ColBERT-style retrieval
///
/// Auto-persists to disk for seamless data durability.
///
/// Supports context manager protocol for automatic cleanup:
///
/// ```python
/// with omendb.create(":memory:", {"dense": {"dim": 768}}) as db:
///     db.set([...])
/// # Automatically flushed on exit
/// ```
#[pyclass]
pub struct VectorDatabase {
    pub(crate) inner: Arc<RwLock<VectorDatabaseInner>>,
    pub(crate) path: String,
    pub(crate) is_persistent: bool,
    /// Optional embedding function: (list[str]) -> ndarray[n, dim]
    pub(crate) embedding_fn: Option<Py<PyAny>>,
    /// Cache of open collection handles (same name = same object)
    pub(crate) collections_cache: RwLock<HashMap<String, Py<VectorDatabase>>>,
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
    ///     - Batch inserts skip the WAL for performance. Data is not durable
    ///       until flush() is called (or the context manager exits).
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
    #[pyo3(name = "set", signature = (id_or_items=None, vector=None, metadata=None, *, ids=None, vectors=None, metadatas=None, document=None, documents=None))]
    fn set_vectors(
        &self,
        py: Python<'_>,
        id_or_items: Option<&Bound<'_, PyAny>>,
        vector: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
        ids: Option<Vec<String>>,
        vectors: Option<Vec<Vec<f32>>>,
        metadatas: Option<&Bound<'_, PyList>>,
        document: Option<String>,
        documents: Option<Vec<String>>,
    ) -> PyResult<usize> {
        // Handle kwargs batch with documents: set(ids=["a"], documents=["hello"])
        if let (Some(ids), Some(docs)) = (&ids, &documents) {
            if ids.len() != docs.len() {
                return Err(PyValueError::new_err(format!(
                    "ids and documents must have same length: {} vs {}",
                    ids.len(),
                    docs.len()
                )));
            }
            if vectors.is_some() {
                return Err(PyValueError::new_err(
                    "Cannot provide both vectors and documents",
                ));
            }
            let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                PyValueError::new_err(
                    "No embedding function configured. Pass embedding_fn to open() or provide vectors directly.",
                )
            })?;
            let embedded = call_embedding_fn(py, emb_fn, docs)?;
            let batch: Vec<_> = ids
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    let meta = metadatas
                        .and_then(|m| m.get_item(i).ok())
                        .map(|m| pyobject_to_json(&m))
                        .transpose()?
                        .unwrap_or_else(|| serde_json::json!({}));
                    Ok((id.clone(), Vector::new(embedded[i].clone()), meta))
                })
                .collect::<PyResult<Vec<_>>>()?;

            let inner_arc = Arc::clone(&self.inner);
            let result = py.detach(|| {
                let mut inner = inner_arc.write();
                inner.store_mut()?.set_batch(batch).map_err(convert_error)
            })?;
            return Ok(result.len());
        }

        // Handle kwargs batch format with vectors: set(ids=["a"], vectors=[[...]])
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
                inner.store_mut()?.set_batch(batch).map_err(convert_error)
            })?;
            return Ok(result.len());
        }

        // Handle single item: set("id", [...], {...}) or set("id", document="hello")
        if let Some(id_or_items) = id_or_items {
            if let Ok(id_str) = id_or_items.extract::<String>() {
                // Validate: can't have both vector and document
                if vector.is_some() && document.is_some() {
                    return Err(PyValueError::new_err(
                        "Cannot provide both vector and document",
                    ));
                }

                let vec_data = if let Some(doc) = &document {
                    // Embed the document
                    let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                        PyValueError::new_err(
                            "No embedding function configured. Pass embedding_fn to open() or provide vectors directly.",
                        )
                    })?;
                    let embedded = call_embedding_fn(py, emb_fn, &[doc.clone()])?;
                    embedded.into_iter().next().unwrap()
                } else {
                    vector.ok_or_else(|| {
                        PyValueError::new_err("vector or document required when id is a string")
                    })?
                };

                let meta = metadata
                    .map(|m| pyobject_to_json(m.as_any()))
                    .transpose()?
                    .unwrap_or_else(|| serde_json::json!({}));

                let mut inner = self.inner.write();
                inner
                    .store_mut()?
                    .set(&id_str, Vector::new(vec_data), meta)
                    .map_err(convert_error)?;
                return Ok(1);
            }

            // Handle batch: set([{...}, {...}])
            if let Ok(items) = id_or_items.cast::<PyList>() {
                // Multi-vector store: use "vectors" key
        if self.live_is_multi_vector()? {
                    let parsed = parse_multi_vec_items(items)?;
                    let inner_arc = Arc::clone(&self.inner);
                    let count = py.detach(move || -> PyResult<usize> {
                        let mut inner = inner_arc.write();
                        let store = inner.store_mut()?;
                        for item in &parsed {
                            store
                                .store(&item.id, item.vectors.clone(), item.metadata.clone())
                                .map_err(convert_error)?;
                        }
                        Ok(parsed.len())
                    })?;
                    return Ok(count);
                }

                // Single-vector store: use "vector" or "document" key
                let mut parsed = parse_batch_items_with_text(items)?;

                // Embed any items that have documents instead of vectors
                let doc_indices: Vec<usize> = parsed
                    .iter()
                    .enumerate()
                    .filter(|(_, item)| item.document.is_some())
                    .map(|(i, _)| i)
                    .collect();

                if !doc_indices.is_empty() {
                    let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                        PyValueError::new_err(
                            "No embedding function configured. Pass embedding_fn to open() or provide vectors directly.",
                        )
                    })?;
                    let docs: Vec<String> = doc_indices
                        .iter()
                        .map(|&i| parsed[i].document.clone().unwrap())
                        .collect();
                    let embedded = call_embedding_fn(py, emb_fn, &docs)?;
                    for (j, &idx) in doc_indices.iter().enumerate() {
                        parsed[idx].vector = Vector::new(embedded[j].clone());
                    }
                }

                // Check if any items have text
                let has_text = parsed.iter().any(|item| item.text.is_some());

                // Release GIL during batch insert
                let inner_arc = Arc::clone(&self.inner);
                let count = py.detach(move || -> PyResult<usize> {
                    let mut inner = inner_arc.write();
                    let store = inner.store_mut()?;

                    // Auto-enable text search if text field is present
                    if has_text && !store.has_text_search() {
                        store.enable_text_search().map_err(convert_error)?;
                    }

                    // Insert items - use batch path when no text for performance
                    let results = if has_text {
                        // Slow path: items with text must be inserted individually
                        let mut results = Vec::with_capacity(parsed.len());
                        for item in parsed {
                            let result = if let Some(text) = item.text {
                                store
                                    .set_with_text(&item.id, item.vector, &text, item.metadata)
                                    .map_err(convert_error)?
                            } else {
                                store
                                    .set(&item.id, item.vector, item.metadata)
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
                        store.set_batch(batch).map_err(convert_error)?
                    };

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

    /// Delete vectors by ID.
    ///
    /// Accepts either a single ID string or a list of IDs.
    ///
    /// Examples:
    ///     >>> db.delete("doc1")  # Single ID
    ///     1
    ///
    ///     >>> db.delete(["doc1", "doc2"])  # Multiple IDs
    ///     2
    ///
    ///     >>> db.delete(["nonexistent"])  # Silently skips missing IDs
    ///     0
    fn delete(&self, ids: &Bound<'_, PyAny>) -> PyResult<usize> {
        // Try single string first
        if let Ok(single_id) = ids.extract::<String>() {
            let inner = self.inner.write();
            return inner
                .store()?
                .delete_batch(&[single_id])
                .map_err(convert_error);
        }
        // Fall back to list of strings
        let id_vec: Vec<String> = ids.extract()?;
        let inner = self.inner.write();
        inner.store()?.delete_batch(&id_vec).map_err(convert_error)
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
        let parsed_filter = crate::filters::parse_filter(filter)?;

        let inner = self.inner.write();
        inner.store()?.delete_by_filter(&parsed_filter).map_err(convert_error)
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
                let parsed_filter = crate::filters::parse_filter(f)?;
                Ok(inner.store()?.count_by_filter(&parsed_filter))
            }
            None => Ok(inner.store()?.len()),
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
        let store = inner.store_mut()?;

        // Handle text update - requires re-indexing
        if let Some(ref new_text) = text {
            // Get existing data
            let (existing_vec, existing_meta) = store.get(&id).ok_or_else(|| {
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
            if store.has_text_search() {
                store
                    .set_with_text(&id, final_vec, new_text, final_meta)
                    .map_err(convert_error)?;
            } else {
                // Text search not enabled, just update metadata
                store
                    .set(&id, final_vec, final_meta)
                    .map_err(convert_error)?;
            }
            return Ok(());
        }

        // No text update - use standard update path
        let vector = vector.map(Vector::new);
        let metadata_json = if let Some(m) = metadata {
            Some(pyobject_to_json(m.as_any())?)
        } else {
            None
        };

        store.update(&id, vector, metadata_json).map_err(convert_error)
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

        if let Some((vector, metadata)) = inner.store()?.get(&id) {
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
        let fetched: Vec<_> = py.detach(|| -> PyResult<Vec<_>> {
            let inner = inner_arc.read();
            let store = inner.store()?;
            Ok(ids
                .into_iter()
                .map(|id| {
                    let data = store.get(&id);
                    (id, data)
                })
                .collect())
        })?;

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
    ///     >>> with omendb.create(":memory:", {"dense": {"dim": 768}}) as db:
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
        inner.store_mut()?.flush().map_err(convert_error)?;
        Ok(false) // Don't suppress exceptions
    }

    /// ef_search property - controls the search quality/speed tradeoff.
    ///
    /// Higher values give better recall but slower search.
    ///
    /// Examples:
    ///     >>> db.ef_search = 200  # High quality
    ///     >>> print(db.ef_search)
    ///     200
    #[getter]
    fn ef_search(&self) -> PyResult<usize> {
        let inner = self.inner.read();
        Ok(inner.store()?.ef_search())
    }

    #[setter]
    fn set_ef_search(&mut self, value: usize) -> PyResult<()> {
        let inner = self.inner.write();
        inner.store()?.set_ef_search(value);
        Ok(())
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
        Ok(inner.store()?.len())
    }

    /// Boolean truth value - True if database is non-empty.
    ///
    /// Examples:
    ///     >>> if db:
    ///     ...     print("has data")
    fn __bool__(&self) -> PyResult<bool> {
        let inner = self.inner.read();
        Ok(inner.store()?.len() > 0)
    }

    /// Get database dimensions.
    ///
    /// Returns:
    ///     int: Dimensionality of vectors in this database
    #[getter]
    fn dimensions(&self) -> PyResult<usize> {
        Ok(self.live_dimensions()?)
    }

    /// Whether this is a multi-vector store (for ColBERT-style retrieval).
    #[getter]
    fn is_multi_vector(&self) -> PyResult<bool> {
        Ok(self.live_is_multi_vector()?)
    }

    /// The embedding function, if configured.
    #[getter]
    fn embedding_fn(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.embedding_fn.as_ref().map(|f| f.clone_ref(py))
    }

    /// Check if database is empty.
    fn is_empty(&self) -> PyResult<bool> {
        let inner = self.inner.read();
        Ok(inner.store()?.is_empty())
    }

    /// Get comprehensive database diagnostics.
    ///
    /// Returns:
    ///     dict: Store info including counts, memory, config, and segment details
    ///
    /// Examples:
    ///     >>> info = db.info()
    ///     >>> info["vector_count"]
    ///     10000
    ///     >>> info["total_memory_bytes"]
    ///     5242880
    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.read();
        let info = inner.store()?.info();
        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("vector_count", info.vector_count)?;
        dict.set_item("deleted_count", info.deleted_count)?;
        dict.set_item("dimensions", info.dimensions)?;
        dict.set_item("metric", info.metric.canonical_name())?;
        dict.set_item("frozen_segment_count", info.frozen_segment_count)?;
        dict.set_item("mutable_segment_vectors", info.mutable_segment_vectors)?;
        dict.set_item("vector_bytes", info.vector_bytes)?;
        dict.set_item("graph_bytes", info.graph_bytes)?;
        dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
        dict.set_item("wal_entries", info.wal_entries)?;
        dict.set_item("is_persistent", info.is_persistent)?;
        dict.set_item("hnsw_m", info.hnsw_m)?;
        dict.set_item("hnsw_ef_construction", info.hnsw_ef_construction)?;
        dict.set_item("hnsw_ef_search", info.hnsw_ef_search)?;
        dict.set_item("quantization", info.quantization)?;
        dict.set_item("segment_capacity", info.segment_capacity)?;
        dict.set_item("schema", schema_to_pydict(py, &info.schema)?)?;

        Ok(dict.into())
    }

    /// Get the authoritative collection schema for this database.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.read();
        schema_to_pydict(py, &inner.store()?.schema())
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
        let ids = borrowed.inner.read().store()?.ids();
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
        let items: Vec<_> = py.detach(|| -> PyResult<Vec<_>> {
            let inner = inner_arc.read();
            Ok(inner.store()?.items())
        })?;

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
    fn exists(&self, id: String) -> PyResult<bool> {
        let inner = self.inner.read();
        Ok(inner.store()?.contains(&id))
    }

    /// Support `in` operator for checking ID existence.
    ///
    /// Examples:
    ///     >>> "doc1" in db
    ///     True
    fn __contains__(&self, id: String) -> PyResult<bool> {
        let inner = self.inner.read();
        Ok(inner.store()?.contains(&id))
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
        let ids = borrowed.inner.read().store()?.ids();
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
        dict.set_item("dimensions", self.live_dimensions()?)?;
        dict.set_item("count", inner.store()?.len())?;
        dict.set_item("path", &self.path)?;
        Ok(dict.into())
    }
}
