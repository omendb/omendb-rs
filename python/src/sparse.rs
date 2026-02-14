use crate::conversions::{convert_error, extract_query_vector, json_to_pyobject, pyobject_to_json};
use crate::database::VectorDatabase;
use crate::filters::parse_filter;
use omendb_lib::vector::sparse::SparseVector;
use omendb_lib::vector::Vector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::sync::Arc;

/// Parse a sparse vector from Python arguments.
///
/// Accepts either:
/// - (indices, values) as two lists
/// - a dict mapping dimension -> weight
fn parse_sparse_vector(
    indices_or_dict: &Bound<'_, PyAny>,
    values: Option<Vec<f32>>,
) -> PyResult<SparseVector> {
    // Try dict first: {dim: weight, ...}
    if let Ok(dict) = indices_or_dict.downcast::<PyDict>() {
        let mut pairs = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let dim: u32 = k.extract()?;
            let weight: f32 = v.extract()?;
            pairs.push((dim, weight));
        }
        return Ok(SparseVector::from_pairs(pairs));
    }

    // Otherwise: (indices, values) as two lists
    let indices: Vec<u32> = indices_or_dict.extract()?;
    let values = values.ok_or_else(|| {
        PyValueError::new_err(
            "values required when indices is a list. Use set_sparse(id, indices, values, metadata) or set_sparse(id, {dim: weight}, metadata)",
        )
    })?;

    Ok(SparseVector::from_pairs(
        indices.into_iter().zip(values).collect(),
    ))
}

#[pymethods]
impl VectorDatabase {
    /// Enable sparse vector indexing for SPLADE-style retrieval.
    ///
    /// Must be called before set_sparse() or sparse_search().
    ///
    /// Examples:
    ///     >>> db.enable_sparse()
    ///     >>> db.has_sparse()
    ///     True
    fn enable_sparse(&self) -> PyResult<()> {
        let mut inner = self.inner.write();
        inner.store.enable_sparse();
        Ok(())
    }

    /// Check if sparse indexing is enabled.
    ///
    /// Returns:
    ///     bool: True if sparse indexing is enabled
    fn has_sparse(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_sparse()
    }

    /// Insert or update a sparse vector.
    ///
    /// The sparse vector is indexed for dot-product search. If the ID already
    /// exists, the old sparse vector is replaced.
    ///
    /// Args:
    ///     id (str): Unique identifier
    ///     indices_or_dict: Either a list of dimension indices or a dict {dim: weight}
    ///     values (list[float], optional): Weights for each dimension (required if indices is a list)
    ///     metadata (dict, optional): Arbitrary metadata
    ///
    /// Examples:
    ///     With indices and values:
    ///     >>> db.set_sparse("doc1", [10, 42, 100], [0.5, 1.2, 0.8], {"title": "Hello"})
    ///
    ///     With dict:
    ///     >>> db.set_sparse("doc1", {10: 0.5, 42: 1.2, 100: 0.8}, metadata={"title": "Hello"})
    #[pyo3(name = "set_sparse", signature = (id, indices_or_dict, values=None, metadata=None))]
    fn set_sparse(
        &self,
        id: &str,
        indices_or_dict: &Bound<'_, PyAny>,
        values: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let sparse = parse_sparse_vector(indices_or_dict, values)?;
        let meta = metadata
            .map(|m| pyobject_to_json(m.as_any()))
            .transpose()?
            .unwrap_or_else(|| serde_json::json!({}));

        let mut inner = self.inner.write();
        inner
            .store
            .set_sparse(id, sparse, meta)
            .map_err(convert_error)
    }

    /// Insert or update both dense and sparse vectors together.
    ///
    /// Args:
    ///     id (str): Unique identifier
    ///     vector (list[float]): Dense vector
    ///     indices_or_dict: Sparse vector indices (list) or weights dict
    ///     values (list[float], optional): Sparse weights (required if indices is a list)
    ///     metadata (dict, optional): Arbitrary metadata
    ///
    /// Examples:
    ///     >>> db.set_hybrid_sparse("doc1", [0.1, 0.2, 0.3],
    ///     ...     {10: 0.5, 42: 1.2}, metadata={"title": "Hello"})
    #[pyo3(name = "set_hybrid_sparse", signature = (id, vector, indices_or_dict, values=None, metadata=None))]
    fn set_hybrid_sparse(
        &self,
        id: &str,
        vector: Vec<f32>,
        indices_or_dict: &Bound<'_, PyAny>,
        values: Option<Vec<f32>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let sparse = parse_sparse_vector(indices_or_dict, values)?;
        let meta = metadata
            .map(|m| pyobject_to_json(m.as_any()))
            .transpose()?
            .unwrap_or_else(|| serde_json::json!({}));

        let mut inner = self.inner.write();
        inner
            .store
            .set_hybrid_sparse(id, Vector::new(vector), sparse, meta)
            .map_err(convert_error)
    }

    /// Search sparse vectors by dot product similarity.
    ///
    /// Returns results sorted by dot product score (higher = more similar).
    ///
    /// Args:
    ///     indices_or_dict: Query sparse vector as indices list or dict
    ///     values (list[float], optional): Query weights (required if indices is a list)
    ///     k (int): Number of results to return (default: 10)
    ///     filter (dict, optional): MongoDB-style metadata filter
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score, metadata} sorted by score descending
    ///
    /// Examples:
    ///     >>> results = db.sparse_search({10: 0.5, 42: 1.2}, k=5)
    ///     >>> results = db.sparse_search([10, 42], [0.5, 1.2], k=5)
    #[pyo3(name = "sparse_search", signature = (indices_or_dict, values=None, k=10, filter=None))]
    fn sparse_search(
        &self,
        py: Python<'_>,
        indices_or_dict: &Bound<'_, PyAny>,
        values: Option<Vec<f32>>,
        k: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }

        let query = parse_sparse_vector(indices_or_dict, values)?;
        let rust_filter = filter.map(parse_filter).transpose()?;

        let inner_arc = Arc::clone(&self.inner);
        let results = py.detach(|| {
            let inner = inner_arc.read();
            inner
                .store
                .sparse_search(&query, k, rust_filter.as_ref())
                .map_err(convert_error)
        })?;

        // Convert: distance is -dot_product, so score = -distance
        let mut py_results = Vec::with_capacity(results.len());
        for r in &results {
            let dict = PyDict::new(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", -r.distance)?;
            dict.set_item("metadata", json_to_pyobject(py, &r.metadata)?)?;
            py_results.push(dict.into());
        }

        Ok(py_results)
    }

    /// Hybrid dense + sparse search with Reciprocal Rank Fusion (RRF).
    ///
    /// Performs both dense HNSW search and sparse inverted index search,
    /// then fuses results using RRF scoring.
    ///
    /// Args:
    ///     query_vector (list[float]): Dense query vector
    ///     sparse_indices_or_dict: Sparse query as indices list or dict
    ///     sparse_values (list[float], optional): Sparse weights (required if indices is a list)
    ///     k (int): Number of results to return (default: 10)
    ///     alpha (float): Blending factor. 1.0 = pure dense, 0.0 = pure sparse (default: 0.5)
    ///     filter (dict, optional): MongoDB-style metadata filter
    ///
    /// Returns:
    ///     list[dict]: Results with {id, score, metadata} sorted by RRF score descending
    ///
    /// Examples:
    ///     >>> results = db.hybrid_sparse_search(
    ///     ...     query_vector=[0.1, 0.2, 0.3],
    ///     ...     sparse_indices_or_dict={10: 0.5, 42: 1.2},
    ///     ...     k=10, alpha=0.5
    ///     ... )
    #[pyo3(name = "hybrid_sparse_search", signature = (query_vector, sparse_indices_or_dict, sparse_values=None, k=10, alpha=0.5, filter=None))]
    fn hybrid_sparse_search(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        sparse_indices_or_dict: &Bound<'_, PyAny>,
        sparse_values: Option<Vec<f32>>,
        k: usize,
        alpha: f32,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be greater than 0"));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(format!(
                "alpha must be between 0.0 and 1.0, got {}",
                alpha
            )));
        }

        let dense_query = Vector::new(extract_query_vector(query_vector)?);
        let sparse_query = parse_sparse_vector(sparse_indices_or_dict, sparse_values)?;
        let rust_filter = filter.map(parse_filter).transpose()?;

        // Ensure index is ready
        {
            let inner = self.inner.read();
            if inner.store.needs_index_rebuild() {
                drop(inner);
                let mut inner = self.inner.write();
                inner.store.ensure_index_ready().map_err(convert_error)?;
            }
        }

        let inner_arc = Arc::clone(&self.inner);
        let results = py.detach(|| {
            let inner = inner_arc.read();
            inner
                .store
                .hybrid_sparse_search(&dense_query, &sparse_query, k, alpha, rust_filter.as_ref())
                .map_err(convert_error)
        })?;

        // Convert: distance is -rrf_score, so score = -distance
        let mut py_results = Vec::with_capacity(results.len());
        for r in &results {
            let dict = PyDict::new(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("score", -r.distance)?;
            dict.set_item("metadata", json_to_pyobject(py, &r.metadata)?)?;
            py_results.push(dict.into());
        }

        Ok(py_results)
    }
}
