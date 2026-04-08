use crate::conversions::{
    call_embedding_fn, convert_error, extract_query_vector, json_to_pyobject,
};
use crate::database::VectorDatabase;
use crate::filters::parse_filter;
use crate::open::parse_text_search_config;
use omendb_lib::vector::Vector;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;

#[pymethods]
impl VectorDatabase {
    /// Enable text search for hybrid (vector + text) search.
    ///
    /// Note: This is called automatically when using set() with items that have
    /// a `text` field. Only call manually if you need custom text-search config.
    ///
    /// Args:
    ///     config (dict, optional): Text search config with:
    ///         - buffer_mb / writer_buffer_mb: Writer buffer size in MB (default: 50)
    ///         - tokenizer: Tokenizer preset: "default", "code", or "raw"
    ///
    /// Examples:
    ///     >>> db.enable_text_search({"buffer_mb": 100})
    ///     >>> db.enable_text_search({"tokenizer": "code"})
    #[pyo3(name = "enable_text_search", signature = (config=None))]
    fn enable_text_search(&self, config: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let mut inner = self.inner.write();
        let config = parse_text_search_config(config)?;

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

        let results = inner.store.search_text(query, k).map_err(convert_error)?;

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
    #[pyo3(name = "search_hybrid", signature = (query_vector, query_text=None, k=10, filter=None, alpha=None, rrf_k=None, subscores=None))]
    fn search_hybrid(
        &self,
        py: Python<'_>,
        query_vector: &Bound<'_, PyAny>,
        query_text: Option<&str>,
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

        // Resolve query vector and text
        // If first arg is a string and embedding_fn is set, auto-embed it
        let (query_vec, actual_query_text) = if let Ok(text) = query_vector.extract::<String>() {
            let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                    PyValueError::new_err(
                        "String query requires an embedding function. Pass embedding_fn to open() or provide (vector, text) arguments.",
                    )
                })?;
            let embedded = call_embedding_fn(py, emb_fn, &[text.clone()])?;
            let vec = Vector::new(embedded.into_iter().next().unwrap());
            // Use the string as both vector query (embedded) and text query
            let text_query = query_text.map(String::from).unwrap_or(text);
            (vec, text_query)
        } else {
            let vec = Vector::new(extract_query_vector(query_vector)?);
            let text_query = query_text
                .ok_or_else(|| {
                    PyValueError::new_err("query_text is required when query_vector is provided")
                })?
                .to_string();
            (vec, text_query)
        };

        let rust_filter = filter.map(parse_filter).transpose()?;

        let mut inner = self.inner.write();

        // Auto-flush text index to ensure search sees latest inserts
        if inner.store.has_text_search() {
            inner.store.flush().map_err(convert_error)?;
        }

        // Use subscores path when requested
        if subscores.unwrap_or(false) {
            let results = inner
                .store
                .search_hybrid_with_subscores(
                    &query_vec,
                    &actual_query_text,
                    k,
                    rust_filter.as_ref(),
                    alpha,
                    rrf_k,
                )
                .map_err(convert_error)?;

            let mut py_results = Vec::with_capacity(results.len());
            for (hybrid_result, metadata) in results {
                let dict = PyDict::new(py);
                dict.set_item("id", &hybrid_result.id)?;
                dict.set_item("score", hybrid_result.score)?;
                dict.set_item("metadata", json_to_pyobject(py, &metadata)?)?;

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
        let results = inner
            .store
            .search_hybrid(
                &query_vec,
                &actual_query_text,
                k,
                rust_filter.as_ref(),
                alpha,
                rrf_k,
            )
            .map_err(convert_error)?;

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
}
