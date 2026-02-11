use crate::conversions::{
    call_embedding_fn, convert_error, extract_batch_queries, extract_multi_vector_query,
    extract_query_vector, results_to_py,
};
use crate::database::VectorDatabase;
use crate::filters::parse_filter;
use omendb_lib::vector::{SearchResult, Vector};
use omendb_lib::{Rerank, SearchOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

#[pymethods]
impl VectorDatabase {
    /// Search for k nearest neighbors (single query).
    ///
    /// Releases the GIL during search for better concurrency with Python threads.
    ///
    /// Args:
    ///     query: Query vector (list of floats or 1D numpy array).
    ///         For multi-vector stores: list of lists or 2D numpy array.
    ///     k (int): Number of nearest neighbors to return
    ///     ef (int, optional): Search width override (default: auto-tuned)
    ///     filter (dict, optional): MongoDB-style metadata filter
    ///     max_distance (float, optional): Filter out results beyond this distance
    ///     rerank (bool, optional): Enable MaxSim reranking for multi-vector stores (default: True)
    ///     rerank_factor (int, optional): Candidates multiplier for reranking (default: 32)
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
    ///
    ///     Multi-vector search (ColBERT-style):
    ///     >>> results = db.search([[0.1]*128, [0.2]*128], k=10)
    ///     >>> results = db.search(query_tokens, k=10, rerank=False)  # Skip reranking
    #[pyo3(name = "search", signature = (query, k, ef=None, filter=None, max_distance=None, rerank=None, rerank_factor=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
        max_distance: Option<f32>,
        rerank: Option<bool>,
        rerank_factor: Option<usize>,
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

        let rust_filter = filter.map(parse_filter).transpose()?;

        // If query is a string, auto-embed it
        if let Ok(query_str) = query.extract::<String>() {
            let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                PyValueError::new_err(
                    "String query requires an embedding function. Pass embedding_fn to open() or provide a vector query.",
                )
            })?;
            let embedded = call_embedding_fn(py, emb_fn, &[query_str])?;
            let query_vec = Vector::new(embedded.into_iter().next().unwrap());

            // Ensure index is ready before releasing GIL
            {
                let inner = self.inner.read();
                if inner.store.needs_index_rebuild() {
                    drop(inner);
                    let mut inner = self.inner.write();
                    inner.store.ensure_index_ready().map_err(convert_error)?;
                }
            }

            let inner_arc = Arc::clone(&self.inner);
            let metric = self.inner.read().store.metric();
            let results = py.detach(|| {
                let inner = inner_arc.read();
                inner.store.search_with_options(
                    &query_vec,
                    k,
                    rust_filter.as_ref(),
                    ef,
                    max_distance,
                )
            });
            let results = results.map_err(convert_error)?;
            return results_to_py(py, &results, metric);
        }

        // Multi-vector store: use query() with SearchOptions
        if self.is_multi_vector {
            let query_tokens = extract_multi_vector_query(query)?;

            // Build rerank mode
            let rerank_mode = match (rerank, rerank_factor) {
                (Some(false), _) => Rerank::Off,
                (_, Some(factor)) => Rerank::Factor(factor),
                _ => Rerank::On, // Default: rerank enabled
            };

            // Build search options
            let mut options = SearchOptions::default().rerank(rerank_mode);
            if let Some(ef_val) = ef {
                options = options.ef(ef_val);
            }
            if let Some(f) = rust_filter {
                options = options.filter(f);
            }
            if let Some(max_dist) = max_distance {
                options = options.max_distance(max_dist);
            }

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
            let metric = self.inner.read().store.metric();
            let results = py.detach(move || {
                let inner = inner_arc.read();
                inner
                    .store
                    .query_with_options(&query_tokens, k, &options)
                    .map_err(convert_error)
            })?;

            return results_to_py(py, &results, metric);
        }

        // Single-vector store: original logic
        let query_vec = Vector::new(extract_query_vector(query)?);

        // Validate query dimensions match the database
        let expected_dims = self.dimensions;
        if expected_dims > 0 && query_vec.dim() != expected_dims {
            return Err(PyValueError::new_err(format!(
                "Query vector dimension ({}) does not match database dimension ({})",
                query_vec.dim(),
                expected_dims
            )));
        }

        // Ensure index is ready before releasing GIL
        {
            let inner = self.inner.read();
            if inner.store.needs_index_rebuild() {
                drop(inner);
                let mut inner = self.inner.write();
                inner.store.ensure_index_ready().map_err(convert_error)?;
            }
        }

        // Clone Arc for use inside detach
        let inner_arc = Arc::clone(&self.inner);
        let metric = self.inner.read().store.metric();

        // Release GIL during compute-intensive search
        let results = py.detach(|| {
            let inner = inner_arc.read();
            inner
                .store
                .search_with_options(&query_vec, k, rust_filter.as_ref(), ef, max_distance)
        });

        let results = results.map_err(convert_error)?;

        // Convert to Python (needs GIL)
        results_to_py(py, &results, metric)
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
        }

        let inner_arc = Arc::clone(&self.inner);

        // Time just the Rust search (no result conversion)
        let t0 = Instant::now();
        for _ in 0..n_iterations {
            py.detach(|| {
                let inner = inner_arc.read();
                let _ = inner
                    .store
                    .search_with_options(&query_vec, k, None, None, None);
            });
        }
        let t_search_total = t0.elapsed().as_micros() as f64;

        // Time segments search directly (bypass VectorStore wrapper)
        let t0 = Instant::now();
        for _ in 0..n_iterations {
            py.detach(|| {
                let inner = inner_arc.read();
                if let Some(ref segments) = inner.store.segments() {
                    let _ = segments.search(&query_vec.data, k, 100);
                }
            });
        }
        let t_hnsw_total = t0.elapsed().as_micros() as f64;

        // Time segments search in tight Rust loop (no GIL release per iteration)
        let t0 = Instant::now();
        py.detach(|| {
            let inner = inner_arc.read();
            if let Some(ref segments) = inner.store.segments() {
                for _ in 0..n_iterations {
                    let _ = segments.search(&query_vec.data, k, 100);
                }
            }
        });
        let t_hnsw_tight_total = t0.elapsed().as_micros() as f64;

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
    ///     filter (dict, optional): MongoDB-style metadata filter (parsed once, shared across batch)
    ///     max_distance (float, optional): Filter out results beyond this distance
    ///
    /// Returns:
    ///     list[list[dict]]: Results for each query
    #[pyo3(name = "search_batch", signature = (queries, k, ef=None, filter=None, max_distance=None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyDict>>,
        max_distance: Option<f32>,
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
        if let Some(max_dist) = max_distance {
            if max_dist < 0.0 {
                return Err(PyValueError::new_err("max_distance must be non-negative"));
            }
        }

        let query_vecs: Vec<Vector> = extract_batch_queries(queries)?
            .into_iter()
            .map(Vector::new)
            .collect();

        // Parse filter once before batch (avoids N re-parses)
        let rust_filter = filter.map(parse_filter).transpose()?;

        // Ensure index and cache are ready
        {
            let mut inner = self.inner.write();
            inner.store.ensure_index_ready().map_err(convert_error)?;
        }

        // Release GIL and search in parallel
        let metric = self.inner.read().store.metric();
        let all_results: Vec<Result<Vec<SearchResult>, _>> = py.detach(|| {
            let inner = self.inner.read();
            inner.store.search_batch_with_filter(
                &query_vecs,
                k,
                rust_filter.as_ref(),
                ef,
                max_distance,
            )
        });

        // Convert to Python
        let mut py_all_results = Vec::with_capacity(all_results.len());
        for result in all_results {
            let results = result.map_err(convert_error)?;
            py_all_results.push(results_to_py(py, &results, metric)?);
        }

        Ok(py_all_results)
    }
}
