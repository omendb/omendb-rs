use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::Vector;
use omendb_lib::{Rerank, SearchOptions};
use std::sync::Arc;

use crate::conversions::{convert_error, distance_to_score, extract_multi_vector_query, extract_query_vector};
use crate::database::VectorDatabase;
use crate::filters::parse_filter;
use crate::types::SearchResult;

#[napi]
impl VectorDatabase {
    /// Search for k nearest neighbors.
    ///
    /// @param query - Query vector (number[] or Float32Array)
    /// @param k - Number of results to return
    /// @param options - Optional search options: {filter?, ef?, maxDistance?}
    /// @returns Array of {id, distance, score, metadata}
    ///
    /// @example
    /// ```javascript
    /// // Basic search
    /// db.search([1, 0, 0, 0], 10);
    ///
    /// // With options
    /// db.search([1, 0, 0, 0], 10, { filter: { category: "A" }, ef: 200 });
    /// db.search([1, 0, 0, 0], 10, { maxDistance: 0.5 });
    /// ```
    #[napi]
    pub async fn search(
        &self,
        #[napi(ts_arg_type = "Array<number> | Float32Array | string")]
        query: Either3<Vec<f64>, Float32Array, String>,
        k: u32,
        #[napi(ts_arg_type = "{ filter?: Record<string, unknown>; ef?: number; maxDistance?: number } | undefined")]
        options: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let (filter, ef, max_distance) = if let Some(ref opts) = options {
            let filter = opts.get("filter").cloned();
            let ef = opts
                .get("ef")
                .and_then(|v| v.as_u64().map(|n| n as u32));
            let max_distance = opts.get("maxDistance").and_then(|v| v.as_f64());
            (filter, ef, max_distance)
        } else {
            (None, None, None)
        };

        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(Error::from_reason(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }
        if let Some(max_dist) = max_distance {
            if max_dist < 0.0 {
                return Err(Error::from_reason("maxDistance must be non-negative"));
            }
        }

        let query_vec = match query {
            Either3::C(text) => {
                let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "String query requires an embedding function. Pass embeddingFn to open() or provide a vector query.",
                    )
                })?;
                let result: Vec<Float32Array> = emb_fn.call_async(vec![text]).await?;
                if result.is_empty() {
                    return Err(Error::from_reason("embeddingFn returned empty result"));
                }
                Vector::new(result[0].to_vec())
            }
            Either3::A(arr) => Vector::new(arr.into_iter().map(|x| x as f32).collect()),
            Either3::B(typed) => Vector::new(typed.to_vec()),
        };

        let expected_dims = self.dimensions;
        if expected_dims > 0 && query_vec.dim() != expected_dims as usize {
            return Err(Error::from_reason(format!(
                "Query vector dimension ({}) does not match database dimension ({})",
                query_vec.dim(),
                expected_dims
            )));
        }

        let ef_usize = ef.map(|e| e as usize);
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let max_dist_f32 = max_distance.map(|d| d as f32);

        {
            let inner = self.inner.read();
            if inner.store.needs_index_rebuild() {
                drop(inner);
                let mut inner = self.inner.write();
                inner.store.ensure_index_ready().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();
        let metric = inner.store.metric();
        let results = inner
            .store
            .search_with_options(
                &query_vec,
                k as usize,
                metadata_filter.as_ref(),
                ef_usize,
                max_dist_f32,
            )
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let distance = r.distance as f64;
                SearchResult {
                    id: r.id,
                    distance,
                    score: distance_to_score(distance, metric),
                    metadata: r.metadata,
                }
            })
            .collect())
    }

    /// Search multi-vector store with query tokens.
    ///
    /// Internal method used by unified search() for multi-vector stores.
    ///
    /// @param query - Query tokens (number[][] or Float32Array[])
    /// @param k - Number of results to return
    /// @param rerank - Enable MaxSim reranking for better quality (default: true)
    /// @param rerankFactor - Fetch k*rerankFactor candidates before reranking (default: 32)
    /// @returns Array of {id, distance, metadata}
    #[napi]
    pub fn search_multi(
        &self,
        query: Either<Vec<Vec<f64>>, Vec<Float32Array>>,
        k: u32,
        rerank: Option<bool>,
        rerank_factor: Option<u32>,
    ) -> Result<Vec<SearchResult>> {
        if !self.is_multi_vector {
            return Err(Error::new(
                Status::InvalidArg,
                "searchMulti requires a multi-vector store. Use open() with multiVector: true",
            ));
        }

        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let query_tokens = extract_multi_vector_query(query)?;

        let rerank_opt = match (rerank, rerank_factor) {
            (Some(false), _) => Rerank::Off,
            (_, Some(factor)) => Rerank::Factor(factor as usize),
            _ => Rerank::On,
        };
        let options = SearchOptions::default().rerank(rerank_opt);

        let inner = self.inner.read();
        let metric = inner.store.metric();
        let results = inner
            .store
            .query_with_options(&query_tokens, k as usize, &options)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let distance = r.distance as f64;
                SearchResult {
                    id: r.id,
                    distance,
                    score: distance_to_score(distance, metric),
                    metadata: r.metadata,
                }
            })
            .collect())
    }

    /// Batch search with parallel execution (async).
    ///
    /// Runs searches in parallel using rayon on a blocking thread pool,
    /// keeping the Node.js event loop free.
    #[napi]
    pub async fn search_batch(
        &self,
        queries: Vec<Either<Vec<f64>, Float32Array>>,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }
        if let Some(ef_val) = ef {
            if ef_val < k {
                return Err(Error::from_reason(format!(
                    "ef ({}) must be >= k ({})",
                    ef_val, k
                )));
            }
        }

        let query_vecs: Vec<Vector> = queries
            .into_iter()
            .map(|q| Vector::new(extract_query_vector(q)))
            .collect();

        {
            let mut inner = self.inner.write();
            inner.store.ensure_index_ready().map_err(convert_error)?;
        }

        let inner_arc = Arc::clone(&self.inner);
        let k_usize = k as usize;
        let ef_usize = ef.map(|e| e as usize);

        let metric = {
            let inner = self.inner.read();
            inner.store.metric()
        };

        let output = tokio::task::spawn_blocking(move || {
            let inner = inner_arc.read();
            let all_results =
                inner
                    .store
                    .search_batch_with_metadata(&query_vecs, k_usize, ef_usize);

            let mut output = Vec::with_capacity(all_results.len());
            for result in all_results {
                let results = result?;
                output.push(
                    results
                        .into_iter()
                        .map(|r| {
                            let distance = r.distance as f64;
                            SearchResult {
                                id: r.id,
                                distance,
                                score: distance_to_score(distance, metric),
                                metadata: r.metadata,
                            }
                        })
                        .collect(),
                );
            }
            Ok::<_, anyhow::Error>(output)
        })
        .await
        .map_err(|e| Error::from_reason(format!("Task join error: {e}")))?
        .map_err(convert_error)?;

        Ok(output)
    }
}
