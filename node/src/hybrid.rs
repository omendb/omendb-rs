use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::Vector;

use crate::conversions::convert_error;
use crate::database::VectorDatabase;
use crate::filters::parse_filter;
use crate::types::{HybridSearchResult, TextSearchResult};

#[napi]
impl VectorDatabase {
    /// Check if text search is enabled.
    ///
    /// Text search is automatically enabled when using set() with text field.
    #[napi(getter)]
    pub fn has_text_search(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_text_search()
    }

    /// Search using text only (BM25 scoring).
    ///
    /// @param query - Text query
    /// @param k - Number of results
    /// @returns Array of {id, score, metadata}
    #[napi(js_name = "searchText")]
    pub fn search_text(&self, query: String, k: u32) -> Result<Vec<TextSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        {
            let mut inner = self.inner.write();
            if inner.store.has_text_search() {
                inner.store.flush().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();

        let results = inner
            .store
            .search_text(&query, k as usize)
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|(id, score)| {
                let metadata = inner
                    .store
                    .get_metadata_by_id(&id)
                    .unwrap_or(serde_json::json!({}));
                TextSearchResult {
                    id,
                    score: score as f64,
                    metadata,
                }
            })
            .collect())
    }

    /// Hybrid search combining vector similarity and text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine HNSW and BM25 results.
    ///
    /// @param queryVector - Query embedding
    /// @param queryText - Text query for BM25
    /// @param k - Number of results
    /// @param options - Optional: {filter?, alpha?, rrfK?, subscores?}
    /// @returns Array of {id, score, metadata, keywordScore?, semanticScore?}
    ///
    /// @example
    /// ```javascript
    /// // Basic hybrid search
    /// db.searchHybrid([1, 0, 0, 0], "machine learning", 10);
    ///
    /// // With options
    /// db.searchHybrid([1, 0, 0, 0], "query", 10, {
    ///   filter: { type: "ml" },
    ///   alpha: 0.7,
    ///   rrfK: 60,
    ///   subscores: true
    /// });
    /// ```
    #[napi(js_name = "searchHybrid")]
    pub async fn search_hybrid(
        &self,
        #[napi(ts_arg_type = "Array<number> | Float32Array | string")]
        query_vector: Either3<Vec<f64>, Float32Array, String>,
        query_text: Option<String>,
        k: u32,
        #[napi(ts_arg_type = "{ filter?: Record<string, unknown>; alpha?: number; rrfK?: number; subscores?: boolean } | undefined")]
        options: Option<serde_json::Value>,
    ) -> Result<Vec<HybridSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let (filter, alpha, rrf_k, subscores) = if let Some(ref opts) = options {
            let filter = opts.get("filter").cloned();
            let alpha = opts.get("alpha").and_then(|v| v.as_f64());
            let rrf_k = opts.get("rrfK").and_then(|v| v.as_u64().map(|n| n as u32));
            let subscores = opts.get("subscores").and_then(|v| v.as_bool());
            (filter, alpha, rrf_k, subscores)
        } else {
            (None, None, None, None)
        };

        if let Some(a) = alpha {
            if !(0.0..=1.0).contains(&a) {
                return Err(Error::from_reason(format!(
                    "alpha must be between 0.0 and 1.0, got {}",
                    a
                )));
            }
        }
        if let Some(rrf) = rrf_k {
            if rrf == 0 {
                return Err(Error::from_reason("rrfK must be greater than 0"));
            }
        }

        let (query_vec, actual_query_text) = match query_vector {
            Either3::C(text) => {
                let emb_fn = self.embedding_fn.as_ref().ok_or_else(|| {
                    Error::from_reason(
                        "String query requires an embedding function. Pass embeddingFn to open() or provide (vector, text) arguments.",
                    )
                })?;
                let result: Vec<Float32Array> = emb_fn.call_async(vec![text.clone()]).await?;
                if result.is_empty() {
                    return Err(Error::from_reason("embeddingFn returned empty result"));
                }
                let vec = Vector::new(result[0].to_vec());
                let text_q = query_text.unwrap_or(text);
                (vec, text_q)
            }
            Either3::A(arr) => {
                let text_q = query_text.ok_or_else(|| {
                    Error::from_reason("query_text is required when query_vector is provided")
                })?;
                (
                    Vector::new(arr.into_iter().map(|x| x as f32).collect()),
                    text_q,
                )
            }
            Either3::B(typed) => {
                let text_q = query_text.ok_or_else(|| {
                    Error::from_reason("query_text is required when query_vector is provided")
                })?;
                (Vector::new(typed.to_vec()), text_q)
            }
        };
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;
        let alpha_f32 = alpha.map(|a| a as f32);
        let rrf_k_usize = rrf_k.map(|k| k as usize);

        {
            let mut inner = self.inner.write();
            if inner.store.has_text_search() {
                inner.store.flush().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();

        if subscores.unwrap_or(false) {
            let results = inner
                .store
                .search_hybrid_with_subscores(
                    &query_vec,
                    &actual_query_text,
                    k as usize,
                    metadata_filter.as_ref(),
                    alpha_f32,
                    rrf_k_usize,
                )
                .map_err(convert_error)?;

            return Ok(results
                .into_iter()
                .map(|(hybrid_result, metadata)| HybridSearchResult {
                    id: hybrid_result.id,
                    score: hybrid_result.score as f64,
                    metadata,
                    keyword_score: hybrid_result.keyword_score.map(|s| s as f64),
                    semantic_score: hybrid_result.semantic_score.map(|s| s as f64),
                })
                .collect());
        }

        let results = inner
            .store
            .search_hybrid(
                &query_vec,
                &actual_query_text,
                k as usize,
                metadata_filter.as_ref(),
                alpha_f32,
                rrf_k_usize,
            )
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|(id, score, metadata)| HybridSearchResult {
                id,
                score: score as f64,
                metadata,
                keyword_score: None,
                semantic_score: None,
            })
            .collect())
    }

    /// Flush pending changes to disk.
    ///
    /// For hybrid search, this commits text index changes.
    #[napi]
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)
    }
}
