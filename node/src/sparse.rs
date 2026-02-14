use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::sparse::SparseVector;
use omendb_lib::vector::Vector;
use serde_json::Value as JsonValue;

use crate::conversions::convert_error;
use crate::database::VectorDatabase;
use crate::filters::parse_filter;

/// Sparse search result returned from sparseSearch / hybridSparseSearch.
#[napi(object)]
#[derive(Clone)]
pub struct SparseSearchResult {
    pub id: String,
    /// Dot product score (higher = more similar)
    pub score: f64,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

/// Parse a sparse vector from JS input.
///
/// Accepts either:
/// - A JSON object `{dim: weight, ...}` where keys are dimension indices
/// - An object `{indices: number[], values: number[]}` with parallel arrays
fn parse_sparse_from_json(input: &JsonValue) -> Result<SparseVector> {
    // Try {indices: [...], values: [...]} format
    if let (Some(indices), Some(values)) = (input.get("indices"), input.get("values")) {
        let indices: Vec<u32> = indices
            .as_array()
            .ok_or_else(|| Error::from_reason("indices must be an array"))?
            .iter()
            .map(|v| {
                let n = v
                    .as_u64()
                    .ok_or_else(|| Error::from_reason("indices must be unsigned integers"))?;
                u32::try_from(n)
                    .map_err(|_| Error::from_reason(format!("index {n} exceeds u32::MAX")))
            })
            .collect::<Result<Vec<_>>>()?;
        let values: Vec<f32> = values
            .as_array()
            .ok_or_else(|| Error::from_reason("values must be an array"))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|n| n as f32)
                    .ok_or_else(|| Error::from_reason("values must be numbers"))
            })
            .collect::<Result<Vec<_>>>()?;
        if indices.len() != values.len() {
            return Err(Error::from_reason(format!(
                "indices and values must have the same length: {} vs {}",
                indices.len(),
                values.len()
            )));
        }
        let pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
        return SparseVector::from_pairs(pairs).map_err(|e| Error::from_reason(e.to_string()));
    }

    // Try {dim: weight, ...} dict format
    if let Some(obj) = input.as_object() {
        let mut pairs = Vec::with_capacity(obj.len());
        for (k, v) in obj {
            let dim: u32 = k
                .parse()
                .map_err(|_| Error::from_reason(format!("invalid dimension key: {}", k)))?;
            let weight = v.as_f64().ok_or_else(|| {
                Error::from_reason(format!("weight must be a number for dim {}", k))
            })? as f32;
            pairs.push((dim, weight));
        }
        return SparseVector::from_pairs(pairs).map_err(|e| Error::from_reason(e.to_string()));
    }

    Err(Error::from_reason(
        "sparse vector must be {indices: number[], values: number[]} or {dim: weight, ...}",
    ))
}

#[napi]
impl VectorDatabase {
    /// Enable sparse vector indexing for SPLADE-style retrieval.
    ///
    /// Called automatically by setSparse() and setHybridSparse().
    /// Call explicitly before sparseSearch() on an empty index.
    #[napi(js_name = "enableSparse")]
    pub fn enable_sparse(&self) {
        let mut inner = self.inner.write();
        inner.store.enable_sparse();
    }

    /// Check if sparse indexing is enabled.
    #[napi(getter, js_name = "hasSparse")]
    pub fn has_sparse(&self) -> bool {
        let inner = self.inner.read();
        inner.store.has_sparse()
    }

    /// Insert or update a sparse vector.
    ///
    /// @param id - Unique identifier
    /// @param sparse - Sparse vector as {indices: number[], values: number[]} or {dim: weight}
    /// @param metadata - Optional metadata
    ///
    /// @example
    /// ```javascript
    /// db.setSparse("doc1", {indices: [10, 42], values: [0.5, 1.2]}, {title: "Hello"});
    /// db.setSparse("doc2", {"10": 0.5, "42": 1.2}, {title: "World"});
    /// ```
    #[napi(js_name = "setSparse")]
    pub fn set_sparse(
        &self,
        id: String,
        #[napi(ts_arg_type = "{ indices: number[]; values: number[] } | Record<string, number>")]
        sparse: JsonValue,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] metadata: Option<JsonValue>,
    ) -> Result<()> {
        let sparse_vec = parse_sparse_from_json(&sparse)?;
        let meta = metadata.unwrap_or(serde_json::json!({}));

        let mut inner = self.inner.write();
        inner
            .store
            .set_sparse(&id, sparse_vec, meta)
            .map_err(convert_error)
    }

    /// Insert or update both dense and sparse vectors together.
    ///
    /// @param id - Unique identifier
    /// @param vector - Dense vector
    /// @param sparse - Sparse vector
    /// @param metadata - Optional metadata
    #[napi(js_name = "setHybridSparse")]
    pub fn set_hybrid_sparse(
        &self,
        id: String,
        #[napi(ts_arg_type = "Array<number> | Float32Array")] vector: Either<
            Vec<f64>,
            Float32Array,
        >,
        #[napi(ts_arg_type = "{ indices: number[]; values: number[] } | Record<string, number>")]
        sparse: JsonValue,
        #[napi(ts_arg_type = "Record<string, unknown> | undefined")] metadata: Option<JsonValue>,
    ) -> Result<()> {
        let dense = match vector {
            Either::A(arr) => Vector::new(arr.into_iter().map(|x| x as f32).collect()),
            Either::B(typed) => Vector::new(typed.to_vec()),
        };
        let sparse_vec = parse_sparse_from_json(&sparse)?;
        let meta = metadata.unwrap_or(serde_json::json!({}));

        let mut inner = self.inner.write();
        inner
            .store
            .set_hybrid_sparse(&id, dense, sparse_vec, meta)
            .map_err(convert_error)
    }

    /// Search sparse vectors by dot product similarity.
    ///
    /// @param query - Sparse query vector
    /// @param k - Number of results
    /// @param options - Optional: {filter?}
    /// @returns Array of {id, score, metadata} sorted by score descending
    ///
    /// @example
    /// ```javascript
    /// const results = db.sparseSearch({indices: [10, 42], values: [1.0, 0.5]}, 5);
    /// const results = db.sparseSearch({"10": 1.0, "42": 0.5}, 5);
    /// ```
    #[napi(js_name = "sparseSearch")]
    pub fn sparse_search(
        &self,
        #[napi(ts_arg_type = "{ indices: number[]; values: number[] } | Record<string, number>")]
        query: JsonValue,
        k: u32,
        #[napi(ts_arg_type = "{ filter?: Record<string, unknown> } | undefined")] options: Option<
            JsonValue,
        >,
    ) -> Result<Vec<SparseSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let sparse_query = parse_sparse_from_json(&query)?;
        let filter = options.as_ref().and_then(|o| o.get("filter").cloned());
        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;

        let inner = self.inner.read();
        let results = inner
            .store
            .sparse_search(&sparse_query, k as usize, metadata_filter.as_ref())
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| SparseSearchResult {
                id: r.id,
                score: (-r.distance) as f64,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Hybrid dense + sparse search with Reciprocal Rank Fusion (RRF).
    ///
    /// @param queryVector - Dense query vector
    /// @param sparseQuery - Sparse query vector
    /// @param k - Number of results
    /// @param options - Optional: {alpha?, filter?}
    /// @returns Array of {id, score, metadata}
    ///
    /// @example
    /// ```javascript
    /// const results = db.hybridSparseSearch(
    ///   [1, 0, 0],
    ///   {indices: [10, 42], values: [1.0, 0.5]},
    ///   10,
    ///   { alpha: 0.5 }
    /// );
    /// ```
    #[napi(js_name = "hybridSparseSearch")]
    pub fn hybrid_sparse_search(
        &self,
        #[napi(ts_arg_type = "Array<number> | Float32Array")] query_vector: Either<
            Vec<f64>,
            Float32Array,
        >,
        #[napi(ts_arg_type = "{ indices: number[]; values: number[] } | Record<string, number>")]
        sparse_query: JsonValue,
        k: u32,
        #[napi(ts_arg_type = "{ alpha?: number; filter?: Record<string, unknown> } | undefined")]
        options: Option<JsonValue>,
    ) -> Result<Vec<SparseSearchResult>> {
        if k == 0 {
            return Err(Error::from_reason("k must be greater than 0"));
        }

        let dense_query = match query_vector {
            Either::A(arr) => Vector::new(arr.into_iter().map(|x| x as f32).collect()),
            Either::B(typed) => Vector::new(typed.to_vec()),
        };
        let sparse_q = parse_sparse_from_json(&sparse_query)?;

        let (alpha, filter) = if let Some(ref opts) = options {
            let alpha = opts.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let filter = opts.get("filter").cloned();
            (alpha, filter)
        } else {
            (0.5, None)
        };

        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::from_reason(format!(
                "alpha must be between 0.0 and 1.0, got {}",
                alpha
            )));
        }

        let metadata_filter = filter.as_ref().map(parse_filter).transpose()?;

        // Ensure index is ready
        {
            let inner = self.inner.read();
            if inner.store.needs_index_rebuild() {
                drop(inner);
                let mut inner = self.inner.write();
                inner.store.ensure_index_ready().map_err(convert_error)?;
            }
        }

        let inner = self.inner.read();
        let results = inner
            .store
            .hybrid_sparse_search(
                &dense_query,
                &sparse_q,
                k as usize,
                alpha as f32,
                metadata_filter.as_ref(),
            )
            .map_err(convert_error)?;

        Ok(results
            .into_iter()
            .map(|r| SparseSearchResult {
                id: r.id,
                score: (-r.distance) as f64,
                metadata: r.metadata,
            })
            .collect())
    }
}
