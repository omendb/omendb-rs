//! Search request types for collection-oriented APIs.

use crate::vector::sparse::SparseVector;
use crate::vector::types::Vector;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub struct DenseSearchRequest {
    pub vector: Vector,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct SparseSearchRequest {
    pub vector: SparseVector,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct MultiSearchRequest {
    pub tokens: Vec<Vec<f32>>,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct TextSearchRequest {
    pub query: String,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct HybridSearchRequest {
    pub dense: Option<Vector>,
    pub sparse: Option<SparseVector>,
    pub text: Option<String>,
    pub limit: usize,
    pub alpha: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum SearchRequest {
    Dense(DenseSearchRequest),
    Sparse(SparseSearchRequest),
    Multi(MultiSearchRequest),
    Text(TextSearchRequest),
    Hybrid(HybridSearchRequest),
    HybridWithMetadata {
        request: HybridSearchRequest,
        filter: Option<JsonValue>,
    },
}
