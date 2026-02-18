use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value as JsonValue;

#[napi(object)]
#[derive(Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f64,
    /// Normalized similarity score (0-1, higher = more similar)
    pub score: f64,
    /// Metadata as JSON (using serde-json feature)
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

#[napi(object)]
pub struct SetItem {
    pub id: String,
    /// Single vector data (for regular stores)
    pub vector: Option<Float32Array>,
    /// Multi-vector data (for multi-vector stores)
    #[napi(ts_type = "Float32Array[] | undefined")]
    pub vectors: Option<Vec<Float32Array>>,
    /// Optional metadata
    #[napi(ts_type = "Record<string, unknown> | undefined")]
    pub metadata: Option<JsonValue>,
    /// Optional text for hybrid search (auto-enables text search, stored in metadata.text)
    pub text: Option<String>,
    /// Optional document for auto-embedding via embeddingFn
    pub document: Option<String>,
}

#[napi(object)]
pub struct GetResult {
    pub id: String,
    pub vector: Float32Array,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

#[napi(object)]
#[derive(Clone)]
pub struct TextSearchResult {
    pub id: String,
    pub score: f64,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
}

#[napi(object)]
#[derive(Clone)]
pub struct HybridSearchResult {
    pub id: String,
    pub score: f64,
    #[napi(ts_type = "Record<string, unknown>")]
    pub metadata: JsonValue,
    /// BM25 keyword matching score (null if document only matched vector search)
    pub keyword_score: Option<f64>,
    /// Vector similarity score (null if document only matched text search)
    pub semantic_score: Option<f64>,
}

#[napi(object)]
pub struct StatsResult {
    pub dimensions: u32,
    pub count: u32,
    pub path: String,
}

#[napi(object)]
pub struct InfoResult {
    pub vector_count: u32,
    pub deleted_count: u32,
    pub dimensions: u32,
    pub metric: String,
    pub frozen_segment_count: u32,
    pub mutable_segment_vectors: u32,
    pub vector_bytes: u32,
    pub graph_bytes: u32,
    pub total_memory_bytes: u32,
    pub wal_entries: u32,
    pub is_persistent: bool,
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub hnsw_ef_search: u32,
    pub quantization: bool,
    pub segment_capacity: u32,
}
