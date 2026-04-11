use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::catalog::{
    CollectionSchema, DenseSchema, FrozenDenseIndexKind, GraphSchema, GraphTemporalMode,
    MultiEncoderKind, MultiSchema, MutableDenseIndexKind, QuantizationMode, SparseIndexKind,
    SparseSchema, TextSchema,
};
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
    pub schema: CollectionSchemaResult,
}

#[napi(object)]
pub struct CollectionSchemaResult {
    pub name: String,
    pub metric: String,
    pub dense: Option<DenseSchemaResult>,
    pub sparse: Option<SparseSchemaResult>,
    pub multi: Option<MultiSchemaResult>,
    pub text: Option<TextSchemaResult>,
    pub graph: Option<GraphSchemaResult>,
}

#[napi(object)]
pub struct DenseSchemaResult {
    pub dim: u32,
    pub quantization: String,
    pub mutable_index: String,
    pub frozen_index: String,
}

#[napi(object)]
pub struct SparseSchemaResult {
    pub index_kind: String,
    pub max_nonzero: Option<u32>,
}

#[napi(object)]
pub struct MultiSchemaResult {
    pub token_dim: u32,
    pub encoder: String,
    pub repetitions: u8,
    pub partition_bits: u8,
    pub d_proj: Option<u8>,
    pub seed: f64,
    pub max_tokens: Option<u32>,
    pub pool_factor: Option<u8>,
}

#[napi(object)]
pub struct TextSchemaResult {
    pub tokenizer: String,
    pub writer_buffer_mb: u32,
}

#[napi(object)]
pub struct GraphSchemaResult {
    pub enabled: bool,
    pub temporal: String,
    pub provenance: bool,
}

impl From<CollectionSchema> for CollectionSchemaResult {
    fn from(value: CollectionSchema) -> Self {
        Self {
            name: value.name,
            metric: value.metric.canonical_name().to_string(),
            dense: value.dense.map(Into::into),
            sparse: value.sparse.map(Into::into),
            multi: value.multi.map(Into::into),
            text: value.text.map(Into::into),
            graph: value.graph.map(Into::into),
        }
    }
}

impl From<DenseSchema> for DenseSchemaResult {
    fn from(value: DenseSchema) -> Self {
        Self {
            dim: value.dim,
            quantization: match value.quantization {
                QuantizationMode::None => "none".to_string(),
                QuantizationMode::Sq8 => "sq8".to_string(),
            },
            mutable_index: match value.mutable_index {
                MutableDenseIndexKind::Hnsw => "hnsw".to_string(),
            },
            frozen_index: match value.frozen_index {
                FrozenDenseIndexKind::Hnsw => "hnsw".to_string(),
            },
        }
    }
}

impl From<SparseSchema> for SparseSchemaResult {
    fn from(value: SparseSchema) -> Self {
        Self {
            index_kind: match value.index_kind {
                SparseIndexKind::InvertedExact => "inverted_exact".to_string(),
            },
            max_nonzero: value.max_nonzero,
        }
    }
}

impl From<MultiSchema> for MultiSchemaResult {
    fn from(value: MultiSchema) -> Self {
        Self {
            token_dim: value.token_dim,
            encoder: match value.encoder {
                MultiEncoderKind::Muvera => "muvera".to_string(),
            },
            repetitions: value.repetitions,
            partition_bits: value.partition_bits,
            d_proj: value.d_proj,
            seed: value.seed as f64,
            max_tokens: value.max_tokens,
            pool_factor: value.pool_factor,
        }
    }
}

impl From<TextSchema> for TextSchemaResult {
    fn from(value: TextSchema) -> Self {
        Self {
            tokenizer: format!("{:?}", value.tokenizer).to_lowercase(),
            writer_buffer_mb: value.writer_buffer_mb,
        }
    }
}

impl From<GraphSchema> for GraphSchemaResult {
    fn from(value: GraphSchema) -> Self {
        Self {
            enabled: value.enabled,
            temporal: match value.temporal {
                GraphTemporalMode::None => "none".to_string(),
                GraphTemporalMode::ValidAt => "valid_at".to_string(),
                GraphTemporalMode::BiTemporal => "bi_temporal".to_string(),
            },
            provenance: value.provenance,
        }
    }
}
