use super::super::*;
use crate::catalog::{
    FrozenDenseIndexKind, MultiEncoderKind, MutableDenseIndexKind, QuantizationMode,
    SparseIndexKind,
};
use crate::text::{TextSearchConfig, TokenizerPreset};

#[test]
fn test_schema_for_dense_store() {
    let store = VectorStore::new(128);

    let schema = store.schema();
    let dense = schema.dense.expect("dense schema");

    assert_eq!(schema.name, "");
    assert_eq!(schema.metric, Metric::L2);
    assert_eq!(dense.dim, 128);
    assert_eq!(dense.quantization, QuantizationMode::None);
    assert_eq!(dense.mutable_index, MutableDenseIndexKind::Hnsw);
    assert_eq!(dense.frozen_index, FrozenDenseIndexKind::Hnsw);
    assert!(schema.sparse.is_none());
    assert!(schema.multi.is_none());
    assert!(schema.text.is_none());
}

#[test]
fn test_schema_for_quantized_store() {
    let store = VectorStore::new_with_quantization(64);

    let schema = store.schema();
    let dense = schema.dense.expect("dense schema");

    assert_eq!(dense.dim, 64);
    assert_eq!(dense.quantization, QuantizationMode::Sq8);
}

#[test]
fn test_schema_for_sparse_store() {
    let mut store = VectorStore::new(32);
    store.enable_sparse();

    let schema = store.schema();
    let sparse = schema.sparse.expect("sparse schema");

    assert!(schema.dense.is_some());
    assert_eq!(sparse.index_kind, SparseIndexKind::InvertedExact);
    assert_eq!(sparse.max_nonzero, None);
}

#[test]
fn test_schema_for_text_store() {
    let mut store = VectorStore::new(16);
    store
        .enable_text_search_with_config(Some(TextSearchConfig {
            writer_buffer_mb: 20,
            tokenizer: TokenizerPreset::Code,
        }))
        .unwrap();

    let schema = store.schema();
    let text = schema.text.expect("text schema");

    assert!(schema.dense.is_some());
    assert_eq!(text.writer_buffer_mb, 20);
    assert_eq!(text.tokenizer, TokenizerPreset::Code);
}

#[test]
fn test_schema_for_multi_vector_store() {
    let config = MultiVectorConfig {
        repetitions: 10,
        partition_bits: 4,
        seed: 7,
        d_proj: None,
        pool_factor: Some(3),
        max_tokens: Some(256),
    };
    let store = VectorStore::multi_vector_with(96, config).unwrap();

    let schema = store.schema();
    let multi = schema.multi.expect("multi schema");

    assert!(schema.dense.is_none());
    assert_eq!(multi.token_dim, 96);
    assert_eq!(multi.encoder, MultiEncoderKind::Muvera);
    assert_eq!(multi.repetitions, 10);
    assert_eq!(multi.partition_bits, 4);
    assert_eq!(multi.seed, 7);
    assert_eq!(multi.d_proj, None);
    assert_eq!(multi.pool_factor, Some(3));
    assert_eq!(multi.max_tokens, Some(256));
}

#[test]
fn test_info_includes_authoritative_schema() {
    let mut store = VectorStore::new_with_quantization(24);
    store.enable_sparse();
    store
        .enable_text_search_with_config(Some(TextSearchConfig {
            writer_buffer_mb: 20,
            tokenizer: TokenizerPreset::Raw,
        }))
        .unwrap();

    let info = store.info();

    assert_eq!(info.schema.metric, Metric::L2);
    assert_eq!(
        info.schema.dense.expect("dense schema").quantization,
        QuantizationMode::Sq8
    );
    assert!(info.schema.sparse.is_some());
    assert_eq!(
        info.schema.text.expect("text schema").tokenizer,
        TokenizerPreset::Raw
    );
}
