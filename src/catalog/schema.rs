//! Collection schema types for modality-aware storage.

use crate::text::TokenizerPreset;
use crate::Metric;
use serde::{Deserialize, Serialize};

pub type CollectionName = String;
pub type SlotId = u32;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionSchema {
    pub name: CollectionName,
    pub metric: Metric,
    pub dense: Option<DenseSchema>,
    pub sparse: Option<SparseSchema>,
    pub multi: Option<MultiSchema>,
    pub text: Option<TextSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DenseSchema {
    pub dim: u32,
    pub quantization: QuantizationMode,
    pub mutable_index: MutableDenseIndexKind,
    pub frozen_index: FrozenDenseIndexKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SparseSchema {
    pub index_kind: SparseIndexKind,
    pub max_nonzero: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultiSchema {
    pub token_dim: u32,
    pub encoder: MultiEncoderKind,
    pub max_tokens: Option<u32>,
    pub pool_factor: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TextSchema {
    pub tokenizer: TokenizerPreset,
    pub writer_buffer_mb: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationMode {
    None,
    Sq8,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MutableDenseIndexKind {
    Hnsw,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrozenDenseIndexKind {
    Hnsw,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SparseIndexKind {
    InvertedExact,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MultiEncoderKind {
    Muvera,
}
