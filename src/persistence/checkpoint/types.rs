//! Checkpoint payload placeholders.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotMetaCheckpoint {
    pub slot: u32,
    pub id: String,
    pub metadata: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseCheckpoint {
    pub slot: u32,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseCheckpoint {
    pub slot: u32,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCheckpoint {
    pub slot: u32,
    pub tokens: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCheckpoint {
    pub slot: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionCheckpoint {
    pub meta: Vec<SlotMetaCheckpoint>,
    pub dense: Vec<DenseCheckpoint>,
    pub sparse: Vec<SparseCheckpoint>,
    pub multi: Vec<MultiCheckpoint>,
    pub text: Vec<TextCheckpoint>,
}
