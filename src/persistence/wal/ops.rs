//! Transport-neutral WAL operations for modality-aware storage.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutMetaOp {
    pub id: String,
    pub metadata: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutDenseOp {
    pub id: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutSparseOp {
    pub id: String,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutMultiOp {
    pub id: String,
    pub tokens: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutTextOp {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRecordOp {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOp {
    PutMeta(PutMetaOp),
    PutDense(PutDenseOp),
    PutSparse(PutSparseOp),
    PutMulti(PutMultiOp),
    PutText(PutTextOp),
    Delete(DeleteRecordOp),
}
