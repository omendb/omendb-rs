//! Collection mutation payloads.

use crate::vector::sparse::SparseVector;
use crate::vector::types::Vector;
use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub struct PutRecord {
    pub id: String,
    pub dense: Option<Vector>,
    pub sparse: Option<SparseVector>,
    pub multi: Option<Vec<Vec<f32>>>,
    pub text: Option<String>,
    pub metadata: Option<JsonValue>,
}
