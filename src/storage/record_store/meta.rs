//! Record metadata for slot-addressed storage.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordMeta {
    pub id: String,
    pub metadata: Option<JsonValue>,
    pub flags: RecordFlags,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecordFlags {
    pub has_dense: bool,
    pub has_sparse: bool,
    pub has_multi: bool,
    pub has_text: bool,
}
