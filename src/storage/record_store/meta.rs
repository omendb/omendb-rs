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
pub struct RecordFlags(u8);

impl RecordFlags {
    pub const DENSE: u8 = 1 << 0;
    pub const SPARSE: u8 = 1 << 1;
    pub const MULTI: u8 = 1 << 2;
    pub const TEXT: u8 = 1 << 3;

    #[must_use]
    pub const fn has_dense(self) -> bool {
        self.0 & Self::DENSE != 0
    }

    #[must_use]
    pub const fn has_sparse(self) -> bool {
        self.0 & Self::SPARSE != 0
    }

    #[must_use]
    pub const fn has_multi(self) -> bool {
        self.0 & Self::MULTI != 0
    }

    #[must_use]
    pub const fn has_text(self) -> bool {
        self.0 & Self::TEXT != 0
    }

    pub fn set_dense(&mut self, enabled: bool) {
        self.set(Self::DENSE, enabled);
    }

    pub fn set_sparse(&mut self, enabled: bool) {
        self.set(Self::SPARSE, enabled);
    }

    pub fn set_multi(&mut self, enabled: bool) {
        self.set(Self::MULTI, enabled);
    }

    pub fn set_text(&mut self, enabled: bool) {
        self.set(Self::TEXT, enabled);
    }

    fn set(&mut self, mask: u8, enabled: bool) {
        if enabled {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }
}
