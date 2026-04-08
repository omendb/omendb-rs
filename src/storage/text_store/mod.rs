//! Text payload storage contracts.

use crate::catalog::SlotId;
use anyhow::Result;
use parking_lot::RwLock;

#[derive(Debug, Default)]
pub struct TextStore {
    values: RwLock<Vec<Option<String>>>,
}

impl TextStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, slot: SlotId, text: String) -> Result<()> {
        let mut values = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if values.len() <= index {
            values.resize(index + 1, None);
        }
        values[index] = Some(text);
        Ok(())
    }

    pub fn clear(&self, slot: SlotId) -> Result<()> {
        let mut values = self.values.write();
        let index = usize::try_from(slot).expect("slot fits usize");
        if let Some(entry) = values.get_mut(index) {
            *entry = None;
        }
        Ok(())
    }
}
