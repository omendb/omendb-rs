use std::fmt::Debug;

/// Decision made by the compaction filter for a key-value pair
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterDecision {
    /// Keep the key-value pair as is
    Keep,
    /// Remove the key-value pair
    Remove,
    /// Change the value of the key-value pair
    ChangeValue(Vec<u8>),
}

/// Trait for custom logic during compaction
///
/// Allows applications to hook into the compaction process to:
/// - Discard expired data (TTL)
/// - Merge multiple versions of a key (CRDTs, counters)
/// - Transform values
pub trait CompactionFilter: Send + Sync + Debug {
    /// Inspect a key-value pair during compaction.
    /// Returns the action to take.
    fn filter(&self, level: usize, key: &[u8], value: &[u8]) -> FilterDecision;

    /// Called when merging multiple versions of the same key.
    ///
    /// Allows implementing "Merge Operator" logic within compaction.
    /// If implemented, this is called with all versions of a key found during compaction,
    /// ordered from newest to oldest.
    ///
    /// # Arguments
    /// * `level` - The output level of the compaction
    /// * `key` - The key being merged
    /// * `values` - List of values for the key, ordered from NEWEST to OLDEST
    ///
    /// # Returns
    /// * `Some(value)` - The merged value to keep (will be passed to `filter` next)
    /// * `None` - If the default behavior (keep newest) should be used
    fn merge(&self, _level: usize, _key: &[u8], _values: &[&[u8]]) -> Option<Vec<u8>> {
        None
    }
}
