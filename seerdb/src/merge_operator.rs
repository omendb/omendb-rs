use std::fmt::Debug;

/// MergeOperator allows defining custom Read-Modify-Write logic
/// that is applied at compaction time (and read time).
///
/// This enables efficient "blind writes" where a client can issue a
/// `merge(key, operand)` command without reading the current value.
/// The database stores the operand and merges it lazily.
///
/// # Use Cases
/// - Counters (Increment/Decrement)
/// - List Appends
/// - String Concatenation
/// - JSON Updates
/// - Graph Edge Adjacency List Updates (omendb primary use case)
pub trait MergeOperator: Debug + Send + Sync + 'static {
    /// Full Merge: Merges a list of operands into an existing base value.
    ///
    /// # Arguments
    /// * `key` - The key being merged
    /// * `existing_value` - The existing value (if any)
    /// * `operands` - List of operands to apply (ordered oldest to newest)
    ///
    /// # Returns
    /// * `Some(vec)` - The new merged value
    /// * `None` - The merge failed / value corrupted
    fn full_merge(
        &self,
        key: &[u8],
        existing_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Option<Vec<u8>>;

    /// Partial Merge: Merges two operands into a single operand.
    ///
    /// This is an optimization to allow "folding" operands during compaction
    /// even if the base value is not yet found (e.g. in L0/L1 compaction).
    ///
    /// If not implemented (returns None), the system will stack operands until
    /// the base value is found.
    fn partial_merge(
        &self,
        _key: &[u8],
        _left_operand: &[u8],
        _right_operand: &[u8],
    ) -> Option<Vec<u8>> {
        None
    }

    /// Returns the name of the merge operator.
    /// Used for verification to ensure the same operator is used on recovery.
    fn name(&self) -> &str;
}

/// A simple merge operator that appends operands as strings with a delimiter.
/// Useful for testing and simple logs.
#[derive(Debug)]
pub struct StringAppendOperator {
    delimiter: char,
}

impl StringAppendOperator {
    pub fn new(delimiter: char) -> Self {
        Self { delimiter }
    }
}

impl MergeOperator for StringAppendOperator {
    fn full_merge(
        &self,
        _key: &[u8],
        existing_value: Option<&[u8]>,
        operands: &[&[u8]],
    ) -> Option<Vec<u8>> {
        let mut result = String::new();

        if let Some(val) = existing_value {
            if let Ok(s) = std::str::from_utf8(val) {
                result.push_str(s);
            }
        }

        for op in operands {
            if let Ok(s) = std::str::from_utf8(op) {
                if !result.is_empty() {
                    result.push(self.delimiter);
                }
                result.push_str(s);
            }
        }

        Some(result.into_bytes())
    }

    fn partial_merge(
        &self,
        _key: &[u8],
        left_operand: &[u8],
        right_operand: &[u8],
    ) -> Option<Vec<u8>> {
        let mut result = String::new();
        if let Ok(l) = std::str::from_utf8(left_operand) {
            result.push_str(l);
            result.push(self.delimiter);
            if let Ok(r) = std::str::from_utf8(right_operand) {
                result.push_str(r);
                return Some(result.into_bytes());
            }
        }
        None
    }

    fn name(&self) -> &str {
        "StringAppendOperator"
    }
}
