// Memtable: In-memory buffer for recent writes
// Uses concurrent skiplist for lock-free reads/writes

use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::sstable::SSTableBuilder;
use crate::types::{InternalKey, ValueType};

/// Entry type for high-level operations
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    Value(Bytes),
    Tombstone,
    Merge(Vec<Bytes>),
}

/// In-memory sorted table for recent writes
pub struct Memtable {
    /// Underlying skiplist (concurrent, lock-free)
    /// Key: InternalKey (UserKey + Seq + Kind)
    /// Value: Raw bytes (empty for Tombstone)
    data: Arc<SkipMap<InternalKey, Bytes>>,
    /// Current size in bytes (approximate)
    size: AtomicUsize,
    /// Capacity threshold for flushing
    capacity: usize,
    /// Current sequence number generator (shared with DB, or passed in)
    /// For now, we assume sequence numbers are passed in.
    _unused: (),
}

impl Memtable {
    /// Create a new memtable with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Arc::new(SkipMap::new()),
            size: AtomicUsize::new(0),
            capacity,
            _unused: (),
        }
    }

    /// Create memtable with default capacity (64MB)
    pub fn with_default_capacity() -> Self {
        Self::new(64 * 1024 * 1024)
    }

    /// Insert a key-value pair with a specific sequence number
    #[inline]
    pub fn put(&self, key: Bytes, value: Bytes, seq: u64) {
        let size_delta = key.len() + value.len() + 8; // +8 for seq/type
        let internal_key = InternalKey::new(key, seq, ValueType::Value);

        self.data.insert(internal_key, value);
        self.size.fetch_add(size_delta, Ordering::Relaxed);
    }

    /// Delete a key (insert tombstone) with a specific sequence number
    #[inline]
    pub fn delete(&self, key: Bytes, seq: u64) {
        let size_delta = key.len() + 8;
        let internal_key = InternalKey::new(key, seq, ValueType::Deletion);

        self.data.insert(internal_key, Bytes::new());
        self.size.fetch_add(size_delta, Ordering::Relaxed);
    }

    /// Insert a merge operand
    #[inline]
    pub fn merge(&self, key: Bytes, value: Bytes, seq: u64) {
        let size_delta = key.len() + value.len() + 8;
        let internal_key = InternalKey::new(key, seq, ValueType::Merge);

        self.data.insert(internal_key, value);
        self.size.fetch_add(size_delta, Ordering::Relaxed);
    }

    /// Get the latest value for a key (Snapshot Isolation)
    /// Returns (Value, Sequence) if found and visible <= snapshot_seq
    /// Returns None if not found or deleted
    #[inline]
    pub fn get(&self, key: &[u8], snapshot_seq: u64) -> Option<(Bytes, u64)> {
        let lookup_key =
            InternalKey::new(Bytes::copy_from_slice(key), snapshot_seq, ValueType::Value);

        // range(lookup_key..) will find the first key >= lookup_key
        // Since InternalKeys are sorted by UserKey ASC, Seq DESC:
        // Key + Seq(MAX) comes BEFORE Key + Seq(100)
        // So if we search for Key + snapshot_seq, we will find the first version <= snapshot_seq
        for entry in self.data.range(lookup_key..) {
            let ikey = entry.key();

            // Check if we moved to a different user key
            if ikey.user_key.as_ref() != key {
                return None;
            }

            // Since we used range(lookup_key..), and sort is Seq DESC,
            // this entry MUST have seq <= snapshot_seq.

            match ikey.kind {
                ValueType::Value => return Some((entry.value().clone(), ikey.seq)),
                ValueType::Deletion => return None, // Deleted
                ValueType::Merge => continue, // Skip merge operands for simple get (needs merge logic)
                ValueType::Log => continue,
            }
        }

        None
    }

    /// Get Entry (Value, Tombstone, or Merge list) for a key.
    /// This collects all versions/merges visible?
    /// Assuming this retrieves the LATEST state for merge resolution.
    pub fn get_entry(&self, key: &[u8]) -> Option<Entry> {
        let lookup_key = InternalKey::new(Bytes::copy_from_slice(key), u64::MAX, ValueType::Value);

        let mut merges = Vec::new();

        for entry in self.data.range(lookup_key..) {
            let ikey = entry.key();
            if ikey.user_key.as_ref() != key {
                break;
            }

            match ikey.kind {
                ValueType::Value => {
                    if merges.is_empty() {
                        return Some(Entry::Value(entry.value().clone()));
                    } else {
                        // Value implies base.
                        return Some(Entry::Merge(merges));
                    }
                }
                ValueType::Deletion => {
                    if merges.is_empty() {
                        return Some(Entry::Tombstone);
                    } else {
                        return Some(Entry::Merge(merges));
                    }
                }
                ValueType::Merge => {
                    merges.push(entry.value().clone());
                }
                ValueType::Log => continue,
            }
        }

        if !merges.is_empty() {
            return Some(Entry::Merge(merges));
        }

        None
    }

    /// Put an Entry (used for compaction/merge resolution)
    /// Uses a default sequence number (u64::MAX) since this is usually for
    /// memory-only compaction or immediate updates?
    /// WARNING: This bypasses normal seq assignment.
    pub fn put_entry(&self, key: Bytes, entry: Entry) {
        // Using u64::MAX to ensure it is "latest"
        // But this might mess up snapshot isolation if not careful.
        // db.rs seems to use this for "repair" or atomic replacement.
        let seq = u64::MAX;
        match entry {
            Entry::Value(v) => self.put(key, v, seq),
            Entry::Tombstone => self.delete(key, seq),
            Entry::Merge(ops) => {
                // This is tricky. Writing multiple merges?
                // Or does db.rs imply resolving them?
                // If we are putting a resolved Merge, it usually becomes a Value?
                // But if we put Merge(ops), we write them back.
                // We'll write them in order.
                for op in ops {
                    self.merge(key.clone(), op, seq);
                }
            }
        }
    }

    /// Check if key exists (raw check, ignores visibility)
    #[inline]
    pub fn contains_raw(&self, key: &InternalKey) -> bool {
        self.data.contains_key(key)
    }

    /// Get current size in bytes (approximate)
    #[inline]
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if memtable should be flushed
    #[inline]
    pub fn should_flush(&self) -> bool {
        self.size() >= self.capacity
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all entries in sorted order (internal representation)
    pub fn iter(&self) -> impl Iterator<Item = (InternalKey, Bytes)> + '_ {
        self.data
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
    }

    /// Iterate over all entries as (user_key, Entry) pairs, grouped by user key.
    /// This is the high-level interface for flush operations.
    pub fn iter_entries(&self) -> impl Iterator<Item = (Bytes, Entry)> + '_ {
        self.range_from(&[])
    }

    /// Clone the skipmap for creating an immutable snapshot
    pub fn snapshot(&self) -> Arc<SkipMap<InternalKey, Bytes>> {
        Arc::clone(&self.data)
    }

    /// Flush memtable to disk as an SSTable
    pub fn flush(&self, path: impl AsRef<Path>) -> Result<(), crate::sstable::SSTableError> {
        let mut builder = SSTableBuilder::create(path)?;

        // Iterate in sorted order and add to SSTable
        // Keys are encoded InternalKeys, values are handled based on type
        for (ikey, value) in self.iter() {
            let ikey_bytes = ikey.encode();
            match ikey.kind {
                ValueType::Value => builder.add(ikey_bytes, value)?,
                ValueType::Deletion => builder.add_tombstone(ikey_bytes)?,
                ValueType::Merge => builder.add_merge(ikey_bytes, value)?,
                ValueType::Log => {} // Skip log entries
            }
        }

        builder.finish()
    }

    // --- Range Iteration Support for RangeIterator ---

    /// Forward range scan returning (user_key, Entry) pairs.
    /// Groups by user key and returns the latest Entry for each.
    /// End key is exclusive.
    pub fn range(&self, start: &[u8], end: &[u8]) -> impl Iterator<Item = (Bytes, Entry)> + '_ {
        // InternalKey sorts by UserKey ASC, Seq DESC
        // For start: use u64::MAX so we get the first entry for start_key
        // For end: use u64::MAX so all entries for end_key come AFTER the bound (excluded)
        let start_key = InternalKey::new(Bytes::copy_from_slice(start), u64::MAX, ValueType::Value);
        let end_key = InternalKey::new(Bytes::copy_from_slice(end), u64::MAX, ValueType::Value);

        let mut result = Vec::new();
        let mut current_user_key: Option<Bytes> = None;
        let mut current_entry: Option<Entry> = None;
        let mut merge_operands: Vec<Bytes> = Vec::new();

        for entry in self.data.range(start_key..end_key) {
            let ikey = entry.key();
            let user_key = ikey.user_key.clone();

            // New user key - emit previous if any
            if current_user_key.as_ref() != Some(&user_key) {
                if let Some(uk) = current_user_key.take() {
                    if !merge_operands.is_empty() {
                        result.push((uk, Entry::Merge(std::mem::take(&mut merge_operands))));
                    } else if let Some(e) = current_entry.take() {
                        result.push((uk, e));
                    }
                }
                current_user_key = Some(user_key.clone());
                current_entry = None;
                merge_operands.clear();
            }

            // Only take the first (highest seq) entry for each type
            if current_entry.is_none() && merge_operands.is_empty() {
                match ikey.kind {
                    ValueType::Value => current_entry = Some(Entry::Value(entry.value().clone())),
                    ValueType::Deletion => current_entry = Some(Entry::Tombstone),
                    ValueType::Merge => merge_operands.push(entry.value().clone()),
                    ValueType::Log => {}
                }
            } else if matches!(ikey.kind, ValueType::Merge) {
                merge_operands.push(entry.value().clone());
            }
        }

        // Emit last key
        if let Some(uk) = current_user_key {
            if !merge_operands.is_empty() {
                result.push((uk, Entry::Merge(merge_operands)));
            } else if let Some(e) = current_entry {
                result.push((uk, e));
            }
        }

        result.into_iter()
    }

    /// Forward range scan from start key to end, returning (user_key, Entry) pairs.
    pub fn range_from(&self, start: &[u8]) -> impl Iterator<Item = (Bytes, Entry)> + '_ {
        let start_key = InternalKey::new(Bytes::copy_from_slice(start), u64::MAX, ValueType::Value);

        let mut result = Vec::new();
        let mut current_user_key: Option<Bytes> = None;
        let mut current_entry: Option<Entry> = None;
        let mut merge_operands: Vec<Bytes> = Vec::new();

        for entry in self.data.range(start_key..) {
            let ikey = entry.key();
            let user_key = ikey.user_key.clone();

            if current_user_key.as_ref() != Some(&user_key) {
                if let Some(uk) = current_user_key.take() {
                    if !merge_operands.is_empty() {
                        result.push((uk, Entry::Merge(std::mem::take(&mut merge_operands))));
                    } else if let Some(e) = current_entry.take() {
                        result.push((uk, e));
                    }
                }
                current_user_key = Some(user_key.clone());
                current_entry = None;
                merge_operands.clear();
            }

            if current_entry.is_none() && merge_operands.is_empty() {
                match ikey.kind {
                    ValueType::Value => current_entry = Some(Entry::Value(entry.value().clone())),
                    ValueType::Deletion => current_entry = Some(Entry::Tombstone),
                    ValueType::Merge => merge_operands.push(entry.value().clone()),
                    ValueType::Log => {}
                }
            } else if matches!(ikey.kind, ValueType::Merge) {
                merge_operands.push(entry.value().clone());
            }
        }

        if let Some(uk) = current_user_key {
            if !merge_operands.is_empty() {
                result.push((uk, Entry::Merge(merge_operands)));
            } else if let Some(e) = current_entry {
                result.push((uk, e));
            }
        }

        result.into_iter()
    }

    /// Reverse range scan returning (user_key, Entry) pairs.
    pub fn range_rev(&self, start: &[u8], end: &[u8]) -> impl Iterator<Item = (Bytes, Entry)> + '_ {
        // Collect forward then reverse (simpler than true reverse iteration with grouping)
        let entries: Vec<_> = self.range(start, end).collect();
        entries.into_iter().rev()
    }

    // --- Internal Iteration Support ---

    /// Iterate over all entries in reverse sorted order (internal keys)
    pub fn iter_rev(&self) -> impl Iterator<Item = (InternalKey, Bytes)> + '_ {
        self.data
            .iter()
            .rev()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
    }

    /// Internal range scan in reverse (for internal use)
    pub fn range_rev_internal<'a>(
        &'a self,
        start: Option<&'a InternalKey>,
        end: Option<&'a InternalKey>,
    ) -> impl Iterator<Item = (InternalKey, Bytes)> + 'a {
        use std::ops::Bound;

        let start_bound = match start {
            Some(k) => Bound::Included(k),
            None => Bound::Unbounded,
        };
        let end_bound = match end {
            Some(k) => Bound::Excluded(k),
            None => Bound::Unbounded,
        };

        self.data
            .range((start_bound, end_bound))
            .rev()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memtable_mvcc_put_get() {
        let memtable = Memtable::new(1024);

        // Write v1 at seq 10
        memtable.put(Bytes::from("key1"), Bytes::from("val1"), 10);

        // Write v2 at seq 20
        memtable.put(Bytes::from("key1"), Bytes::from("val2"), 20);

        // Get at seq 15 -> Should see v1
        assert_eq!(memtable.get(b"key1", 15), Some((Bytes::from("val1"), 10)));

        // Get at seq 25 -> Should see v2
        assert_eq!(memtable.get(b"key1", 25), Some((Bytes::from("val2"), 20)));
    }

    #[test]
    fn test_memtable_mvcc_delete() {
        let memtable = Memtable::new(1024);

        memtable.put(Bytes::from("key1"), Bytes::from("val1"), 10);
        assert_eq!(memtable.get(b"key1", 20), Some((Bytes::from("val1"), 10)));

        // Delete at seq 20
        memtable.delete(Bytes::from("key1"), 20);

        // Get at seq 25 -> Should return None (Deleted)
        assert_eq!(memtable.get(b"key1", 25), None);

        // Get at seq 15 -> Should still see v1
        assert_eq!(memtable.get(b"key1", 15), Some((Bytes::from("val1"), 10)));
    }
}
