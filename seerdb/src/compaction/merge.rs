// Merge iterator for compaction
// Merges multiple sorted SSTables with MVCC garbage collection

use crate::compaction::{CompactionFilter, FilterDecision};
use crate::sstable::{SSTable, SSTableError};
use crate::types::InternalKey;
use bytes::Bytes;
use std::sync::Arc;

/// Merged entries from all SSTables, sorted by key
pub struct MergeIterator {
    entries: std::vec::IntoIter<(Bytes, Bytes)>,
}

impl MergeIterator {
    /// Create a new merge iterator from multiple SSTables
    ///
    /// Collects all entries, sorts by key, and applies MVCC garbage collection.
    /// For L0 compaction, "newest" means from HIGHER source_id (later in vector).
    /// L0 SSTables are ordered oldest→newest, so higher index = newer data.
    ///
    /// # MVCC Garbage Collection
    /// - `oldest_snapshot`: Sequence number of the oldest active snapshot.
    ///   Pass `u64::MAX` if no snapshots are active (GC everything possible).
    /// - For each user_key, keeps:
    ///   - The newest version (always)
    ///   - Any version with seq >= oldest_snapshot (visible to active snapshots)
    /// - Drops old versions with seq < oldest_snapshot when a newer version exists.
    pub fn new(
        sstables: Vec<SSTable>,
        level: usize,
        filter: Option<Arc<dyn CompactionFilter>>,
    ) -> Result<Self, SSTableError> {
        // Default: no GC (keep all versions)
        Self::with_gc(sstables, level, filter, u64::MAX)
    }

    /// Create a merge iterator with MVCC garbage collection
    ///
    /// # Arguments
    /// * `sstables` - SSTables to merge (older first for L0)
    /// * `level` - Target compaction level
    /// * `filter` - Optional compaction filter
    /// * `oldest_snapshot` - Oldest active snapshot seq (u64::MAX = no snapshots)
    pub fn with_gc(
        mut sstables: Vec<SSTable>,
        level: usize,
        filter: Option<Arc<dyn CompactionFilter>>,
        oldest_snapshot: u64,
    ) -> Result<Self, SSTableError> {
        let mut all_entries = Vec::new();

        // Collect all entries from all SSTables
        for (source_id, sstable) in sstables.iter_mut().enumerate() {
            let iter = sstable.iter()?;

            for result in iter {
                let (key, value) = result?;
                all_entries.push((key, value, source_id));
            }
        }

        // Sort by encoded key first, then by source_id DESCENDING (higher = newer)
        // With MVCC encoding, this sorts by (user_key ASC, seq DESC, source_id DESC)
        all_entries.sort_by(|a, b| {
            match a.0.cmp(&b.0) {
                std::cmp::Ordering::Equal => b.2.cmp(&a.2), // Higher source_id first (NEWEST)
                other => other,
            }
        });

        let mut finalized_entries = Vec::new();

        if let Some(filter) = filter {
            // Group by ENCODED key and apply filter logic
            // (filter API expects encoded keys, not user keys)
            let mut i = 0;
            while i < all_entries.len() {
                let key = &all_entries[i].0;
                let mut j = i + 1;

                // Find all versions of this ENCODED key
                while j < all_entries.len() && all_entries[j].0 == *key {
                    j += 1;
                }

                // Slice of all versions for this key (already sorted by recency)
                let versions = &all_entries[i..j];

                // 1. Merge phase
                let values: Vec<&[u8]> = versions.iter().map(|(_, v, _)| v.as_ref()).collect();

                // Default: pick newest (first in slice)
                let newest_value = &versions[0].1;
                let mut merged_value_bytes = newest_value.clone();

                // Try custom merge
                if let Some(merged) = filter.merge(level, key, &values) {
                    // If merged, we treat it as a new INLINE value
                    // Prepend FLAG_INLINE (1)
                    let mut with_flag = Vec::with_capacity(1 + merged.len());
                    with_flag.push(crate::sstable::FLAG_INLINE);
                    with_flag.extend_from_slice(&merged);
                    merged_value_bytes = Bytes::from(with_flag);
                }

                // 2. Filter phase
                match filter.filter(level, key, &merged_value_bytes) {
                    FilterDecision::Keep => {
                        finalized_entries.push((key.clone(), merged_value_bytes));
                    }
                    FilterDecision::Remove => {
                        // Skip
                    }
                    FilterDecision::ChangeValue(new_val) => {
                        // Treat as new INLINE value
                        let mut with_flag = Vec::with_capacity(1 + new_val.len());
                        with_flag.push(crate::sstable::FLAG_INLINE);
                        with_flag.extend_from_slice(&new_val);
                        finalized_entries.push((key.clone(), Bytes::from(with_flag)));
                    }
                }

                // Advance to next key
                i = j;
            }
        } else {
            // Fast path with MVCC GC: Group by user_key, apply GC rules
            Self::apply_mvcc_gc(&all_entries, oldest_snapshot, &mut finalized_entries);
        }

        Ok(Self {
            entries: finalized_entries.into_iter(),
        })
    }

    /// Apply MVCC garbage collection to sorted entries
    ///
    /// Groups entries by user_key and keeps:
    /// - Newest version (always)
    /// - Versions with seq >= oldest_snapshot (visible to snapshots)
    fn apply_mvcc_gc(
        all_entries: &[(Bytes, Bytes, usize)],
        oldest_snapshot: u64,
        finalized_entries: &mut Vec<(Bytes, Bytes)>,
    ) {
        if all_entries.is_empty() {
            return;
        }

        let mut i = 0;
        while i < all_entries.len() {
            let (encoded_key, value, _) = &all_entries[i];

            // Try to decode as InternalKey
            if let Some(ikey) = InternalKey::decode(encoded_key.clone()) {
                let user_key = &ikey.user_key;

                // Find all versions of this user_key
                let mut j = i + 1;
                while j < all_entries.len() {
                    if let Some(next_ikey) = InternalKey::decode(all_entries[j].0.clone()) {
                        if next_ikey.user_key != *user_key {
                            break;
                        }
                    } else {
                        // Not an InternalKey, different key
                        break;
                    }
                    j += 1;
                }

                // Process all versions of this user_key [i..j)
                // First entry (i) is newest due to sort order (seq DESC)
                let mut kept_newest = false;

                for (enc_key, val, _) in &all_entries[i..j] {
                    if let Some(ver_ikey) = InternalKey::decode(enc_key.clone()) {
                        // Keep if:
                        // 1. This is the newest version (first one), OR
                        // 2. seq >= oldest_snapshot (visible to active snapshot)
                        if !kept_newest || ver_ikey.seq >= oldest_snapshot {
                            finalized_entries.push((enc_key.clone(), val.clone()));
                            kept_newest = true;
                        }
                        // else: GC this old version
                    }
                }

                i = j;
            } else {
                // Not an InternalKey (legacy format) - keep with simple dedup
                finalized_entries.push((encoded_key.clone(), value.clone()));

                // Skip duplicates of the same non-MVCC key
                let mut j = i + 1;
                while j < all_entries.len() && all_entries[j].0 == *encoded_key {
                    j += 1;
                }
                i = j;
            }
        }
    }
}

impl Iterator for MergeIterator {
    type Item = Result<(Bytes, Bytes), SSTableError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(Ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::SSTableBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_merge_single_sstable() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sst");

        // Build single SSTable
        let mut builder = SSTableBuilder::create(&path).unwrap();
        builder
            .add(Bytes::from("key1"), Bytes::from("value1"))
            .unwrap();
        builder
            .add(Bytes::from("key2"), Bytes::from("value2"))
            .unwrap();
        builder
            .add(Bytes::from("key3"), Bytes::from("value3"))
            .unwrap();
        builder.finish().unwrap();
        let sstable = SSTable::open(&path).unwrap();

        // Merge (single iterator)
        let mut merge = MergeIterator::new(vec![sstable], 0, None).unwrap();

        let (k, v) = merge.next().unwrap().unwrap();
        assert_eq!(k, Bytes::from("key1"));
        // MergeIterator returns FLAG-prefixed values (for compaction)
        assert_eq!(v[0], crate::sstable::FLAG_INLINE);
        assert_eq!(&v[1..], b"value1");

        let (k, v) = merge.next().unwrap().unwrap();
        assert_eq!(k, Bytes::from("key2"));
        assert_eq!(v[0], crate::sstable::FLAG_INLINE);
        assert_eq!(&v[1..], b"value2");

        let (k, v) = merge.next().unwrap().unwrap();
        assert_eq!(k, Bytes::from("key3"));
        assert_eq!(v[0], crate::sstable::FLAG_INLINE);
        assert_eq!(&v[1..], b"value3");

        assert!(merge.next().is_none());
    }

    #[test]
    fn test_merge_two_sstables() {
        let dir = tempdir().unwrap();

        // Build first SSTable
        let path1 = dir.path().join("test1.sst");
        let mut builder1 = SSTableBuilder::create(&path1).unwrap();
        builder1
            .add(Bytes::from("key1"), Bytes::from("value1"))
            .unwrap();
        builder1
            .add(Bytes::from("key3"), Bytes::from("value3"))
            .unwrap();
        builder1.finish().unwrap();
        let sstable1 = SSTable::open(&path1).unwrap();

        // Build second SSTable
        let path2 = dir.path().join("test2.sst");
        let mut builder2 = SSTableBuilder::create(&path2).unwrap();
        builder2
            .add(Bytes::from("key2"), Bytes::from("value2"))
            .unwrap();
        builder2
            .add(Bytes::from("key4"), Bytes::from("value4"))
            .unwrap();
        builder2.finish().unwrap();
        let sstable2 = SSTable::open(&path2).unwrap();

        // Merge
        let mut merge = MergeIterator::new(vec![sstable1, sstable2], 0, None).unwrap();

        let entries: Vec<_> = std::iter::from_fn(|| merge.next())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].0, Bytes::from("key1"));
        assert_eq!(entries[1].0, Bytes::from("key2"));
        assert_eq!(entries[2].0, Bytes::from("key3"));
        assert_eq!(entries[3].0, Bytes::from("key4"));
    }

    #[test]
    fn test_merge_with_duplicates() {
        let dir = tempdir().unwrap();

        // Build first SSTable (older - like L0 order: older SSTables come first)
        let path1 = dir.path().join("test1.sst");
        let mut builder1 = SSTableBuilder::create(&path1).unwrap();
        builder1
            .add(Bytes::from("key1"), Bytes::from("old_value1"))
            .unwrap();
        builder1
            .add(Bytes::from("key3"), Bytes::from("value3"))
            .unwrap();
        builder1.finish().unwrap();
        let sstable1 = SSTable::open(&path1).unwrap();

        // Build second SSTable (newer - later in L0, higher index)
        let path2 = dir.path().join("test2.sst");
        let mut builder2 = SSTableBuilder::create(&path2).unwrap();
        builder2
            .add(Bytes::from("key1"), Bytes::from("new_value1"))
            .unwrap();
        builder2
            .add(Bytes::from("key2"), Bytes::from("new_value2"))
            .unwrap();
        builder2.finish().unwrap();
        let sstable2 = SSTable::open(&path2).unwrap();

        // Merge (older first, newer second - like L0 order)
        let mut merge = MergeIterator::new(vec![sstable1, sstable2], 0, None).unwrap();

        let entries: Vec<_> = std::iter::from_fn(|| merge.next())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0, Bytes::from("key1"));
        // MergeIterator returns FLAG-prefixed values (for compaction)
        assert_eq!(entries[0].1[0], crate::sstable::FLAG_INLINE);
        assert_eq!(&entries[0].1[1..], b"new_value1"); // Keeps newer (higher source_id)
        assert_eq!(entries[1].0, Bytes::from("key2"));
        assert_eq!(entries[2].0, Bytes::from("key3"));
    }

    #[test]
    fn test_merge_many_sstables() {
        let dir = tempdir().unwrap();
        let mut sstables = Vec::new();

        // Build 5 SSTables with interleaved keys
        for i in 0..5 {
            let path = dir.path().join(format!("test{}.sst", i));
            let mut builder = SSTableBuilder::create(&path).unwrap();

            for j in 0..10 {
                let key = format!("key_{:03}", i + j * 5);
                let value = format!("value_{}", i + j * 5);
                builder.add(Bytes::from(key), Bytes::from(value)).unwrap();
            }

            builder.finish().unwrap();
            let sstable = SSTable::open(&path).unwrap();
            sstables.push(sstable);
        }

        // Merge all
        let mut merge = MergeIterator::new(sstables, 0, None).unwrap();

        let mut count = 0;
        let mut last_key = None;

        while let Some(Ok((key, _value))) = merge.next() {
            // Verify sorted order
            if let Some(ref last) = last_key {
                assert!(key > *last, "Keys not in sorted order");
            }
            last_key = Some(key);
            count += 1;
        }

        assert_eq!(count, 50); // 5 SSTables * 10 keys each
    }

    // Test compaction filter
    #[derive(Debug)]
    struct TestFilter;

    impl CompactionFilter for TestFilter {
        fn filter(&self, _level: usize, key: &[u8], value: &[u8]) -> FilterDecision {
            // Remove key "remove_me"
            if key == b"remove_me" {
                return FilterDecision::Remove;
            }

            // Change value for "change_me"
            if key == b"change_me" {
                return FilterDecision::ChangeValue(b"changed".to_vec());
            }

            // Check value content (skip flag byte)
            if value.len() > 1 && &value[1..] == b"filter_me" {
                return FilterDecision::Remove;
            }

            FilterDecision::Keep
        }

        fn merge(&self, _level: usize, key: &[u8], values: &[&[u8]]) -> Option<Vec<u8>> {
            if key == b"merge_me" {
                // Concat all values (skipping flags)
                let mut merged = Vec::new();
                for v in values {
                    if v.len() > 1 {
                        merged.extend_from_slice(&v[1..]);
                    }
                }
                return Some(merged);
            }
            None
        }
    }

    #[test]
    fn test_merge_with_filter() {
        let dir = tempdir().unwrap();

        let path1 = dir.path().join("test1.sst");
        let mut builder1 = SSTableBuilder::create(&path1).unwrap();
        builder1
            .add(Bytes::from("keep_me"), Bytes::from("val1"))
            .unwrap();
        builder1
            .add(Bytes::from("remove_me"), Bytes::from("val2"))
            .unwrap();
        builder1
            .add(Bytes::from("change_me"), Bytes::from("val3"))
            .unwrap();
        builder1
            .add(Bytes::from("filter_by_val"), Bytes::from("filter_me"))
            .unwrap();
        builder1
            .add(Bytes::from("merge_me"), Bytes::from("part1"))
            .unwrap();
        builder1.finish().unwrap();
        let sstable1 = SSTable::open(&path1).unwrap();

        let path2 = dir.path().join("test2.sst");
        let mut builder2 = SSTableBuilder::create(&path2).unwrap();
        builder2
            .add(Bytes::from("merge_me"), Bytes::from("part2"))
            .unwrap();
        builder2.finish().unwrap();
        let sstable2 = SSTable::open(&path2).unwrap();

        let filter = Arc::new(TestFilter);
        let mut merge = MergeIterator::new(vec![sstable1, sstable2], 0, Some(filter)).unwrap();

        let entries: Vec<_> = std::iter::from_fn(|| merge.next())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // "change_me" -> "changed"
        // "filter_by_val" -> Removed
        // "keep_me" -> "val1"
        // "merge_me" -> "part2part1" (newest first, so part2 then part1)
        // "remove_me" -> Removed

        assert_eq!(entries.len(), 3);

        // change_me
        assert_eq!(entries[0].0, Bytes::from("change_me"));
        assert_eq!(&entries[0].1[1..], b"changed");

        // keep_me
        assert_eq!(entries[1].0, Bytes::from("keep_me"));
        assert_eq!(&entries[1].1[1..], b"val1");

        // merge_me
        assert_eq!(entries[2].0, Bytes::from("merge_me"));
        // part2 is newer (sstable2), part1 is older (sstable1)
        // merge receives [part2, part1]
        // our merge logic concatenates them -> part2part1
        assert_eq!(&entries[2].1[1..], b"part2part1");
    }

    #[test]
    fn test_mvcc_gc_no_snapshots() {
        // Test that old versions are GC'd when no snapshots are active
        use crate::types::{InternalKey, ValueType};

        let dir = tempdir().unwrap();
        let path = dir.path().join("mvcc.sst");

        // Build SSTable with multiple versions of same key
        let mut builder = SSTableBuilder::create(&path).unwrap();

        // key1 @ seq 300 (newest)
        let ikey1 = InternalKey::new(Bytes::from("key1"), 300, ValueType::Value);
        builder
            .add_internal(&ikey1, Bytes::from("value_v3"))
            .unwrap();

        // key1 @ seq 200 (middle)
        let ikey2 = InternalKey::new(Bytes::from("key1"), 200, ValueType::Value);
        builder
            .add_internal(&ikey2, Bytes::from("value_v2"))
            .unwrap();

        // key1 @ seq 100 (oldest)
        let ikey3 = InternalKey::new(Bytes::from("key1"), 100, ValueType::Value);
        builder
            .add_internal(&ikey3, Bytes::from("value_v1"))
            .unwrap();

        builder.finish().unwrap();

        let sstable = SSTable::open(&path).unwrap();

        // With no snapshots (oldest_snapshot = u64::MAX), only newest should survive
        let merge = MergeIterator::with_gc(vec![sstable], 0, None, u64::MAX).unwrap();
        let entries: Vec<_> = merge.collect::<Result<Vec<_>, _>>().unwrap();

        // Should have only 1 entry (the newest version)
        assert_eq!(entries.len(), 1, "Expected 1 entry, got {}", entries.len());

        // Decode the key to verify it's the newest version
        let ikey = InternalKey::decode(entries[0].0.clone()).unwrap();
        assert_eq!(ikey.user_key, Bytes::from("key1"));
        assert_eq!(ikey.seq, 300);
    }

    #[test]
    fn test_mvcc_gc_with_snapshot() {
        // Test that versions needed by snapshots are preserved
        use crate::types::{InternalKey, ValueType};

        let dir = tempdir().unwrap();
        let path = dir.path().join("mvcc.sst");

        // Build SSTable with multiple versions
        let mut builder = SSTableBuilder::create(&path).unwrap();

        // key1 @ seq 300 (newest)
        let ikey1 = InternalKey::new(Bytes::from("key1"), 300, ValueType::Value);
        builder
            .add_internal(&ikey1, Bytes::from("value_v3"))
            .unwrap();

        // key1 @ seq 200 (middle)
        let ikey2 = InternalKey::new(Bytes::from("key1"), 200, ValueType::Value);
        builder
            .add_internal(&ikey2, Bytes::from("value_v2"))
            .unwrap();

        // key1 @ seq 100 (oldest)
        let ikey3 = InternalKey::new(Bytes::from("key1"), 100, ValueType::Value);
        builder
            .add_internal(&ikey3, Bytes::from("value_v1"))
            .unwrap();

        builder.finish().unwrap();

        let sstable = SSTable::open(&path).unwrap();

        // Snapshot at seq 150 means we need seq >= 150
        // So seq 300 and seq 200 should survive, seq 100 should be GC'd
        let merge = MergeIterator::with_gc(vec![sstable], 0, None, 150).unwrap();
        let entries: Vec<_> = merge.collect::<Result<Vec<_>, _>>().unwrap();

        // Should have 2 entries (seq 300 and seq 200)
        assert_eq!(
            entries.len(),
            2,
            "Expected 2 entries, got {}",
            entries.len()
        );

        // Verify the sequences
        let ikey0 = InternalKey::decode(entries[0].0.clone()).unwrap();
        let ikey1 = InternalKey::decode(entries[1].0.clone()).unwrap();

        assert_eq!(ikey0.seq, 300);
        assert_eq!(ikey1.seq, 200);
    }

    #[test]
    fn test_mvcc_gc_preserves_different_keys() {
        // Test that GC correctly handles multiple different keys
        use crate::types::{InternalKey, ValueType};

        let dir = tempdir().unwrap();
        let path = dir.path().join("mvcc.sst");

        let mut builder = SSTableBuilder::create(&path).unwrap();

        // key1 @ seq 200
        let ikey1 = InternalKey::new(Bytes::from("key1"), 200, ValueType::Value);
        builder
            .add_internal(&ikey1, Bytes::from("key1_v2"))
            .unwrap();

        // key1 @ seq 100
        let ikey2 = InternalKey::new(Bytes::from("key1"), 100, ValueType::Value);
        builder
            .add_internal(&ikey2, Bytes::from("key1_v1"))
            .unwrap();

        // key2 @ seq 150
        let ikey3 = InternalKey::new(Bytes::from("key2"), 150, ValueType::Value);
        builder
            .add_internal(&ikey3, Bytes::from("key2_v1"))
            .unwrap();

        builder.finish().unwrap();

        let sstable = SSTable::open(&path).unwrap();

        // No snapshots - should keep only newest of each key
        let merge = MergeIterator::with_gc(vec![sstable], 0, None, u64::MAX).unwrap();
        let entries: Vec<_> = merge.collect::<Result<Vec<_>, _>>().unwrap();

        // Should have 2 entries (newest of key1, and key2)
        assert_eq!(
            entries.len(),
            2,
            "Expected 2 entries, got {}",
            entries.len()
        );

        let ikey0 = InternalKey::decode(entries[0].0.clone()).unwrap();
        let ikey1 = InternalKey::decode(entries[1].0.clone()).unwrap();

        assert_eq!(ikey0.user_key, Bytes::from("key1"));
        assert_eq!(ikey0.seq, 200);

        assert_eq!(ikey1.user_key, Bytes::from("key2"));
        assert_eq!(ikey1.seq, 150);
    }
}
