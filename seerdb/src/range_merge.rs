// K-way merge iterator for range scans
// Merges sorted iterators from memtable + multiple SSTable levels using a min-heap

use bytes::Bytes;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::memtable::Entry;
use crate::MergeOperator;

// Use SIMD-accelerated comparison when available, fallback to standard otherwise
#[cfg(feature = "simd")]
use crate::simd;

#[cfg(not(feature = "simd"))]
mod simd {
    use std::cmp::Ordering;
    #[inline]
    pub fn compare_keys(a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }
}

/// Entry in the min-heap for k-way merge
struct HeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    key: Bytes,
    entry: Entry,
    level: usize, // Lower = newer (for LSM semantics)
    iter: I,
}

impl<I> Ord for HeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: use Reverse wrapper, so normal comparison
        // Primary: by key (ascending) - using SIMD-accelerated comparison!
        // Secondary: by level (ascending, so lower level = newer wins)
        match simd::compare_keys(&self.key, &other.key) {
            Ordering::Equal => self.level.cmp(&other.level),
            ord => ord,
        }
    }
}

impl<I> PartialOrd for HeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I> Eq for HeapEntry<I> where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>
{
}

impl<I> PartialEq for HeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.level == other.level
    }
}

/// K-way merge iterator
/// Lazily merges sorted iterators from multiple levels using a min-heap
pub struct KWayMergeIterator<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    heap: BinaryHeap<Reverse<HeapEntry<I>>>,
    last_key: Option<Bytes>,
    merge_operator: Option<Arc<dyn MergeOperator>>,
    pending_operands: Vec<Bytes>,
}

impl<I> KWayMergeIterator<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    /// Create a new k-way merge iterator
    /// Iterators should be provided in order: [memtable, L0, L1, ... LN]
    /// Level 0 (memtable) is newest, higher levels are older
    pub fn new(
        iterators: Vec<I>,
        merge_operator: Option<Arc<dyn MergeOperator>>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut heap = BinaryHeap::new();

        // Prime the heap with first entry from each iterator
        for (level, mut iter) in iterators.into_iter().enumerate() {
            if let Some(result) = iter.next() {
                match result {
                    Ok((key, entry)) => {
                        heap.push(Reverse(HeapEntry {
                            key,
                            entry,
                            level,
                            iter,
                        }));
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(Self {
            heap,
            last_key: None,
            merge_operator,
            pending_operands: Vec::new(),
        })
    }

    #[inline]
    fn resolve_merges(
        &mut self,
        key: &Bytes,
        base: Option<&Bytes>,
    ) -> Result<Entry, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(op) = &self.merge_operator {
            // Operands are stored newest-first (as we encountered them), but MergeOperator

            // typically expects them in chronological order (oldest-first) to apply them correctly.

            // e.g. "Value" -> Merge(A) -> Merge(B).

            // We see B then A. pending_operands = [B, A].

            // We want to apply A then B.

            let ops: Vec<&[u8]> = self
                .pending_operands
                .iter()
                .rev()
                .map(|b| b.as_ref())
                .collect();

            match op.full_merge(key, base.map(|b| b.as_ref()), &ops) {
                Some(res) => Ok(Entry::Value(Bytes::from(res))),

                None => Ok(Entry::Tombstone),
            }
        } else {
            // No merge operator - can't resolve.

            // Fallback: return the newest merge operand as if it were a value?

            // Or just fail? existing get() returns the raw bytes.

            if let Some(first) = self.pending_operands.first() {
                Ok(Entry::Merge(vec![first.clone()]))
            } else {
                Ok(Entry::Tombstone)
            }
        }
    }
}

impl<I> Iterator for KWayMergeIterator<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    type Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Pop smallest (newest) key/level from heap
            let Reverse(mut entry) = self.heap.pop()?;

            // Advance this iterator and push back into heap immediately
            if let Some(result) = entry.iter.next() {
                match result {
                    Ok((next_key, next_entry)) => {
                        self.heap.push(Reverse(HeapEntry {
                            key: next_key,
                            entry: next_entry,
                            level: entry.level,
                            iter: entry.iter,
                        }));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            // Check if this is a duplicate (older version) of the last key we PROCESSED
            if self.last_key.as_ref().is_some_and(|last| entry.key == last) {
                // We are seeing an older version of the same key.
                // Since we fully resolve keys in one go (see below), this implies
                // we have already emitted the resolved value for this key.
                // So we should skip this entry.
                continue;
            }

            // It is a NEW key.
            self.last_key = Some(entry.key.clone());
            let current_key = entry.key.clone();

            match entry.entry {
                Entry::Value(_) | Entry::Tombstone => {
                    // Fast path: Newest version is absolute (Value or Tombstone).
                    // No merging needed.
                    // We just emit it. Subsequent older versions will be caught by the check above.
                    if let Entry::Tombstone = entry.entry {
                        // Skip tombstones in output (don't return deleted keys)
                        continue;
                    }
                    return Some(Ok((entry.key, entry.entry)));
                }
                Entry::Merge(operand) => {
                    // Newest version is a Merge.
                    // We must accumulate all merge operands for this key from older versions.
                    self.pending_operands.clear();
                    self.pending_operands.extend(operand.iter().rev().cloned());

                    // Look ahead in the heap for older versions of THIS key
                    let mut base_found = false;
                    let mut resolved_entry: Option<Entry> = None;

                    loop {
                        // Peek at next item
                        let is_same_key = if let Some(Reverse(next)) = self.heap.peek() {
                            next.key == current_key
                        } else {
                            false
                        };

                        if !is_same_key {
                            break; // No more versions of this key
                        }

                        // Pop older version
                        // Safe: we just peeked and confirmed existence
                        let Reverse(mut next_entry) =
                            self.heap.pop().expect("heap not empty after peek");

                        // Advance iterator
                        if let Some(result) = next_entry.iter.next() {
                            match result {
                                Ok((nk, ne)) => {
                                    self.heap.push(Reverse(HeapEntry {
                                        key: nk,
                                        entry: ne,
                                        level: next_entry.level,
                                        iter: next_entry.iter,
                                    }));
                                }
                                Err(e) => return Some(Err(e)),
                            }
                        }

                        match next_entry.entry {
                            Entry::Value(val) => {
                                // Found base value. Resolve.
                                match self.resolve_merges(&current_key, Some(&val)) {
                                    Ok(res) => resolved_entry = Some(res),
                                    Err(e) => return Some(Err(e)),
                                }
                                base_found = true;
                                break;
                            }
                            Entry::Tombstone => {
                                // Found base tombstone. Resolve against None.
                                match self.resolve_merges(&current_key, None) {
                                    Ok(res) => resolved_entry = Some(res),
                                    Err(e) => return Some(Err(e)),
                                }
                                base_found = true;
                                break;
                            }
                            Entry::Merge(op) => {
                                // Another merge operand (older). Stack it.
                                self.pending_operands.extend(op.iter().rev().cloned());
                            }
                        }
                    } // end inner loop

                    if !base_found {
                        // Ran out of versions without finding base Value/Tombstone.
                        // Assume base is None (fresh key).
                        match self.resolve_merges(&current_key, None) {
                            Ok(res) => resolved_entry = Some(res),
                            Err(e) => return Some(Err(e)),
                        }
                    }

                    // Emit result
                    if let Some(final_entry) = resolved_entry {
                        if let Entry::Tombstone = final_entry {
                            continue; // Result is deleted
                        }
                        return Some(Ok((current_key, final_entry)));
                    }
                }
            }
        }
    }
}

/// Wrapper for HeapEntry to implement Max-Heap ordering with correct level priority
struct RevHeapEntry<I>(HeapEntry<I>)
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>;

impl<I> Ord for RevHeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a Max-Heap that yields Largest Key first.
        // If keys are equal, we want Newest Level (Lowest Level) first.

        // Compare Keys: Normal comparison. A < B. MaxHeap yields B. Correct.
        match simd::compare_keys(&self.0.key, &other.0.key) {
            Ordering::Equal => {
                // Compare Levels:
                // We want Level 0 to be "Larger" than Level 1 so it is popped first.
                // 0 < 1.
                // So we must reverse the level comparison.
                other.0.level.cmp(&self.0.level)
            }
            ord => ord,
        }
    }
}

impl<I> PartialOrd for RevHeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I> Eq for RevHeapEntry<I> where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>
{
}

impl<I> PartialEq for RevHeapEntry<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.key == other.0.key && self.0.level == other.0.level
    }
}

/// K-way merge iterator for REVERSE iteration
pub struct KWayMergeIteratorRev<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    heap: BinaryHeap<RevHeapEntry<I>>,
    last_key: Option<Bytes>,
    merge_operator: Option<Arc<dyn MergeOperator>>,
    pending_operands: Vec<Bytes>,
}

impl<I> KWayMergeIteratorRev<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    pub fn new(
        iterators: Vec<I>,
        merge_operator: Option<Arc<dyn MergeOperator>>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut heap = BinaryHeap::new();

        for (level, mut iter) in iterators.into_iter().enumerate() {
            if let Some(result) = iter.next() {
                match result {
                    Ok((key, entry)) => {
                        heap.push(RevHeapEntry(HeapEntry {
                            key,
                            entry,
                            level,
                            iter,
                        }));
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(Self {
            heap,
            last_key: None,
            merge_operator,
            pending_operands: Vec::new(),
        })
    }

    // Copied from KWayMergeIterator (TODO: Refactor to shared trait?)
    #[inline]
    fn resolve_merges(
        &mut self,
        key: &Bytes,
        base: Option<&Bytes>,
    ) -> Result<Entry, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(op) = &self.merge_operator {
            let ops: Vec<&[u8]> = self
                .pending_operands
                .iter()
                .rev()
                .map(|b| b.as_ref())
                .collect();

            match op.full_merge(key, base.map(|b| b.as_ref()), &ops) {
                Some(res) => Ok(Entry::Value(Bytes::from(res))),
                None => Ok(Entry::Tombstone),
            }
        } else if let Some(first) = self.pending_operands.first() {
            Ok(Entry::Merge(vec![first.clone()]))
        } else {
            Ok(Entry::Tombstone)
        }
    }
}

impl<I> Iterator for KWayMergeIteratorRev<I>
where
    I: Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>,
{
    type Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Pop largest (newest) key/level from heap
            let RevHeapEntry(mut entry) = self.heap.pop()?;

            // Advance iterator
            if let Some(result) = entry.iter.next() {
                match result {
                    Ok((next_key, next_entry)) => {
                        self.heap.push(RevHeapEntry(HeapEntry {
                            key: next_key,
                            entry: next_entry,
                            level: entry.level,
                            iter: entry.iter,
                        }));
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            // Check duplicate (last PROCESSED key)
            if self
                .last_key
                .as_ref()
                .is_some_and(|last| entry.key == *last)
            {
                continue;
            }

            self.last_key = Some(entry.key.clone());
            let current_key = entry.key.clone();

            match entry.entry {
                Entry::Value(_) | Entry::Tombstone => {
                    if let Entry::Tombstone = entry.entry {
                        continue;
                    }
                    return Some(Ok((entry.key, entry.entry)));
                }
                Entry::Merge(operand) => {
                    // Identical merge logic
                    self.pending_operands.clear();
                    self.pending_operands.extend(operand.iter().rev().cloned());

                    let mut base_found = false;
                    let mut resolved_entry: Option<Entry> = None;

                    loop {
                        let is_same_key = if let Some(next) = self.heap.peek() {
                            next.0.key == current_key
                        } else {
                            false
                        };

                        if !is_same_key {
                            break;
                        }

                        let RevHeapEntry(mut next_entry) = self
                            .heap
                            .pop()
                            .expect("heap should not be empty during merge iteration");

                        if let Some(result) = next_entry.iter.next() {
                            match result {
                                Ok((nk, ne)) => {
                                    self.heap.push(RevHeapEntry(HeapEntry {
                                        key: nk,
                                        entry: ne,
                                        level: next_entry.level,
                                        iter: next_entry.iter,
                                    }));
                                }
                                Err(e) => return Some(Err(e)),
                            }
                        }

                        match next_entry.entry {
                            Entry::Value(val) => {
                                match self.resolve_merges(&current_key, Some(&val)) {
                                    Ok(res) => resolved_entry = Some(res),
                                    Err(e) => return Some(Err(e)),
                                }
                                base_found = true;
                                break;
                            }
                            Entry::Tombstone => {
                                match self.resolve_merges(&current_key, None) {
                                    Ok(res) => resolved_entry = Some(res),
                                    Err(e) => return Some(Err(e)),
                                }
                                base_found = true;
                                break;
                            }
                            Entry::Merge(op) => {
                                self.pending_operands.extend(op.iter().rev().cloned());
                            }
                        }
                    }

                    if !base_found {
                        match self.resolve_merges(&current_key, None) {
                            Ok(res) => resolved_entry = Some(res),
                            Err(e) => return Some(Err(e)),
                        }
                    }

                    if let Some(final_entry) = resolved_entry {
                        if let Entry::Tombstone = final_entry {
                            continue;
                        }
                        return Some(Ok((current_key, final_entry)));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memtable::Entry;

    fn ok_iter(
        items: Vec<(Bytes, Entry)>,
    ) -> impl Iterator<Item = Result<(Bytes, Entry), Box<dyn std::error::Error + Send + Sync>>>
    {
        items.into_iter().map(Ok)
    }

    #[test]
    fn test_kway_single_iterator() {
        let iter1 = ok_iter(vec![
            (Bytes::from("a"), Entry::Value(Bytes::from("1"))),
            (Bytes::from("b"), Entry::Value(Bytes::from("2"))),
            (Bytes::from("c"), Entry::Value(Bytes::from("3"))),
        ]);

        let mut merge = KWayMergeIterator::new(vec![iter1], None).unwrap();

        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("a"), Entry::Value(Bytes::from("1")))
        );
        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("b"), Entry::Value(Bytes::from("2")))
        );
        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("c"), Entry::Value(Bytes::from("3")))
        );
        assert!(merge.next().is_none());
    }

    #[test]
    fn test_kway_merge_stacking() {
        // Mock MergeOperator? KWayMergeIterator expects Arc<dyn MergeOperator>
        // It's hard to mock traits in simple unit tests without struct impls.
        // We will rely on integration tests for full merge logic.
        // But we can test that it consumes items correctly.

        // With NO merge operator, it should return first merge operand as value?
        // Or treating it as raw merge entry.

        let iter1 = ok_iter(vec![(
            Bytes::from("a"),
            Entry::Merge(vec![Bytes::from("op1")]),
        )]);
        let iter2 = ok_iter(vec![(
            Bytes::from("a"),
            Entry::Merge(vec![Bytes::from("op2")]),
        )]);

        let mut merge = KWayMergeIterator::new(vec![iter1, iter2], None).unwrap();

        // Logic with no operator: returns newest merge operand (op1) as Merge entry
        let result = merge.next().unwrap().unwrap();
        assert_eq!(result.0, Bytes::from("a"));
        // Our fallback logic returns Entry::Merge(vec!["op1"])
        match result.1 {
            Entry::Merge(val) => {
                assert_eq!(val.len(), 1);
                assert_eq!(val[0], Bytes::from("op1"));
            }
            _ => panic!("Expected Merge entry"),
        }
    }

    #[test]
    fn test_kway_merge_rev_simple() {
        // Iterators yielding in REVERSE order
        let iter1 = ok_iter(vec![
            (Bytes::from("c"), Entry::Value(Bytes::from("3"))),
            (Bytes::from("b"), Entry::Value(Bytes::from("2"))),
            (Bytes::from("a"), Entry::Value(Bytes::from("1"))),
        ]);

        let mut merge = KWayMergeIteratorRev::new(vec![iter1], None).unwrap();

        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("c"), Entry::Value(Bytes::from("3")))
        );
        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("b"), Entry::Value(Bytes::from("2")))
        );
        assert_eq!(
            merge.next().unwrap().unwrap(),
            (Bytes::from("a"), Entry::Value(Bytes::from("1")))
        );
        assert!(merge.next().is_none());
    }

    #[test]
    fn test_kway_merge_rev_multi() {
        // L0 (Newer): [c:3]
        // L1 (Older): [b:2, a:1]

        let iter_l0 = ok_iter(vec![(Bytes::from("c"), Entry::Value(Bytes::from("3")))]);
        let iter_l1 = ok_iter(vec![
            (Bytes::from("b"), Entry::Value(Bytes::from("2"))),
            (Bytes::from("a"), Entry::Value(Bytes::from("1"))),
        ]);

        // Iterator list: [L0, L1]
        let mut merge = KWayMergeIteratorRev::new(vec![iter_l0, iter_l1], None).unwrap();

        // Expect: c, b, a
        assert_eq!(merge.next().unwrap().unwrap().0, Bytes::from("c"));
        assert_eq!(merge.next().unwrap().unwrap().0, Bytes::from("b"));
        assert_eq!(merge.next().unwrap().unwrap().0, Bytes::from("a"));
    }

    #[test]
    fn test_kway_merge_rev_overlap() {
        // L0 (Newer): [b:20 (update)]
        // L1 (Older): [b:2, a:1]

        let iter_l0 = ok_iter(vec![(Bytes::from("b"), Entry::Value(Bytes::from("20")))]);
        let iter_l1 = ok_iter(vec![
            (Bytes::from("b"), Entry::Value(Bytes::from("2"))), // Should be shadowed
            (Bytes::from("a"), Entry::Value(Bytes::from("1"))),
        ]);

        let mut merge = KWayMergeIteratorRev::new(vec![iter_l0, iter_l1], None).unwrap();

        // Expect: b (20), a (1)
        let res1 = merge.next().unwrap().unwrap();
        assert_eq!(res1.0, Bytes::from("b"));
        assert_eq!(res1.1, Entry::Value(Bytes::from("20")));

        let res2 = merge.next().unwrap().unwrap();
        assert_eq!(res2.0, Bytes::from("a"));
        assert_eq!(res2.1, Entry::Value(Bytes::from("1")));
    }
}
