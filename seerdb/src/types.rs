use bytes::{BufMut, Bytes, BytesMut};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// Value type for Internal Key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueType {
    /// A live deletion (tombstone)
    Deletion = 0x00,
    /// A standard value
    Value = 0x01,
    /// A merge operand
    Merge = 0x02,
    /// Log data (not stored in memtable usually)
    Log = 0x03,
}

impl ValueType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x00 => Some(ValueType::Deletion),
            0x01 => Some(ValueType::Value),
            0x02 => Some(ValueType::Merge),
            0x03 => Some(ValueType::Log),
            _ => None,
        }
    }
}

/// Internal Key used for MVCC (Multi-Version Concurrency Control)
///
/// Format: [ User Key ] [ 8 bytes: (SeqNum << 8) | ValueType ]
///
/// Sorting:
/// 1. User Key (Ascending)
/// 2. Sequence Number (Descending) - Latest version comes first
#[derive(Debug, Clone, Eq)]
pub struct InternalKey {
    pub user_key: Bytes,
    pub seq: u64,
    pub kind: ValueType,
}

impl InternalKey {
    pub fn new(user_key: Bytes, seq: u64, kind: ValueType) -> Self {
        Self {
            user_key,
            seq,
            kind,
        }
    }

    /// Create a search key that will match the latest version of a user key
    /// (Sequence number = MAX)
    pub fn for_lookup(user_key: Bytes) -> Self {
        Self {
            user_key,
            seq: u64::MAX,
            kind: ValueType::Value, // Kind doesn't matter for lookup start
        }
    }

    /// Encode the InternalKey into bytes for storage
    /// Uses big-endian encoding for the sequence number wrapper to preserve sort order
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(self.user_key.len() + 8);
        buf.extend_from_slice(&self.user_key);

        // Pack Seq + Type into 64 bits
        // We use 56 bits for Seq, 8 bits for Type.
        // This limits Seq to ~72 quadrillion (plenty).
        // To sort Descending, we store (MAX - Seq).
        let packed = (self.seq << 8) | (self.kind as u64);
        let inverted = !packed; // Bitwise NOT reverses the order

        buf.put_u64(inverted);
        buf.freeze()
    }

    /// Decode an InternalKey from bytes
    pub fn decode(bytes: Bytes) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }

        let split_idx = bytes.len() - 8;
        let user_key = bytes.slice(..split_idx);
        let trailer = bytes.slice(split_idx..);

        let inverted = u64::from_be_bytes(trailer.as_ref().try_into().ok()?);
        let packed = !inverted;

        let kind_u8 = (packed & 0xFF) as u8;
        let seq = packed >> 8;

        let kind = ValueType::from_u8(kind_u8)?;

        Some(Self {
            user_key,
            seq,
            kind,
        })
    }

    /// Extract just the user key from an encoded buffer (zero copy)
    /// If the key is shorter than 9 bytes (min: 1 byte user key + 8 byte trailer),
    /// returns the key unchanged (assumes it's a plain user key, not an InternalKey).
    pub fn extract_user_key(bytes: &Bytes) -> Bytes {
        if bytes.len() <= 8 {
            // Too short to be an InternalKey - treat as plain user key
            return bytes.clone();
        }
        bytes.slice(..bytes.len() - 8)
    }
}

impl PartialEq for InternalKey {
    fn eq(&self, other: &Self) -> bool {
        self.user_key == other.user_key && self.seq == other.seq && self.kind == other.kind
    }
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Compare User Key (Ascending)
        match self.user_key.cmp(&other.user_key) {
            Ordering::Equal => {
                // 2. Compare Sequence Number (Descending)
                // Higher sequence number should come FIRST (Less)
                other.seq.cmp(&self.seq)
            }
            ord => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_key_encoding() {
        let key = InternalKey::new(Bytes::from("key"), 100, ValueType::Value);
        let encoded = key.encode();
        let decoded = InternalKey::decode(encoded).unwrap();

        assert_eq!(decoded.user_key, Bytes::from("key"));
        assert_eq!(decoded.seq, 100);
        assert_eq!(decoded.kind, ValueType::Value);
    }

    #[test]
    fn test_internal_key_sorting() {
        // key v2 (latest)
        let k2 = InternalKey::new(Bytes::from("abc"), 200, ValueType::Value);
        // key v1 (older)
        let k1 = InternalKey::new(Bytes::from("abc"), 100, ValueType::Value);
        // key b (different key)
        let kb = InternalKey::new(Bytes::from("abd"), 100, ValueType::Value);

        // k2 should be "Less" than k1 because it's newer (descending sort)
        assert_eq!(k2.cmp(&k1), Ordering::Less);

        // k1 should be "Less" than kb because "abc" < "abd"
        assert_eq!(k1.cmp(&kb), Ordering::Less);
    }

    #[test]
    fn test_encoded_byte_sorting() {
        let k2 = InternalKey::new(Bytes::from("abc"), 200, ValueType::Value);
        let k1 = InternalKey::new(Bytes::from("abc"), 100, ValueType::Value);

        let e2 = k2.encode();
        let e1 = k1.encode();

        // Raw byte comparison should match logical comparison
        assert!(e2 < e1);
    }
}

/// Tracks active snapshot sequence numbers for MVCC garbage collection.
///
/// During compaction, old versions of keys can be garbage collected only if
/// no active snapshot needs them. This tracker maintains the set of active
/// snapshot sequence numbers and provides O(1) access to the oldest one.
///
/// # Thread Safety
/// All operations are thread-safe. The `oldest_snapshot()` method uses an
/// atomic cache to avoid lock contention during compaction.
pub struct SnapshotTracker {
    /// Active snapshot sequence numbers (BTreeSet for O(1) min lookup)
    active: Mutex<BTreeSet<u64>>,
    /// Cached oldest snapshot for lock-free access during compaction
    oldest_cached: AtomicU64,
}

impl SnapshotTracker {
    /// Create a new snapshot tracker with no active snapshots.
    pub fn new() -> Self {
        Self {
            active: Mutex::new(BTreeSet::new()),
            oldest_cached: AtomicU64::new(u64::MAX), // No snapshots = GC everything
        }
    }

    /// Register a new snapshot with the given sequence number.
    /// Called when `DB::snapshot()` creates a new snapshot.
    pub fn register(&self, seq: u64) {
        let mut active = self.active.lock().expect("SnapshotTracker lock poisoned");
        active.insert(seq);
        // Update cached oldest (first element in BTreeSet is minimum)
        if let Some(&oldest) = active.iter().next() {
            self.oldest_cached.store(oldest, AtomicOrdering::Release);
        }
    }

    /// Unregister a snapshot when it's dropped.
    /// Called from `Snapshot::drop()`.
    pub fn unregister(&self, seq: u64) {
        let mut active = self.active.lock().expect("SnapshotTracker lock poisoned");
        active.remove(&seq);
        // Update cached oldest (u64::MAX if no snapshots)
        let oldest = active.iter().next().copied().unwrap_or(u64::MAX);
        self.oldest_cached.store(oldest, AtomicOrdering::Release);
    }

    /// Get the oldest active snapshot sequence number.
    ///
    /// Returns `u64::MAX` if no snapshots are active, meaning all old versions
    /// can be garbage collected during compaction.
    ///
    /// This is a lock-free operation using the cached value.
    pub fn oldest_snapshot(&self) -> u64 {
        self.oldest_cached.load(AtomicOrdering::Acquire)
    }

    /// Get the number of active snapshots (for debugging/metrics).
    pub fn active_count(&self) -> usize {
        self.active
            .lock()
            .expect("SnapshotTracker lock poisoned")
            .len()
    }
}

impl Default for SnapshotTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for automatic snapshot unregistration.
///
/// When this handle is dropped, it automatically unregisters the snapshot
/// from the tracker. This is held by `Snapshot` to ensure proper cleanup.
pub struct SnapshotHandle {
    seq: u64,
    tracker: Arc<SnapshotTracker>,
}

impl SnapshotHandle {
    /// Create a new handle that will unregister on drop.
    pub fn new(seq: u64, tracker: Arc<SnapshotTracker>) -> Self {
        tracker.register(seq);
        Self { seq, tracker }
    }

    /// Get the sequence number this handle tracks.
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

impl Drop for SnapshotHandle {
    fn drop(&mut self) {
        self.tracker.unregister(self.seq);
    }
}

#[cfg(test)]
mod snapshot_tracker_tests {
    use super::*;

    #[test]
    fn test_tracker_empty() {
        let tracker = SnapshotTracker::new();
        assert_eq!(tracker.oldest_snapshot(), u64::MAX);
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_tracker_single_snapshot() {
        let tracker = SnapshotTracker::new();
        tracker.register(100);
        assert_eq!(tracker.oldest_snapshot(), 100);
        assert_eq!(tracker.active_count(), 1);

        tracker.unregister(100);
        assert_eq!(tracker.oldest_snapshot(), u64::MAX);
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_tracker_multiple_snapshots() {
        let tracker = SnapshotTracker::new();
        tracker.register(100);
        tracker.register(200);
        tracker.register(50);

        // Oldest should be 50
        assert_eq!(tracker.oldest_snapshot(), 50);
        assert_eq!(tracker.active_count(), 3);

        // Remove oldest, next oldest is 100
        tracker.unregister(50);
        assert_eq!(tracker.oldest_snapshot(), 100);

        // Remove 200, oldest still 100
        tracker.unregister(200);
        assert_eq!(tracker.oldest_snapshot(), 100);

        // Remove last
        tracker.unregister(100);
        assert_eq!(tracker.oldest_snapshot(), u64::MAX);
    }

    #[test]
    fn test_snapshot_handle_auto_unregister() {
        let tracker = Arc::new(SnapshotTracker::new());

        {
            let _handle = SnapshotHandle::new(100, Arc::clone(&tracker));
            assert_eq!(tracker.oldest_snapshot(), 100);
            assert_eq!(tracker.active_count(), 1);
        } // handle dropped here

        assert_eq!(tracker.oldest_snapshot(), u64::MAX);
        assert_eq!(tracker.active_count(), 0);
    }
}
