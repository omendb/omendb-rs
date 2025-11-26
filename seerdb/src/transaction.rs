//! Optimistic Concurrency Control (OCC) Transactions
//!
//! Provides snapshot isolation with write-write conflict detection at commit time.
//!
//! # Example
//!
//! ```rust,no_run
//! use seerdb::{DB, DBOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let db = DB::open(DBOptions::default())?;
//!
//! // Begin a transaction
//! let mut txn = db.begin_transaction();
//!
//! // Read a value (recorded in read-set for conflict detection)
//! let balance = txn.get(b"account:alice")?;
//!
//! // Buffer writes (not visible until commit)
//! txn.put(b"account:alice", b"100");
//! txn.put(b"account:bob", b"50");
//!
//! // Commit atomically (validates no conflicts, then writes)
//! txn.commit()?;
//! # Ok(())
//! # }
//! ```

use bytes::Bytes;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use crate::db::{DBError, Result, DB};
use crate::types::SnapshotHandle;
use crate::wal::{BatchOp, Record};

/// Error returned when transaction commit fails due to conflicts.
#[derive(Debug, Clone)]
pub struct TransactionConflict {
    /// Keys that had write-write conflicts
    pub conflicting_keys: Vec<Bytes>,
}

impl std::fmt::Display for TransactionConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "transaction conflict on {} key(s)",
            self.conflicting_keys.len()
        )
    }
}

impl std::error::Error for TransactionConflict {}

/// Buffered write operation
#[derive(Clone, Debug)]
enum WriteOp {
    Put(Bytes),
    Delete,
}

/// Optimistic transaction with snapshot isolation.
///
/// Provides read-your-writes semantics within the transaction and validates
/// for conflicts at commit time using Optimistic Concurrency Control (OCC).
///
/// # Isolation Level
///
/// Snapshot Isolation: All reads see a consistent snapshot from when the
/// transaction started. Writes are buffered until commit.
///
/// # Conflict Detection
///
/// At commit time, validates that no key in the read-set has been modified
/// by another transaction since this transaction started. If conflicts are
/// detected, the commit fails with `TransactionConflict`.
pub struct Transaction<'db> {
    db: &'db DB,
    /// Sequence number when transaction started (snapshot point)
    start_seq: u64,
    /// Buffered write operations
    write_buffer: HashMap<Bytes, WriteOp>,
    /// Keys read during this transaction (for OCC validation)
    read_set: HashSet<Bytes>,
    /// Whether transaction is still active
    active: bool,
    /// GC handle to prevent snapshot from being garbage collected
    #[allow(dead_code)]
    gc_handle: SnapshotHandle,
}

impl<'db> Transaction<'db> {
    /// Create a new transaction.
    pub(crate) fn new(db: &'db DB, start_seq: u64, gc_handle: SnapshotHandle) -> Self {
        Self {
            db,
            start_seq,
            write_buffer: HashMap::new(),
            read_set: HashSet::new(),
            active: true,
            gc_handle,
        }
    }

    /// Get a value by key.
    ///
    /// Returns buffered writes first (read-your-writes), then falls back to
    /// the snapshot. Records the key in the read-set for conflict detection.
    ///
    /// # Errors
    ///
    /// Returns error if the transaction has already been committed or aborted.
    pub fn get(&mut self, key: impl AsRef<[u8]>) -> Result<Option<Bytes>> {
        if !self.active {
            return Err(DBError::TransactionAborted);
        }

        let key_bytes = Bytes::copy_from_slice(key.as_ref());

        // Check write buffer first (read-your-writes)
        if let Some(op) = self.write_buffer.get(&key_bytes) {
            return Ok(match op {
                WriteOp::Put(v) => Some(v.clone()),
                WriteOp::Delete => None,
            });
        }

        // Record in read-set for OCC validation
        self.read_set.insert(key_bytes.clone());

        // Read from snapshot
        self.db.get_at_seq(key.as_ref(), self.start_seq)
    }

    /// Put a key-value pair.
    ///
    /// Buffers the write until commit. The value is immediately visible to
    /// subsequent reads within this transaction.
    ///
    /// # Errors
    ///
    /// Returns error if the transaction has already been committed or aborted.
    pub fn put(&mut self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) -> Result<()> {
        if !self.active {
            return Err(DBError::TransactionAborted);
        }

        let key_bytes = Bytes::copy_from_slice(key.as_ref());
        let value_bytes = Bytes::copy_from_slice(value.as_ref());
        self.write_buffer
            .insert(key_bytes, WriteOp::Put(value_bytes));
        Ok(())
    }

    /// Delete a key.
    ///
    /// Buffers the delete until commit. The key appears deleted to subsequent
    /// reads within this transaction.
    ///
    /// # Errors
    ///
    /// Returns error if the transaction has already been committed or aborted.
    pub fn delete(&mut self, key: impl AsRef<[u8]>) -> Result<()> {
        if !self.active {
            return Err(DBError::TransactionAborted);
        }

        let key_bytes = Bytes::copy_from_slice(key.as_ref());
        self.write_buffer.insert(key_bytes, WriteOp::Delete);
        Ok(())
    }

    /// Commit the transaction.
    ///
    /// Validates that no key in the read-set has been modified since the
    /// transaction started, then atomically writes all buffered operations.
    ///
    /// # Errors
    ///
    /// - `DBError::TransactionConflict` if OCC validation fails
    /// - `DBError::TransactionAborted` if already committed/aborted
    /// - `DBError::Wal` if WAL write fails
    pub fn commit(mut self) -> Result<()> {
        if !self.active {
            return Err(DBError::TransactionAborted);
        }
        self.active = false;

        // Early return for empty transactions (no lock needed)
        if self.write_buffer.is_empty() && self.read_set.is_empty() {
            return Ok(());
        }

        // Prepare WAL operations OUTSIDE the lock (reduces critical section)
        let wal_ops: Vec<BatchOp> = self
            .write_buffer
            .iter()
            .map(|(k, op)| match op {
                WriteOp::Put(v) => BatchOp::Put {
                    key: k.clone(),
                    value: v.clone(),
                },
                WriteOp::Delete => BatchOp::Delete { key: k.clone() },
            })
            .collect();

        // Acquire commit lock to serialize validation+write
        // This prevents TOCTOU race where concurrent transactions both pass validation
        // because neither has written yet, leading to lost updates.
        let _commit_guard = self.db.commit_lock.lock().expect("commit lock poisoned");

        // OCC Validation: check for write-write conflicts
        let conflicts = self.validate_read_set()?;
        if !conflicts.is_empty() {
            return Err(DBError::TransactionConflict(TransactionConflict {
                conflicting_keys: conflicts,
            }));
        }

        // No writes - just validation passed
        if wal_ops.is_empty() {
            return Ok(());
        }

        // Get base sequence number (don't increment yet - see below)
        // commit_lock ensures only one commit at a time, so this is safe
        let op_count = wal_ops.len() as u64;
        let base_seq = self.db.next_seq.load(Ordering::SeqCst);

        // Atomic batch write
        let batch_record = Record::Batch {
            base_seq,
            operations: wal_ops,
        };

        self.db
            .pipelined_wal
            .put(batch_record, |records| {
                self.db.apply_wal_records(records);
            })
            .map_err(DBError::Wal)?;

        // Increment next_seq AFTER memtable write completes.
        // This prevents a race where a new transaction could start with
        // start_seq > our commit seq, causing it to miss our writes in validation.
        self.db
            .next_seq
            .store(base_seq + op_count, Ordering::SeqCst);

        // Lock released here when _commit_guard drops
        Ok(())
    }

    /// Abort the transaction, discarding all buffered writes.
    ///
    /// This is a no-op since writes are only buffered. Dropping the transaction
    /// without calling commit() has the same effect.
    pub fn abort(mut self) {
        self.active = false;
        self.write_buffer.clear();
        self.read_set.clear();
    }

    /// Check if the transaction is still active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get the number of buffered write operations.
    pub fn write_count(&self) -> usize {
        self.write_buffer.len()
    }

    /// Get the number of keys in the read-set.
    pub fn read_count(&self) -> usize {
        self.read_set.len()
    }

    /// Validate read-set for OCC conflicts.
    ///
    /// Returns a list of keys that have been modified since transaction start.
    /// A conflict occurs when a key's latest sequence number is >= start_seq,
    /// meaning it was potentially written after the transaction began.
    fn validate_read_set(&self) -> Result<Vec<Bytes>> {
        let mut conflicts = Vec::new();

        for key in &self.read_set {
            // Get the latest sequence number for this key
            if let Some(latest_seq) = self.db.get_latest_seq(key)? {
                // Use >= because start_seq is the next_seq at transaction start,
                // so any write with seq >= start_seq happened concurrently or after
                if latest_seq >= self.start_seq {
                    conflicts.push(key.clone());
                }
            }
        }

        Ok(conflicts)
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        // Just mark as inactive - gc_handle will auto-unregister
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DBOptions;
    use tempfile::tempdir;

    #[test]
    fn test_transaction_read_write() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        // Pre-populate
        db.put(b"key1", b"value1").unwrap();

        let mut txn = db.begin_transaction();

        // Read existing key
        assert_eq!(txn.get(b"key1").unwrap(), Some(Bytes::from("value1")));

        // Buffer a write
        txn.put(b"key2", b"value2").unwrap();

        // Read-your-writes
        assert_eq!(txn.get(b"key2").unwrap(), Some(Bytes::from("value2")));

        // key2 not visible outside transaction yet
        assert_eq!(db.get(b"key2").unwrap(), None);

        // Commit
        txn.commit().unwrap();

        // Now visible
        assert_eq!(db.get(b"key2").unwrap(), Some(Bytes::from("value2")));
    }

    #[test]
    fn test_transaction_delete() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key1", b"value1").unwrap();

        let mut txn = db.begin_transaction();
        txn.delete(b"key1").unwrap();

        // Deleted in transaction view
        assert_eq!(txn.get(b"key1").unwrap(), None);

        // Still visible outside
        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from("value1")));

        txn.commit().unwrap();

        // Now deleted
        assert_eq!(db.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_transaction_abort() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let mut txn = db.begin_transaction();
        txn.put(b"key1", b"value1").unwrap();
        txn.abort();

        // Not committed
        assert_eq!(db.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_transaction_conflict() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"balance", b"100").unwrap();

        // Start transaction and read balance
        let mut txn = db.begin_transaction();
        let _balance = txn.get(b"balance").unwrap();

        // Concurrent write (simulated)
        db.put(b"balance", b"50").unwrap();

        // Try to commit - should fail due to conflict
        txn.put(b"balance", b"200").unwrap();
        let result = txn.commit();

        assert!(result.is_err());
        match result {
            Err(DBError::TransactionConflict(c)) => {
                assert_eq!(c.conflicting_keys.len(), 1);
                assert_eq!(c.conflicting_keys[0], Bytes::from("balance"));
            }
            _ => panic!("Expected TransactionConflict"),
        }
    }

    #[test]
    fn test_transaction_no_conflict_on_unread_keys() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();

        // Start transaction, only read key1
        let mut txn = db.begin_transaction();
        let _v = txn.get(b"key1").unwrap();

        // Concurrent write to key2 (not read by txn)
        db.put(b"key2", b"new_value").unwrap();

        // Write to key1 (was read, but not modified externally)
        txn.put(b"key1", b"updated").unwrap();

        // Should succeed - key2 was not in read-set
        txn.commit().unwrap();

        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from("updated")));
    }

    #[test]
    fn test_transaction_empty_commit() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let txn = db.begin_transaction();
        // Empty transaction should succeed
        txn.commit().unwrap();
    }

    #[test]
    fn test_transaction_write_only_no_conflict() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        db.put(b"key1", b"value1").unwrap();

        // Start transaction, write without reading
        let mut txn = db.begin_transaction();
        txn.put(b"key1", b"new_value").unwrap();

        // Concurrent modification
        db.put(b"key1", b"concurrent").unwrap();

        // Should succeed - write-only doesn't check conflicts
        // (This is blind write, typical for OCC)
        txn.commit().unwrap();

        // Last writer wins
        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from("new_value")));
    }
}
