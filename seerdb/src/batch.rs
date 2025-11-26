// Batch write operations for atomic multi-key updates
//
// Allows collecting multiple put/delete operations and committing them atomically
// to the WAL and memtable. This is more efficient than individual operations as it:
// - Writes to WAL once (instead of N times)
// - Reduces channel overhead
// - Better cache locality

use bytes::Bytes;
use std::sync::atomic::Ordering;

use crate::db::{DBError, Result, DB};
use crate::wal::{BatchOp, Record};

/// Operation type in a batch
#[derive(Clone, Debug)]
enum Operation {
    /// Insert or update a key-value pair
    Put { key: Bytes, value: Bytes },
    /// Delete a key
    Delete { key: Bytes },
}

/// Atomic write batch
///
/// Collects multiple write operations (puts and deletes) and commits them atomically.
/// All operations in a batch succeed or fail together, providing transactional semantics.
///
/// # Examples
///
/// ```rust,no_run
/// use seerdb::{DB, DBOptions};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let db = DB::open(DBOptions::default())?;
///
/// // Create a batch
/// let mut batch = db.batch();
///
/// // Add operations
/// batch.put(b"user:1:name", b"Alice");
/// batch.put(b"user:1:email", b"alice@example.com");
/// batch.delete(b"user:1:temp");
///
/// // Commit atomically
/// batch.commit()?;
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// Batching is significantly faster than individual operations because:
/// - Single WAL write instead of multiple
/// - Reduced thread synchronization overhead
/// - Better CPU cache locality
///
/// Typical improvement: 2-5x faster for batches of 100+ operations
pub struct Batch<'db> {
    /// Reference to parent database
    db: &'db DB,
    /// Collected operations
    operations: Vec<Operation>,
}

impl<'db> Batch<'db> {
    /// Create a new batch for the given database
    pub(crate) fn new(db: &'db DB) -> Self {
        Self {
            db,
            operations: Vec::new(),
        }
    }

    /// Create a new batch with preallocated capacity
    ///
    /// Use this when you know the approximate number of operations
    /// to avoid reallocations.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use seerdb::{DB, DBOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = DB::open(DBOptions::default())?;
    /// // Preallocate for 1000 operations
    /// let mut batch = db.batch_with_capacity(1000);
    ///
    /// for i in 0..1000 {
    ///     batch.put(format!("key_{}", i).as_bytes(), b"value");
    /// }
    /// batch.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_capacity(db: &'db DB, capacity: usize) -> Self {
        Self {
            db,
            operations: Vec::with_capacity(capacity),
        }
    }

    /// Add a put operation to the batch
    ///
    /// The operation is not written to disk until `commit()` is called.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use seerdb::{DB, DBOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = DB::open(DBOptions::default())?;
    /// let mut batch = db.batch();
    /// batch.put(b"key1", b"value1");
    /// batch.put(b"key2", b"value2");
    /// batch.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn put(&mut self, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) {
        let key = Bytes::copy_from_slice(key.as_ref());
        let value = Bytes::copy_from_slice(value.as_ref());
        self.operations.push(Operation::Put { key, value });
    }

    /// Add a delete operation to the batch
    ///
    /// The operation is not written to disk until `commit()` is called.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use seerdb::{DB, DBOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = DB::open(DBOptions::default())?;
    /// let mut batch = db.batch();
    /// batch.delete(b"old_key");
    /// batch.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&mut self, key: impl AsRef<[u8]>) {
        let key = Bytes::copy_from_slice(key.as_ref());
        self.operations.push(Operation::Delete { key });
    }

    /// Get the number of operations in the batch
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Clear all operations from the batch
    ///
    /// Useful if you want to reuse the batch without deallocating.
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    /// Commit all operations in the batch atomically
    ///
    /// Writes all operations to the WAL and memtable. If any operation fails,
    /// none of the operations take effect (atomic semantics).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - WAL write fails
    /// - Memtable flush is triggered and fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use seerdb::{DB, DBOptions};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db = DB::open(DBOptions::default())?;
    /// let mut batch = db.batch();
    /// batch.put(b"key1", b"value1");
    /// batch.put(b"key2", b"value2");
    /// batch.commit()?;  // Atomic: both succeed or both fail
    /// # Ok(())
    /// # }
    /// ```
    pub fn commit(self) -> Result<()> {
        if self.operations.is_empty() {
            return Ok(());
        }

        // Reserve sequence numbers for all operations in the batch
        let op_count = self.operations.len() as u64;
        let base_seq = self.db.next_seq.fetch_add(op_count, Ordering::SeqCst);

        // Convert internal operations to WAL BatchOp format
        let wal_ops: Vec<BatchOp> = self
            .operations
            .iter()
            .map(|op| match op {
                Operation::Put { key, value } => BatchOp::Put {
                    key: key.clone(),
                    value: value.clone(),
                },
                Operation::Delete { key } => BatchOp::Delete { key: key.clone() },
            })
            .collect();

        // Write single atomic batch record to WAL (durability)
        // This ensures atomicity: either ALL operations are written or NONE
        let batch_record = Record::Batch {
            base_seq,
            operations: wal_ops,
        };

        if self.db.options.skip_wal {
            // Skip WAL entirely: maximum write speed, no durability until flush
            // WARNING: Data loss on crash before flush
            self.db.apply_wal_records(&[batch_record]);
        } else {
            // Pipelined Group Commit (WAL + Memtable)
            self.db
                .pipelined_wal
                .put(batch_record, |records| {
                    self.db.apply_wal_records(records);
                })
                .map_err(DBError::Wal)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DBOptions, DB};
    use tempfile::tempdir;

    #[test]
    fn test_batch_basic() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let mut batch = db.batch();
        batch.put(b"key1", b"value1");
        batch.put(b"key2", b"value2");
        batch.delete(b"key3");

        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());

        batch.commit().unwrap();

        assert_eq!(db.get(b"key1").unwrap(), Some(Bytes::from("value1")));
        assert_eq!(db.get(b"key2").unwrap(), Some(Bytes::from("value2")));
        assert_eq!(db.get(b"key3").unwrap(), None);
    }

    #[test]
    fn test_batch_empty() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let batch = db.batch();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        // Committing empty batch should succeed
        batch.commit().unwrap();
    }

    #[test]
    fn test_batch_with_capacity() {
        let dir = tempdir().unwrap();
        let opts = DBOptions {
            data_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let db = DB::open(opts).unwrap();

        let mut batch = db.batch_with_capacity(100);
        for i in 0..100 {
            batch.put(format!("key_{}", i).as_bytes(), b"value");
        }

        assert_eq!(batch.len(), 100);
        batch.commit().unwrap();

        // Verify all keys exist
        for i in 0..100 {
            let key = format!("key_{}", i);
            assert_eq!(db.get(key.as_bytes()).unwrap(), Some(Bytes::from("value")));
        }
    }
}
