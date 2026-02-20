use crate::conversions::convert_error;
use crate::database::VectorDatabase;
use omendb_lib::vector::VectorStoreOptions;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl VectorDatabase {
    /// Flush pending changes to disk.
    ///
    /// For hybrid search, this commits text index changes.
    /// Text search results are not visible until flush is called.
    ///
    /// Examples:
    ///     >>> db.set_with_text([...])
    ///     >>> db.flush()  # Text now searchable
    fn flush(&self) -> PyResult<()> {
        let mut inner = self.inner.write();
        inner.store.flush().map_err(convert_error)
    }

    /// Close the database and release file locks.
    ///
    /// Flushes pending changes to disk, then replaces the internal store
    /// with an empty in-memory database to release file handles.
    ///
    /// After calling close(), the database is no longer usable.
    /// Any subsequent operations will fail or return empty results.
    ///
    /// This is useful when you need to reopen the same database path
    /// in the same process without relying on garbage collection.
    ///
    /// Note:
    ///     For most use cases, prefer the context manager (`with` statement)
    ///     which automatically flushes on exit:
    ///
    ///         with omendb.open("./db", dimensions=128) as db:
    ///             db.set([...])
    ///         # Flushed automatically
    ///
    /// Examples:
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    ///     >>> db.set([{"id": "1", "vector": [0.1] * 128}])
    ///     >>> db.close()  # Release file locks
    ///     >>> # Can now reopen the same path
    ///     >>> db = omendb.open("./mydb", dimensions=128)
    fn close(&self) -> PyResult<()> {
        let mut inner = self.inner.write();
        // Flush first to ensure all data is persisted
        inner.store.flush().map_err(convert_error)?;
        // Replace with minimal in-memory store to release file lock
        let dummy_store = VectorStoreOptions::default()
            .dimensions(self.dimensions)
            .build()
            .map_err(convert_error)?;
        inner.store = dummy_store;
        Ok(())
    }

    /// Compact the database by removing deleted records and reclaiming space.
    ///
    /// This operation removes tombstoned records, reassigns indices to be
    /// contiguous, and rebuilds the search index. Call after bulk deletes
    /// to reclaim memory and improve search performance.
    ///
    /// Note: flush() automatically compacts when more than 25% of slots are
    /// tombstones, so explicit compact() calls are optional for most workloads.
    ///
    /// Returns:
    ///     int: Number of deleted records that were removed
    ///
    /// Examples:
    ///     After bulk delete:
    ///
    ///     >>> db.delete(stale_ids)
    ///     >>> removed = db.compact()
    ///     >>> print(f"Removed {removed} deleted records")
    ///     >>> db.flush()  # Persist compacted state
    ///
    ///     Periodic maintenance:
    ///
    ///     >>> removed = db.compact()
    ///
    /// Performance:
    ///     Compaction rebuilds the HNSW index, which is O(n log n).
    ///     Call periodically after bulk deletes, not after every delete.
    fn compact(&self) -> PyResult<usize> {
        let mut inner = self.inner.write();
        inner.store.compact().map_err(convert_error)
    }

    /// Merge vectors from another database into this one.
    ///
    /// Args:
    ///     other (VectorDatabase): Source database to merge from
    ///     key_prefix (str, optional): Prefix to prepend to all IDs from the source.
    ///         Useful when merging subdirectory indexes into a parent.
    ///         Example: key_prefix="subdir/" turns "foo.py" into "subdir/foo.py"
    ///
    /// Returns:
    ///     int: Number of vectors merged
    ///
    /// Note:
    ///     - Conflicting IDs are skipped (existing wins)
    ///     - Source database is not modified
    ///     - Both databases must have the same dimensions
    ///
    /// Examples:
    ///     >>> db.merge_from(other_db)
    ///     >>> db.merge_from(other_db, key_prefix="subdir/")
    #[pyo3(signature = (other, key_prefix=None))]
    fn merge_from(&self, other: &VectorDatabase, key_prefix: Option<&str>) -> PyResult<usize> {
        use std::sync::Arc;

        if Arc::ptr_eq(&self.inner, &other.inner) {
            return Err(PyValueError::new_err("cannot merge a database into itself"));
        }

        // Acquire locks in pointer-address order to prevent AB/BA deadlock
        // when a.merge_from(b) and b.merge_from(a) run concurrently.
        let self_addr = Arc::as_ptr(&self.inner) as usize;
        let other_addr = Arc::as_ptr(&other.inner) as usize;

        if self_addr < other_addr {
            let mut inner = self.inner.write();
            let other_inner = other.inner.read();
            inner
                .store
                .merge_from_with_prefix(&other_inner.store, key_prefix)
                .map_err(convert_error)
        } else {
            let other_inner = other.inner.read();
            let mut inner = self.inner.write();
            inner
                .store
                .merge_from_with_prefix(&other_inner.store, key_prefix)
                .map_err(convert_error)
        }
    }

    /// Optimize index for cache-efficient search.
    ///
    /// Reorders graph nodes and vectors using BFS traversal to improve memory locality.
    /// Nodes frequently accessed together during search will be stored adjacently,
    /// reducing cache misses and improving QPS by 6-40%.
    ///
    /// Call this after loading data and before querying for best results.
    ///
    /// Returns:
    ///     int: Number of nodes reordered (0 if index empty/not initialized)
    ///
    /// Examples:
    ///     >>> db.set([...])  # Load data
    ///     >>> db.optimize()  # Optimize for search
    ///     >>> db.search(...)  # Faster queries
    fn optimize(&mut self) -> PyResult<usize> {
        let mut inner = self.inner.write();
        inner.store.optimize().map_err(convert_error)
    }
}
