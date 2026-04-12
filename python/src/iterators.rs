use crate::conversions::json_to_pyobject;
use crate::database::VectorDatabaseInner;
use parking_lot::RwLock;
use pyo3::conversion::IntoPyObject;
use pyo3::prelude::*;
use pyo3::Py;
use std::collections::HashMap;
use std::sync::Arc;

/// Lazy iterator for VectorDatabase IDs.
///
/// Memory efficient: iterates over IDs one at a time from a snapshot.
/// Handles deletions during iteration gracefully (skips deleted IDs).
#[pyclass]
pub struct VectorDatabaseIdIterator {
    /// Reference to the database inner state
    pub(crate) inner: Arc<RwLock<VectorDatabaseInner>>,
    /// IDs to iterate over
    pub(crate) ids: Vec<String>,
    /// Current position
    pub(crate) index: usize,
}

#[pymethods]
impl VectorDatabaseIdIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<String>> {
        // Loop to skip items deleted during iteration
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            // Check if ID still exists
            let inner = self.inner.read();
            if inner.store()?.contains(id) {
                return Ok(Some(id.clone()));
            }
            // Item was deleted during iteration, continue to next
        }
        Ok(None)
    }
}

/// Lazy iterator for VectorDatabase items.
///
/// Enables `for item in db:` syntax with true lazy evaluation.
/// Memory efficient: stores only IDs (~20MB for 1M items), fetches vectors one at a time.
/// Handles items deleted during iteration gracefully (skips them).
#[pyclass]
pub struct VectorDatabaseIterator {
    /// Reference to the database inner state
    pub(crate) inner: Arc<RwLock<VectorDatabaseInner>>,
    /// IDs to iterate over (lightweight - just strings)
    pub(crate) ids: Vec<String>,
    /// Current position
    pub(crate) index: usize,
}

#[pymethods]
impl VectorDatabaseIterator {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<HashMap<String, Py<PyAny>>>> {
        // Loop to skip items deleted during iteration
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            // Fetch item lazily - only loads one vector at a time
            let inner = self.inner.read();
            if let Some((vec, meta)) = inner.store()?.get(id) {
                let mut result = HashMap::new();
                result.insert(
                    "id".to_string(),
                    id.clone().into_pyobject(py).unwrap().unbind().into(),
                );
                result.insert(
                    "vector".to_string(),
                    vec.data.clone().into_pyobject(py).unwrap().unbind(),
                );
                result.insert("metadata".to_string(), json_to_pyobject(py, &meta)?);
                return Ok(Some(result));
            }
            // Item was deleted during iteration, continue to next
        }
        Ok(None)
    }
}
