use crate::conversions::convert_error;
use crate::database::{VectorDatabase, VectorDatabaseInner};
use omendb_lib::vector::VectorStore;
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::Py;
use std::collections::HashMap;
use std::sync::Arc;

#[pymethods]
impl VectorDatabase {
    /// Create or get a named collection within this database.
    ///
    /// Collections are separate namespaces that share the same database path.
    /// Each collection has its own vectors and metadata, isolated from others.
    ///
    /// Args:
    ///     name (str): Collection name (alphanumeric and underscores only)
    ///
    /// Returns:
    ///     VectorDatabase: A new database instance for this collection
    ///
    /// Raises:
    ///     ValueError: If name is empty or contains invalid characters
    ///
    /// Examples:
    ///     >>> db = omendb.create("./mydb", {"dense": {"dim": 128}})
    ///     >>> users = db.collection("users")
    ///     >>> products = db.collection("products")
    ///     >>> users.set([{"id": "u1", "vector": [...]}])
    ///     >>> products.set([{"id": "p1", "vector": [...]}])
    ///
    ///     Separate namespaces:
    ///
    ///     >>> # IDs are scoped to collection
    ///     >>> users.set([{"id": "doc1", ...}])
    ///     >>> products.set([{"id": "doc1", ...}])  # No conflict!
    ///
    ///     Collection handles are cached - same name returns same object:
    ///
    ///     >>> col1 = db.collection("users")
    ///     >>> col2 = db.collection("users")
    ///     >>> col1 is col2  # True - same object
    #[pyo3(signature = (name, embedding_fn=None))]
    fn collection(
        &self,
        py: Python<'_>,
        name: String,
        embedding_fn: Option<Py<PyAny>>,
    ) -> PyResult<Py<VectorDatabase>> {
        // Validate collection name
        if name.is_empty() {
            return Err(PyValueError::new_err("Collection name cannot be empty"));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(PyValueError::new_err(
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        // Only persistent databases support collections
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage",
            ));
        }

        // Check cache first
        {
            let cache = self.collections_cache.read();
            if let Some(cached) = cache.get(&name) {
                return Ok(cached.clone_ref(py));
            }
        }

        // Not in cache - create new collection
        let mut cache = self.collections_cache.write();

        // Double-check after acquiring write lock
        if let Some(cached) = cache.get(&name) {
            return Ok(cached.clone_ref(py));
        }

        // Create collection path: {base_path}/collections/{name}
        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);

        // Ensure collections directory exists
        std::fs::create_dir_all(collection_path.parent().unwrap()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create collections directory: {}", e))
        })?;

        let (parent_dimensions, parent_token_dimension, parent_is_multi, parent_multi_config) = {
            let inner = self.inner.read();
            (
                inner.store()?.dimensions(),
                inner.store()?.token_dimension(),
                inner.store()?.is_multi_vector(),
                inner.store()?.multi_vector_config(),
            )
        };

        // Open or create the collection as a separate VectorStore while preserving modality.
        let current_dimensions = self.live_dimensions()?;

        let store = if collection_path.with_extension("omen").exists() {
            VectorStore::open(&collection_path).map_err(convert_error)?
        } else if parent_is_multi {
            let token_dim = parent_token_dimension
                .unwrap_or(parent_dimensions)
                .max(current_dimensions);
            let config = parent_multi_config.expect("multi-vector parent should expose config");
            VectorStore::multi_vector_with(token_dim, config)
                .map_err(convert_error)?
                .persist(&collection_path)
                .map_err(convert_error)?
        } else if current_dimensions == 0 {
            VectorStore::open(&collection_path).map_err(convert_error)?
        } else {
            VectorStore::open_with_dimensions(&collection_path, current_dimensions)
                .map_err(convert_error)?
        };
        let collection_db = VectorDatabase {
            inner: Arc::new(RwLock::new(VectorDatabaseInner {
                store: Some(store),
            })),
            path: collection_path.to_string_lossy().to_string(),
            is_persistent: true,
            embedding_fn: embedding_fn
                .or_else(|| self.embedding_fn.as_ref().map(|f| f.clone_ref(py))),
            collections_cache: RwLock::new(HashMap::new()),
        };

        // Cache and return
        let py_db = Py::new(py, collection_db)?;
        cache.insert(name, py_db.clone_ref(py));
        Ok(py_db)
    }

    /// List all collections in this database.
    ///
    /// Returns:
    ///     list[str]: Names of all collections
    fn collections(&self) -> PyResult<Vec<String>> {
        if !self.is_persistent {
            return Ok(Vec::new());
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");

        if !collections_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        let entries = std::fs::read_dir(&collections_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read collections: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to read entry: {}", e)))?;
            // Collections are stored as .omen files
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(collection_name) = name.strip_suffix(".omen") {
                        names.push(collection_name.to_string());
                    }
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Delete a collection from this database.
    ///
    /// Args:
    ///     name (str): Name of the collection to delete
    ///
    /// Raises:
    ///     ValueError: If collection doesn't exist
    ///     RuntimeError: If deletion fails
    ///
    /// Examples:
    ///     >>> db = omendb.create("./mydb", {"dense": {"dim": 128}})
    ///     >>> db.delete_collection("old_data")
    fn delete_collection(&self, name: String) -> PyResult<()> {
        if !self.is_persistent {
            return Err(PyValueError::new_err(
                "Collections require persistent storage",
            ));
        }

        if name.is_empty() {
            return Err(PyValueError::new_err("Collection name cannot be empty"));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(PyValueError::new_err(
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");
        let omen_path = collections_dir.join(format!("{}.omen", name));
        let wal_path = collections_dir.join(format!("{}.wal", name));

        if !omen_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Collection '{}' not found",
                name
            )));
        }

        // Remove files first, then cache (if files fail, cache stays consistent)
        std::fs::remove_file(&omen_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete collection: {}", e)))?;
        let _ = std::fs::remove_file(&wal_path);

        {
            let mut cache = self.collections_cache.write();
            cache.remove(&name);
        }

        Ok(())
    }
}
