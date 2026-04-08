use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::VectorStore;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::database::{EmbeddingFn, VectorDatabase, VectorDatabaseInner};

#[napi]
impl VectorDatabase {
    /// Get or create a named collection.
    ///
    /// Collection handles share state - changes made through one handle
    /// are immediately visible through another (no flush required).
    #[napi]
    pub fn collection(
        &self,
        name: String,
        #[napi(ts_arg_type = "((texts: string[]) => Float32Array[]) | undefined")]
        embedding_fn: Option<EmbeddingFn>,
    ) -> Result<VectorDatabase> {
        if name.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name cannot be empty",
            ));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        if !self.is_persistent {
            return Err(Error::new(
                Status::InvalidArg,
                "Collections require persistent storage",
            ));
        }

        let col_embedding_fn = embedding_fn
            .map(Arc::new)
            .or_else(|| self.embedding_fn.clone());
        let current_dimensions = self.live_dimensions();

        {
            let cache = self.collections_cache.read();
            if let Some(cached_inner) = cache.get(&name) {
                let base_path = std::path::Path::new(&self.path);
                let collection_path = base_path.join("collections").join(&name);
                return Ok(VectorDatabase {
                    inner: Arc::clone(cached_inner),
                    path: collection_path.to_string_lossy().to_string(),
                    is_persistent: true,
                    embedding_fn: col_embedding_fn.clone(),
                    collections_cache: RwLock::new(HashMap::new()),
                });
            }
        }

        let mut cache = self.collections_cache.write();

        if let Some(cached_inner) = cache.get(&name) {
            let base_path = std::path::Path::new(&self.path);
            let collection_path = base_path.join("collections").join(&name);
            return Ok(VectorDatabase {
                inner: Arc::clone(cached_inner),
                path: collection_path.to_string_lossy().to_string(),
                is_persistent: true,
                embedding_fn: col_embedding_fn.clone(),
                collections_cache: RwLock::new(HashMap::new()),
            });
        }

        let base_path = std::path::Path::new(&self.path);
        let collection_path = base_path.join("collections").join(&name);
        let collection_omen_path = collection_path.with_extension("omen");

        std::fs::create_dir_all(collection_path.parent().unwrap()).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to create collections directory: {}", e),
            )
        })?;

        let (parent_dimensions, parent_token_dimension, parent_is_multi, parent_multi_config) = {
            let inner = self.inner.read();
            (
                inner.store.dimensions(),
                inner.store.token_dimension(),
                inner.store.is_multi_vector(),
                inner.store.multi_vector_config(),
            )
        };

        let store = if collection_omen_path.exists() {
            VectorStore::open(&collection_path).map_err(crate::conversions::convert_error)?
        } else if parent_is_multi {
            let token_dim = parent_token_dimension
                .unwrap_or(parent_dimensions)
                .max(current_dimensions as usize);
            let config =
                parent_multi_config.expect("multi-vector parent should expose multi-vector config");
            VectorStore::multi_vector_with(token_dim, config)
                .map_err(crate::conversions::convert_error)?
                .persist(&collection_path)
                .map_err(crate::conversions::convert_error)?
        } else if current_dimensions == 0 {
            VectorStore::open(&collection_path).map_err(crate::conversions::convert_error)?
        } else {
            VectorStore::open_with_dimensions(&collection_path, current_dimensions as usize)
                .map_err(crate::conversions::convert_error)?
        };
        let inner = Arc::new(RwLock::new(VectorDatabaseInner { store }));

        cache.insert(name, Arc::clone(&inner));

        Ok(VectorDatabase {
            inner,
            path: collection_path.to_string_lossy().to_string(),
            is_persistent: true,
            embedding_fn: col_embedding_fn,
            collections_cache: RwLock::new(HashMap::new()),
        })
    }

    /// List all collections.
    #[napi]
    pub fn collections(&self) -> Result<Vec<String>> {
        if !self.is_persistent {
            return Ok(Vec::new());
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");

        if !collections_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        let entries = std::fs::read_dir(&collections_dir).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to read collections: {}", e),
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                Error::new(
                    Status::GenericFailure,
                    format!("Failed to read entry: {}", e),
                )
            })?;
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

    /// Delete a collection.
    #[napi]
    pub fn delete_collection(&self, name: String) -> Result<()> {
        if name.is_empty() {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name cannot be empty",
            ));
        }
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(Error::new(
                Status::InvalidArg,
                "Collection name must contain only alphanumeric characters and underscores",
            ));
        }

        if !self.is_persistent {
            return Err(Error::new(
                Status::InvalidArg,
                "Collections require persistent storage",
            ));
        }

        let base_path = std::path::Path::new(&self.path);
        let collections_dir = base_path.join("collections");
        let omen_path = collections_dir.join(format!("{}.omen", name));
        let wal_path = collections_dir.join(format!("{}.wal", name));

        if !omen_path.exists() {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Collection '{}' does not exist", name),
            ));
        }

        std::fs::remove_file(&omen_path).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to delete collection: {}", e),
            )
        })?;

        let _ = std::fs::remove_file(&wal_path);

        {
            let mut cache = self.collections_cache.write();
            cache.remove(&name);
        }

        Ok(())
    }
}
