//! C FFI bindings for `OmenDB`
//!
//! Provides a C-compatible API for embedding `OmenDB` in other languages.
//!
//! # Safety
//!
//! All functions that take raw pointers are marked `unsafe` because:
//! - The caller must ensure pointer validity
//! - The caller must ensure proper memory management
//!
//! # Example (C)
//! ```c
//! #include "omendb.h"
//!
//! omendb_db_t* db = omendb_open("./vectors", 384, NULL);
//! if (!db) {
//!     printf("Error: %s\n", omendb_last_error());
//!     return 1;
//! }
//!
//! // Insert vectors
//! const char* items = "[{\"id\":\"doc1\",\"vector\":[0.1,...],\"metadata\":{}}]";
//! omendb_set(db, items);
//!
//! // Search
//! float query[384] = {0.1, ...};
//! char* results = NULL;
//! omendb_search(db, query, 384, 10, NULL, &results);
//! printf("Results: %s\n", results);
//! omendb_free_string(results);
//!
//! omendb_close(db);
//! ```

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use omendb::vector::{MetadataFilter, Vector, VectorStore, VectorStoreOptions};
use serde_json::{json, Value as JsonValue};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(err).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Opaque database handle
pub struct OmenDB {
    store: VectorStore,
    dimensions: usize,
}

/// Open a database at the given path
///
/// # Arguments
/// * `path` - Path to database directory (UTF-8)
/// * `dimensions` - Vector dimensionality
/// * `config_json` - Optional JSON config string (NULL for defaults)
///
/// Config JSON format:
/// ```json
/// {
///   "m": 16,                    // Number of neighbors per node (default: 16)
///   "ef_construction": 100,     // Build quality (default: 100)
///   "ef_search": 100            // Search quality (default: 100)
/// }
/// ```
///
/// # Returns
/// Database handle on success, NULL on failure (check `omendb_last_error`)
///
/// # Safety
/// - `path` must be a valid, null-terminated UTF-8 string
/// - `config_json` must be NULL or a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_open(
    path: *const c_char,
    dimensions: usize,
    config_json: *const c_char,
) -> *mut OmenDB {
    clear_last_error();

    if path.is_null() {
        set_last_error("Null path pointer".to_string());
        return ptr::null_mut();
    }

    let path = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid path: {e}"));
            return ptr::null_mut();
        }
    };

    // Parse config if provided
    let config: Option<JsonValue> = if config_json.is_null() {
        None
    } else {
        let config_str = match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid config string: {e}"));
                return ptr::null_mut();
            }
        };
        match serde_json::from_str(config_str) {
            Ok(v) => Some(v),
            Err(e) => {
                set_last_error(format!("Invalid config JSON: {e}"));
                return ptr::null_mut();
            }
        }
    };

    // Reject multi-vector config (not supported in FFI)
    if let Some(ref cfg) = config {
        if cfg.get("multi_vector").is_some() {
            set_last_error(
                "Multi-vector stores are not supported in the C FFI. Use the Python or Node.js bindings."
                    .to_string(),
            );
            return ptr::null_mut();
        }
    }

    // Build store with optional config
    let result = if let Some(cfg) = config {
        let mut options = VectorStoreOptions::new().dimensions(dimensions);

        if let Some(m) = cfg.get("m").and_then(JsonValue::as_u64) {
            options = options.m(m as usize);
        }
        if let Some(ef_c) = cfg.get("ef_construction").and_then(JsonValue::as_u64) {
            options = options.ef_construction(ef_c as usize);
        }
        if let Some(ef_s) = cfg.get("ef_search").and_then(JsonValue::as_u64) {
            options = options.ef_search(ef_s as usize);
        }

        options.open(Path::new(path))
    } else {
        VectorStore::open_with_dimensions(Path::new(path), dimensions)
    };

    match result {
        Ok(store) => {
            if store.is_multi_vector() {
                set_last_error(
                    "Cannot open multi-vector store via C FFI. Use the Python or Node.js bindings."
                        .to_string(),
                );
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(OmenDB { store, dimensions }))
        }
        Err(e) => {
            set_last_error(format!("Failed to open database: {e}"));
            ptr::null_mut()
        }
    }
}

/// Close database and free resources
///
/// # Safety
/// - `db` must be NULL or a valid pointer returned by `omendb_open`
/// - After calling this function, `db` is invalid and must not be used
#[no_mangle]
pub unsafe extern "C" fn omendb_close(db: *mut OmenDB) {
    if !db.is_null() {
        drop(Box::from_raw(db));
    }
}

/// Insert or replace vectors
///
/// # Arguments
/// * `db` - Database handle
/// * `items_json` - JSON array: `[{"id": "...", "vector": [...], "metadata": {...}}, ...]`
///
/// # Returns
/// Number of vectors inserted, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `items_json` must be a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_set(db: *mut OmenDB, items_json: *const c_char) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if items_json.is_null() {
        set_last_error("Null items_json pointer".to_string());
        return -1;
    }

    let items_str = match CStr::from_ptr(items_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {e}"));
            return -1;
        }
    };

    let items: Vec<JsonValue> = match serde_json::from_str(items_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {e}"));
            return -1;
        }
    };

    let mut count = 0i64;
    for item in items {
        let id = if let Some(s) = item.get("id").and_then(|v| v.as_str()) {
            s
        } else {
            set_last_error("Item missing 'id' field".to_string());
            return -1;
        };

        let vector_data: Vec<f32> = if let Some(arr) = item.get("vector").and_then(|v| v.as_array())
        {
            let mut data = Vec::with_capacity(arr.len());
            for (j, v) in arr.iter().enumerate() {
                match v.as_f64() {
                    Some(f) => data.push(f as f32),
                    None => {
                        set_last_error(format!(
                            "Vector element at index {} is not a number: {}",
                            j, v
                        ));
                        return -1;
                    }
                }
            }
            data
        } else {
            set_last_error("Item missing 'vector' field".to_string());
            return -1;
        };

        let metadata = item.get("metadata").cloned().unwrap_or(json!({}));

        let vector = Vector::new(vector_data);
        if let Err(e) = db.store.set(id, vector, metadata) {
            set_last_error(format!("Set failed after {count} items: {e}"));
            return -1;
        }
        count += 1;
    }

    count
}

/// Get vectors by ID
///
/// # Arguments
/// * `db` - Database handle
/// * `ids_json` - JSON array of IDs: `["id1", "id2", ...]`
/// * `result` - Output pointer for result JSON (caller must free with `omendb_free_string`)
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `ids_json` must be a valid, null-terminated UTF-8 string
/// - `result` must be a valid pointer to a `*mut c_char`
#[no_mangle]
pub unsafe extern "C" fn omendb_get(
    db: *mut OmenDB,
    ids_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let Some(db) = db.as_ref() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if ids_json.is_null() {
        set_last_error("Null ids_json pointer".to_string());
        return -1;
    }

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

    let ids_str = match CStr::from_ptr(ids_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {e}"));
            return -1;
        }
    };

    let ids: Vec<String> = match serde_json::from_str(ids_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {e}"));
            return -1;
        }
    };

    let mut results = Vec::new();
    for id in ids {
        if let Some((vector, metadata)) = db.store.get(&id) {
            results.push(json!({
                "id": id,
                "vector": vector.data,
                "metadata": metadata
            }));
        }
    }

    let json_str = match serde_json::to_string(&results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            *result = cstr.into_raw();
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {e}"));
            -1
        }
    }
}

/// Delete vectors by ID
///
/// # Returns
/// Number of vectors deleted, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `ids_json` must be a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_delete(db: *mut OmenDB, ids_json: *const c_char) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if ids_json.is_null() {
        set_last_error("Null ids_json pointer".to_string());
        return -1;
    }

    let ids_str = match CStr::from_ptr(ids_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {e}"));
            return -1;
        }
    };

    let ids: Vec<String> = match serde_json::from_str(ids_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {e}"));
            return -1;
        }
    };

    match db.store.delete_batch(&ids) {
        Ok(count) => i64::try_from(count).unwrap_or(i64::MAX),
        Err(e) => {
            set_last_error(format!("Delete failed: {e}"));
            -1
        }
    }
}

/// Search for similar vectors
///
/// # Arguments
/// * `db` - Database handle
/// * `query` - Query vector (float array)
/// * `query_len` - Length of query vector
/// * `k` - Number of results to return
/// * `filter_json` - Optional filter JSON (NULL for no filter)
/// * `result` - Output pointer for result JSON (caller must free with `omendb_free_string`)
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `query` must point to at least `query_len` valid f32 values
/// - `filter_json` must be NULL or a valid, null-terminated UTF-8 string
/// - `result` must be a valid pointer to a `*mut c_char`
#[no_mangle]
pub unsafe extern "C" fn omendb_search(
    db: *mut OmenDB,
    query: *const f32,
    query_len: usize,
    k: usize,
    filter_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if query.is_null() {
        set_last_error("Null query pointer".to_string());
        return -1;
    }

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

    if query_len != db.dimensions {
        set_last_error(format!(
            "Query dimension mismatch: expected {}, got {query_len}",
            db.dimensions
        ));
        return -1;
    }

    let query_vec: Vec<f32> = std::slice::from_raw_parts(query, query_len).to_vec();
    let query = Vector::new(query_vec);

    // Parse filter if provided
    let filter: Option<MetadataFilter> = if filter_json.is_null() {
        None
    } else {
        let filter_str = match CStr::from_ptr(filter_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid filter string: {e}"));
                return -1;
            }
        };
        match serde_json::from_str::<JsonValue>(filter_str) {
            Ok(v) => match MetadataFilter::from_json(&v) {
                Ok(f) => Some(f),
                Err(e) => {
                    set_last_error(format!("Invalid filter format: {e}"));
                    return -1;
                }
            },
            Err(e) => {
                set_last_error(format!("Invalid filter JSON: {e}"));
                return -1;
            }
        }
    };

    // Search using the store's search method (returns index, distance, metadata)
    let results = match db.store.search(&query, k, filter.as_ref()) {
        Ok(r) => r,
        Err(e) => {
            set_last_error(format!("Search failed: {e}"));
            return -1;
        }
    };

    // Convert results to JSON, mapping internal indices to string IDs
    let json_results: Vec<JsonValue> = results
        .into_iter()
        .map(|result| {
            json!({
                "id": result.id,
                "distance": result.distance,
                "metadata": result.metadata
            })
        })
        .collect();

    let json_str = match serde_json::to_string(&json_results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            *result = cstr.into_raw();
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {e}"));
            -1
        }
    }
}

/// Get number of vectors in database
///
/// # Safety
/// - `db` must be NULL or a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_count(db: *const OmenDB) -> i64 {
    clear_last_error();
    match db.as_ref() {
        Some(db) => i64::try_from(db.store.len()).unwrap_or(i64::MAX),
        None => {
            set_last_error("Null database handle".to_string());
            -1
        }
    }
}

/// Save database to disk
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_save(db: *mut OmenDB) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.flush() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(format!("Save failed: {e}"));
            -1
        }
    }
}

/// Get last error message
///
/// # Returns
/// Error message string (valid until next FFI call), or NULL if no error
#[no_mangle]
pub extern "C" fn omendb_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

/// Free a string returned by `OmenDB`
///
/// # Safety
/// - `s` must be NULL or a valid pointer returned by an `OmenDB` function
/// - After calling this function, `s` is invalid and must not be used
#[no_mangle]
pub unsafe extern "C" fn omendb_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Get `OmenDB` version
#[no_mangle]
pub extern "C" fn omendb_version() -> *const c_char {
    // Use compile-time version from Cargo.toml with null terminator
    concat!(env!("CARGO_PKG_VERSION"), "\0")
        .as_ptr()
        .cast::<c_char>()
}

/// Enable text search for hybrid search
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_enable_text_search(db: *mut OmenDB) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.enable_text_search() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(format!("Failed to enable text search: {e}"));
            -1
        }
    }
}

/// Check if text search is enabled
///
/// # Returns
/// 1 if enabled, 0 if not, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_has_text_search(db: *const OmenDB) -> i32 {
    clear_last_error();
    let Some(db) = db.as_ref() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };
    i32::from(db.store.has_text_search())
}

/// Set vectors with text for hybrid search
///
/// # Arguments
/// * `db` - Database handle
/// * `items_json` - JSON array: `[{"id": "...", "vector": [...], "text": "...", "metadata": {...}}, ...]`
///
/// # Returns
/// Number of vectors inserted, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `items_json` must be a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_set_with_text(db: *mut OmenDB, items_json: *const c_char) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if !db.store.has_text_search() {
        set_last_error(
            "Text search not enabled. Call omendb_enable_text_search first.".to_string(),
        );
        return -1;
    }

    if items_json.is_null() {
        set_last_error("Null items_json pointer".to_string());
        return -1;
    }

    let items_str = match CStr::from_ptr(items_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {e}"));
            return -1;
        }
    };

    let items: Vec<JsonValue> = match serde_json::from_str(items_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {e}"));
            return -1;
        }
    };

    let mut count = 0i64;
    for item in items {
        let id = if let Some(s) = item.get("id").and_then(|v| v.as_str()) {
            s
        } else {
            set_last_error("Item missing 'id' field".to_string());
            return -1;
        };

        let vector_data: Vec<f32> = if let Some(arr) = item.get("vector").and_then(|v| v.as_array())
        {
            let mut data = Vec::with_capacity(arr.len());
            for (j, v) in arr.iter().enumerate() {
                match v.as_f64() {
                    Some(f) => data.push(f as f32),
                    None => {
                        set_last_error(format!(
                            "Vector element at index {} is not a number: {}",
                            j, v
                        ));
                        return -1;
                    }
                }
            }
            data
        } else {
            set_last_error("Item missing 'vector' field".to_string());
            return -1;
        };

        let text = if let Some(s) = item.get("text").and_then(|v| v.as_str()) {
            s
        } else {
            set_last_error("Item missing 'text' field".to_string());
            return -1;
        };

        let metadata = item.get("metadata").cloned().unwrap_or(json!({}));

        let vector = Vector::new(vector_data);
        if let Err(e) = db.store.set_with_text(id, vector, text, metadata) {
            set_last_error(format!("Set with text failed after {count} items: {e}"));
            return -1;
        }
        count += 1;
    }

    count
}

/// Text-only search (BM25)
///
/// # Arguments
/// * `db` - Database handle
/// * `query` - Text query string
/// * `k` - Number of results
/// * `result` - Output pointer for result JSON
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - All pointer arguments must be valid
#[no_mangle]
pub unsafe extern "C" fn omendb_text_search(
    db: *mut OmenDB,
    query: *const c_char,
    k: usize,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let Some(db) = db.as_ref() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if query.is_null() {
        set_last_error("Null query pointer".to_string());
        return -1;
    }

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid query string: {e}"));
            return -1;
        }
    };

    let search_results = match db.store.search_text(query_str, k) {
        Ok(r) => r,
        Err(e) => {
            set_last_error(format!("Text search failed: {e}"));
            return -1;
        }
    };

    let json_results: Vec<JsonValue> = search_results
        .into_iter()
        .map(|(id, score)| {
            let metadata = db
                .store
                .get(&id)
                .map(|(_, meta)| meta)
                .unwrap_or(serde_json::json!({}));
            json!({"id": id, "score": score, "metadata": metadata})
        })
        .collect();

    let json_str = match serde_json::to_string(&json_results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            *result = cstr.into_raw();
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {e}"));
            -1
        }
    }
}

/// Hybrid search combining vector and text
///
/// # Arguments
/// * `db` - Database handle
/// * `query_vector` - Query vector (float array)
/// * `query_len` - Length of query vector
/// * `query_text` - Text query string
/// * `k` - Number of results
/// * `alpha` - Weight for vector vs text (0.0=text only, 1.0=vector only, <0 for default 0.5)
/// * `rrf_k` - RRF constant (0 for default 60)
/// * `filter_json` - Optional filter JSON string (NULL for no filter)
/// * `result` - Output pointer for result JSON
///
/// # Returns
/// 0 on success, -1 on error
///
/// Result JSON format: `[{"id": "...", "score": 0.5, "metadata": {...}}, ...]`
///
/// # Safety
/// - All pointer arguments must be valid (except filter_json which can be NULL)
#[no_mangle]
pub unsafe extern "C" fn omendb_hybrid_search(
    db: *mut OmenDB,
    query_vector: *const f32,
    query_len: usize,
    query_text: *const c_char,
    k: usize,
    alpha: f32,
    rrf_k: usize,
    filter_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if query_vector.is_null() {
        set_last_error("Null query_vector pointer".to_string());
        return -1;
    }

    if query_text.is_null() {
        set_last_error("Null query_text pointer".to_string());
        return -1;
    }

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

    if query_len != db.dimensions {
        set_last_error(format!(
            "Query dimension mismatch: expected {}, got {query_len}",
            db.dimensions
        ));
        return -1;
    }

    let query_vec: Vec<f32> = std::slice::from_raw_parts(query_vector, query_len).to_vec();
    let vector = Vector::new(query_vec);

    let text_str = match CStr::from_ptr(query_text).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid text query: {e}"));
            return -1;
        }
    };

    // Use None for default (0.5), otherwise use provided alpha
    let alpha_opt = if alpha < 0.0 { None } else { Some(alpha) };
    let rrf_k_opt = if rrf_k == 0 { None } else { Some(rrf_k) };

    // Parse optional filter
    let filter = if filter_json.is_null() {
        None
    } else {
        let filter_str = match CStr::from_ptr(filter_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid filter string: {e}"));
                return -1;
            }
        };
        match serde_json::from_str::<JsonValue>(filter_str) {
            Ok(v) => match MetadataFilter::from_json(&v) {
                Ok(f) => Some(f),
                Err(e) => {
                    set_last_error(format!("Invalid filter format: {e}"));
                    return -1;
                }
            },
            Err(e) => {
                set_last_error(format!("Invalid filter JSON: {e}"));
                return -1;
            }
        }
    };

    let search_results =
        match db
            .store
            .search_hybrid(&vector, text_str, k, filter.as_ref(), alpha_opt, rrf_k_opt)
        {
            Ok(r) => r,
            Err(e) => {
                set_last_error(format!("Hybrid search failed: {e}"));
                return -1;
            }
        };

    let json_results: Vec<JsonValue> = search_results
        .into_iter()
        .map(|(id, score, metadata)| json!({"id": id, "score": score, "metadata": metadata}))
        .collect();

    let json_str = match serde_json::to_string(&json_results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            *result = cstr.into_raw();
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {e}"));
            -1
        }
    }
}

/// Flush pending changes (commits text index)
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_flush(db: *mut OmenDB) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.flush() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(format!("Flush failed: {e}"));
            -1
        }
    }
}

/// Update a vector's data and/or metadata
///
/// # Arguments
/// * `db` - Database handle
/// * `id` - Vector ID to update (null-terminated UTF-8)
/// * `vector` - New vector data (NULL to keep existing)
/// * `vector_dim` - Length of vector array (ignored if vector is NULL)
/// * `metadata_json` - New metadata JSON (NULL to keep existing)
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `id` must be a valid, null-terminated UTF-8 string
/// - `vector` must be NULL or point to at least `vector_dim` valid f32 values
/// - `metadata_json` must be NULL or a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_update(
    db: *mut OmenDB,
    id: *const c_char,
    vector: *const f32,
    vector_dim: usize,
    metadata_json: *const c_char,
) -> i32 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if id.is_null() {
        set_last_error("Null id pointer".to_string());
        return -1;
    }

    let id_str = match CStr::from_ptr(id).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid id string: {e}"));
            return -1;
        }
    };

    let vector_opt = if vector.is_null() {
        None
    } else {
        if vector_dim != db.dimensions {
            set_last_error(format!(
                "Vector dimension mismatch: expected {}, got {vector_dim}",
                db.dimensions
            ));
            return -1;
        }
        let data: Vec<f32> = std::slice::from_raw_parts(vector, vector_dim).to_vec();
        Some(Vector::new(data))
    };

    let metadata_opt: Option<JsonValue> = if metadata_json.is_null() {
        None
    } else {
        let meta_str = match CStr::from_ptr(metadata_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid metadata string: {e}"));
                return -1;
            }
        };
        match serde_json::from_str(meta_str) {
            Ok(v) => Some(v),
            Err(e) => {
                set_last_error(format!("Invalid metadata JSON: {e}"));
                return -1;
            }
        }
    };

    match db.store.update(id_str, vector_opt, metadata_opt) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(format!("Update failed: {e}"));
            -1
        }
    }
}

/// Delete vectors matching a metadata filter
///
/// # Arguments
/// * `db` - Database handle
/// * `filter_json` - MongoDB-style filter JSON string
///
/// # Returns
/// Count of deleted items, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `filter_json` must be a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_delete_by_filter(
    db: *mut OmenDB,
    filter_json: *const c_char,
) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if filter_json.is_null() {
        set_last_error("Null filter_json pointer".to_string());
        return -1;
    }

    let filter_str = match CStr::from_ptr(filter_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid filter string: {e}"));
            return -1;
        }
    };

    let filter_value: JsonValue = match serde_json::from_str(filter_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("Invalid filter JSON: {e}"));
            return -1;
        }
    };

    let filter = match MetadataFilter::from_json(&filter_value) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Invalid filter format: {e}"));
            return -1;
        }
    };

    match db.store.delete_by_filter(&filter) {
        Ok(count) => i64::try_from(count).unwrap_or(i64::MAX),
        Err(e) => {
            set_last_error(format!("Delete by filter failed: {e}"));
            -1
        }
    }
}

/// Compact the database by removing deleted records
///
/// # Returns
/// Count of compacted (removed) items, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_compact(db: *mut OmenDB) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.compact() {
        Ok(count) => i64::try_from(count).unwrap_or(i64::MAX),
        Err(e) => {
            set_last_error(format!("Compact failed: {e}"));
            -1
        }
    }
}

/// Optimize index for cache-efficient search
///
/// Reorders nodes for better memory locality, improving search performance.
///
/// # Returns
/// Count of reordered items, or -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_optimize(db: *mut OmenDB) -> i64 {
    clear_last_error();

    let Some(db) = db.as_mut() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    match db.store.optimize() {
        Ok(count) => i64::try_from(count).unwrap_or(i64::MAX),
        Err(e) => {
            set_last_error(format!("Optimize failed: {e}"));
            -1
        }
    }
}

/// Check if an ID exists in the database
///
/// # Returns
/// 1 if exists, 0 if not, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `id` must be a valid, null-terminated UTF-8 string
#[no_mangle]
pub unsafe extern "C" fn omendb_exists(db: *const OmenDB, id: *const c_char) -> i32 {
    clear_last_error();

    let Some(db) = db.as_ref() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if id.is_null() {
        set_last_error("Null id pointer".to_string());
        return -1;
    }

    let id_str = match CStr::from_ptr(id).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid id string: {e}"));
            return -1;
        }
    };

    i32::from(db.store.contains(id_str))
}

/// Get database statistics as JSON
///
/// # Arguments
/// * `db` - Database handle
/// * `result` - Output pointer for stats JSON (caller must free with `omendb_free_string`)
///
/// Result JSON format:
/// ```json
/// {"count": N, "dimensions": D, "quantized": false, "memory_bytes": N}
/// ```
///
/// # Returns
/// 0 on success, -1 on error
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `result` must be a valid pointer to a `*mut c_char`
#[no_mangle]
pub unsafe extern "C" fn omendb_stats(db: *const OmenDB, result: *mut *mut c_char) -> i32 {
    clear_last_error();

    let Some(db) = db.as_ref() else {
        set_last_error("Null database handle".to_string());
        return -1;
    };

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

    let stats = json!({
        "count": db.store.len(),
        "dimensions": db.store.dimensions(),
        "quantized": db.store.is_quantized(),
        "memory_bytes": db.store.memory_usage(),
    });

    let json_str = match serde_json::to_string(&stats) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            *result = cstr.into_raw();
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {e}"));
            -1
        }
    }
}

#[cfg(test)]
mod tests;
