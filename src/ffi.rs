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

use crate::vector::{MetadataFilter, Vector, VectorStore};
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
    _config_json: *const c_char,
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

    match VectorStore::open_with_dimensions(Path::new(path), dimensions) {
        Ok(store) => Box::into_raw(Box::new(OmenDB { store, dimensions })),
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
            s.to_string()
        } else {
            set_last_error("Item missing 'id' field".to_string());
            return -1;
        };

        let vector_data: Vec<f32> = if let Some(arr) = item.get("vector").and_then(|v| v.as_array())
        {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        } else {
            set_last_error("Item missing 'vector' field".to_string());
            return -1;
        };

        let metadata = item.get("metadata").cloned().unwrap_or(json!({}));

        let vector = Vector::new(vector_data);
        if let Err(e) = db.store.set(id, vector, metadata) {
            set_last_error(format!("Set failed: {e}"));
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
        if let Some((vector, metadata)) = db.store.get_by_id(&id) {
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

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

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
        // Filter parsing not yet implemented in FFI
        let _ = filter_str;
        None
    };

    let search_results = if filter.is_some() {
        // Filtered search not yet exposed via FFI (filter ignored)
        match db.store.knn_search(&query, k) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(format!("Search failed: {e}"));
                return -1;
            }
        }
    } else {
        match db.store.knn_search(&query, k) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(format!("Search failed: {e}"));
                return -1;
            }
        }
    };

    // Convert results to JSON
    let mut json_results = Vec::new();
    for (idx, distance) in search_results {
        if let Some(vector) = db.store.get(idx) {
            // Find the string ID for this index
            let id = db
                .store
                .id_to_index
                .iter()
                .find(|(_, &i)| i == idx)
                .map_or_else(|| idx.to_string(), |(id, _)| id.clone());

            let metadata = db
                .store
                .get_by_id(&id)
                .map(|(_, m)| m.clone())
                .unwrap_or(json!({}));

            json_results.push(json!({
                "id": id,
                "distance": distance,
                "vector": vector.data,
                "metadata": metadata
            }));
        }
    }

    let json_str = match serde_json::to_string(&json_results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {e}"));
            return -1;
        }
    };

    if result.is_null() {
        set_last_error("Output pointer is NULL".to_string());
        return -1;
    }

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
    match db.as_ref() {
        Some(db) => i64::try_from(db.store.len()).unwrap_or(i64::MAX),
        None => -1,
    }
}

/// Save database to disk
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[no_mangle]
pub unsafe extern "C" fn omendb_save(db: *const OmenDB) -> i32 {
    clear_last_error();

    let Some(db) = db.as_ref() else {
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
    static VERSION: &[u8] = b"0.0.1\0";
    VERSION.as_ptr().cast::<c_char>()
}
