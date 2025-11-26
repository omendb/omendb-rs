//! C FFI bindings for OmenDB
//!
//! Provides a C-compatible API for embedding OmenDB in other languages.
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
//! const char* items = "[{\"id\":\"doc1\",\"embedding\":[0.1,...],\"metadata\":{}}]";
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

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::cell::RefCell;
use std::path::Path;

use crate::vector::{VectorStore, Vector, MetadataFilter};
use serde_json::{json, Value as JsonValue};

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

fn set_last_error(err: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(err));
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
/// Database handle on success, NULL on failure (check omendb_last_error)
#[no_mangle]
pub extern "C" fn omendb_open(
    path: *const c_char,
    dimensions: usize,
    _config_json: *const c_char,
) -> *mut OmenDB {
    clear_last_error();

    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid path: {}", e));
            return ptr::null_mut();
        }
    };

    match VectorStore::open_with_dimensions(Path::new(path), dimensions) {
        Ok(store) => Box::into_raw(Box::new(OmenDB { store, dimensions })),
        Err(e) => {
            set_last_error(format!("Failed to open database: {}", e));
            ptr::null_mut()
        }
    }
}

/// Close database and free resources
#[no_mangle]
pub extern "C" fn omendb_close(db: *mut OmenDB) {
    if !db.is_null() {
        unsafe { drop(Box::from_raw(db)) };
    }
}

/// Insert or replace vectors
///
/// # Arguments
/// * `db` - Database handle
/// * `items_json` - JSON array: [{"id": "...", "embedding": [...], "metadata": {...}}, ...]
///
/// # Returns
/// Number of vectors inserted, or -1 on error
#[no_mangle]
pub extern "C" fn omendb_set(db: *mut OmenDB, items_json: *const c_char) -> i64 {
    clear_last_error();

    let db = match unsafe { db.as_mut() } {
        Some(db) => db,
        None => {
            set_last_error("Null database handle".to_string());
            return -1;
        }
    };

    let items_str = match unsafe { CStr::from_ptr(items_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {}", e));
            return -1;
        }
    };

    let items: Vec<JsonValue> = match serde_json::from_str(items_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {}", e));
            return -1;
        }
    };

    let mut count = 0i64;
    for item in items {
        let id = match item.get("id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => {
                set_last_error("Item missing 'id' field".to_string());
                return -1;
            }
        };

        let embedding: Vec<f32> = match item.get("embedding").and_then(|v| v.as_array()) {
            Some(arr) => arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect(),
            None => {
                set_last_error("Item missing 'embedding' field".to_string());
                return -1;
            }
        };

        let metadata = item.get("metadata").cloned().unwrap_or(json!({}));

        let vector = Vector::new(embedding);
        if let Err(e) = db.store.set(id, vector, metadata) {
            set_last_error(format!("Set failed: {}", e));
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
/// * `ids_json` - JSON array of IDs: ["id1", "id2", ...]
/// * `result` - Output pointer for result JSON (caller must free with omendb_free_string)
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn omendb_get(
    db: *mut OmenDB,
    ids_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let db = match unsafe { db.as_ref() } {
        Some(db) => db,
        None => {
            set_last_error("Null database handle".to_string());
            return -1;
        }
    };

    let ids_str = match unsafe { CStr::from_ptr(ids_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {}", e));
            return -1;
        }
    };

    let ids: Vec<String> = match serde_json::from_str(ids_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {}", e));
            return -1;
        }
    };

    let mut results = Vec::new();
    for id in ids {
        if let Some((vector, metadata)) = db.store.get_by_id(&id) {
            results.push(json!({
                "id": id,
                "embedding": vector.data,
                "metadata": metadata
            }));
        }
    }

    let json_str = match serde_json::to_string(&results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {}", e));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            unsafe { *result = cstr.into_raw() };
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {}", e));
            -1
        }
    }
}

/// Delete vectors by ID
///
/// # Returns
/// Number of vectors deleted, or -1 on error
#[no_mangle]
pub extern "C" fn omendb_delete(db: *mut OmenDB, ids_json: *const c_char) -> i64 {
    clear_last_error();

    let db = match unsafe { db.as_mut() } {
        Some(db) => db,
        None => {
            set_last_error("Null database handle".to_string());
            return -1;
        }
    };

    let ids_str = match unsafe { CStr::from_ptr(ids_json) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid JSON string: {}", e));
            return -1;
        }
    };

    let ids: Vec<String> = match serde_json::from_str(ids_str) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(format!("JSON parse error: {}", e));
            return -1;
        }
    };

    match db.store.delete_batch(&ids) {
        Ok(count) => count as i64,
        Err(e) => {
            set_last_error(format!("Delete failed: {}", e));
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
/// * `result` - Output pointer for result JSON (caller must free with omendb_free_string)
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn omendb_search(
    db: *mut OmenDB,
    query: *const f32,
    query_len: usize,
    k: usize,
    filter_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 {
    clear_last_error();

    let db = match unsafe { db.as_mut() } {
        Some(db) => db,
        None => {
            set_last_error("Null database handle".to_string());
            return -1;
        }
    };

    if query_len != db.dimensions {
        set_last_error(format!(
            "Query dimension mismatch: expected {}, got {}",
            db.dimensions, query_len
        ));
        return -1;
    }

    let query_vec: Vec<f32> = unsafe { std::slice::from_raw_parts(query, query_len) }.to_vec();
    let query = Vector::new(query_vec);

    // Parse filter if provided
    let filter: Option<MetadataFilter> = if !filter_json.is_null() {
        let filter_str = match unsafe { CStr::from_ptr(filter_json) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(format!("Invalid filter string: {}", e));
                return -1;
            }
        };
        // TODO: Parse filter JSON to MetadataFilter
        // For now, ignore filter
        let _ = filter_str;
        None
    } else {
        None
    };

    let search_results = if let Some(_filter) = filter {
        // TODO: Implement filtered search via FFI
        match db.store.knn_search(&query, k) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(format!("Search failed: {}", e));
                return -1;
            }
        }
    } else {
        match db.store.knn_search(&query, k) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(format!("Search failed: {}", e));
                return -1;
            }
        }
    };

    // Convert results to JSON
    let mut json_results = Vec::new();
    for (idx, distance) in search_results {
        if let Some(vector) = db.store.get(idx) {
            // Find the string ID for this index
            let id = db.store.id_to_index.iter()
                .find(|(_, &i)| i == idx)
                .map(|(id, _)| id.clone())
                .unwrap_or_else(|| idx.to_string());

            let metadata = db.store.get_by_id(&id)
                .map(|(_, m)| m.clone())
                .unwrap_or(json!({}));

            json_results.push(json!({
                "id": id,
                "distance": distance,
                "embedding": vector.data,
                "metadata": metadata
            }));
        }
    }

    let json_str = match serde_json::to_string(&json_results) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("JSON serialize error: {}", e));
            return -1;
        }
    };

    match CString::new(json_str) {
        Ok(cstr) => {
            unsafe { *result = cstr.into_raw() };
            0
        }
        Err(e) => {
            set_last_error(format!("CString error: {}", e));
            -1
        }
    }
}

/// Get number of vectors in database
#[no_mangle]
pub extern "C" fn omendb_count(db: *const OmenDB) -> i64 {
    match unsafe { db.as_ref() } {
        Some(db) => db.store.len() as i64,
        None => -1,
    }
}

/// Save database to disk
#[no_mangle]
pub extern "C" fn omendb_save(db: *const OmenDB) -> i32 {
    clear_last_error();

    let db = match unsafe { db.as_ref() } {
        Some(db) => db,
        None => {
            set_last_error("Null database handle".to_string());
            return -1;
        }
    };

    match db.store.flush() {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(format!("Save failed: {}", e));
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
    LAST_ERROR.with(|e| {
        match &*e.borrow() {
            Some(err) => {
                // This is a bit unsafe - we're returning a pointer to thread-local storage
                // In practice this is fine if the caller uses it immediately
                err.as_ptr() as *const c_char
            }
            None => ptr::null(),
        }
    })
}

/// Free a string returned by OmenDB
#[no_mangle]
pub extern "C" fn omendb_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Get OmenDB version
#[no_mangle]
pub extern "C" fn omendb_version() -> *const c_char {
    static VERSION: &[u8] = b"0.0.1\0";
    VERSION.as_ptr() as *const c_char
}
