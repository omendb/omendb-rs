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
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::ptr;

use omendb::vector::{MetadataFilter, Vector, VectorStore, VectorStoreOptions};
use serde_json::{json, Value as JsonValue};

// ─── Error Handling ──────────────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(err: String) {
    let sanitized = err.replace('\0', "\\0");
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(sanitized).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Wraps an FFI function body with panic catching and error conversion.
///
/// - Clears previous error state
/// - Catches panics (preventing UB from unwinding across the FFI boundary)
/// - Converts `Err(String)` to `set_last_error` + `error_val`
fn ffi_boundary<T, F>(error_val: T, f: F) -> T
where
    F: FnOnce() -> Result<T, String>,
{
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(val)) => val,
        Ok(Err(msg)) => {
            set_last_error(msg);
            error_val
        }
        Err(_) => {
            set_last_error("Internal panic in omendb".to_string());
            error_val
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Read a C string pointer, returning an error if null or invalid UTF-8.
///
/// # Safety
/// `ptr` (if non-null) must point to a valid null-terminated string.
unsafe fn read_cstr<'a>(ptr: *const c_char, name: &str) -> Result<&'a str, String> { unsafe {
    if ptr.is_null() {
        return Err(format!("Null {name} pointer"));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| format!("Invalid {name}: {e}"))
}}

/// Write a JSON string to an output pointer as a CString.
///
/// # Safety
/// `out` must be a valid, writable pointer.
unsafe fn write_result(out: *mut *mut c_char, json: String) -> Result<i32, String> { unsafe {
    if out.is_null() {
        return Err("Output pointer is NULL".to_string());
    }
    let cstr = CString::new(json).map_err(|e| format!("CString error: {e}"))?;
    *out = cstr.into_raw();
    Ok(0)
}}

/// Parse an optional metadata filter from a JSON C string.
/// Returns `Ok(None)` if `ptr` is null.
///
/// # Safety
/// `ptr` (if non-null) must point to a valid null-terminated string.
unsafe fn parse_filter(ptr: *const c_char) -> Result<Option<MetadataFilter>, String> { unsafe {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = read_cstr(ptr, "filter")?;
    let value: JsonValue =
        serde_json::from_str(s).map_err(|e| format!("Invalid filter JSON: {e}"))?;
    let filter =
        MetadataFilter::from_json(&value).map_err(|e| format!("Invalid filter format: {e}"))?;
    Ok(Some(filter))
}}

/// Parse id, vector data, and metadata from a JSON item object.
fn parse_vector_item(item: &JsonValue) -> Result<(&str, Vec<f32>, JsonValue), String> {
    let id = item
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or("Item missing 'id' field")?;

    let arr = item
        .get("vector")
        .and_then(|v| v.as_array())
        .ok_or("Item missing 'vector' field")?;

    let mut data = Vec::with_capacity(arr.len());
    for (j, v) in arr.iter().enumerate() {
        let f = v
            .as_f64()
            .ok_or_else(|| format!("Vector element at index {j} is not a number: {v}"))?;
        data.push(f as f32);
    }

    let metadata = item.get("metadata").cloned().unwrap_or(json!({}));
    Ok((id, data, metadata))
}

// ─── Opaque Handle ───────────────────────────────────────────────────────────

/// Opaque database handle
pub struct OmenDB {
    store: VectorStore,
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

/// Open a database at the given path.
///
/// Config JSON format: `{"m": 16, "ef_construction": 100, "ef_search": 100}`
///
/// Returns database handle on success, NULL on failure (check `omendb_last_error`).
///
/// # Safety
/// - `path` must be a valid, null-terminated UTF-8 string
/// - `config_json` must be NULL or a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_open(
    path: *const c_char,
    dimensions: usize,
    config_json: *const c_char,
) -> *mut OmenDB { unsafe {
    ffi_boundary(ptr::null_mut(), || {
        let path = read_cstr(path, "path")?;

        let config: Option<JsonValue> = if config_json.is_null() {
            None
        } else {
            let s = read_cstr(config_json, "config")?;
            Some(serde_json::from_str(s).map_err(|e| format!("Invalid config JSON: {e}"))?)
        };

        if let Some(ref cfg) = config {
            if cfg.get("multi_vector").is_some() {
                return Err("Multi-vector stores are not supported in the C FFI. \
                    Use the Python or Node.js bindings."
                    .to_string());
            }
        }

        let store = if let Some(cfg) = config {
            let mut opts = VectorStoreOptions::new().dimensions(dimensions);
            if let Some(m) = cfg.get("m").and_then(JsonValue::as_u64) {
                opts = opts.m(m as usize);
            }
            if let Some(ef_c) = cfg.get("ef_construction").and_then(JsonValue::as_u64) {
                opts = opts.ef_construction(ef_c as usize);
            }
            if let Some(ef_s) = cfg.get("ef_search").and_then(JsonValue::as_u64) {
                opts = opts.ef_search(ef_s as usize);
            }
            opts.open(Path::new(path))
        } else {
            VectorStore::open_with_dimensions(Path::new(path), dimensions)
        }
        .map_err(|e| format!("Failed to open database: {e}"))?;

        if store.is_multi_vector() {
            return Err("Cannot open multi-vector store via C FFI. \
                Use the Python or Node.js bindings."
                .to_string());
        }

        Ok(Box::into_raw(Box::new(OmenDB { store })))
    })
}}

/// Close database and free resources.
///
/// # Safety
/// - `db` must be NULL or a valid pointer returned by `omendb_open`
/// - After calling, `db` is invalid and must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_close(db: *mut OmenDB) { unsafe {
    if !db.is_null() {
        drop(Box::from_raw(db));
    }
}}

/// Get `OmenDB` version string.
#[unsafe(no_mangle)]
pub extern "C" fn omendb_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0")
        .as_ptr()
        .cast::<c_char>()
}

/// Get last error message. Returns NULL if no error. Valid until next FFI call.
#[unsafe(no_mangle)]
pub extern "C" fn omendb_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

/// Free a string returned by `OmenDB`.
///
/// # Safety
/// - `s` must be NULL or a valid pointer returned by an `OmenDB` function
/// - After calling, `s` is invalid and must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_free_string(s: *mut c_char) { unsafe {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}}

// ─── CRUD ────────────────────────────────────────────────────────────────────

/// Insert or replace vectors.
///
/// Items JSON: `[{"id": "...", "vector": [...], "metadata": {...}}, ...]`
///
/// Returns number of vectors inserted, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `items_json` must be a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_set(db: *mut OmenDB, items_json: *const c_char) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let items_str = read_cstr(items_json, "items_json")?;
        let items: Vec<JsonValue> =
            serde_json::from_str(items_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let mut count = 0i64;
        for item in &items {
            let (id, data, metadata) = parse_vector_item(item)?;
            db.store
                .set(id, Vector::new(data), metadata)
                .map_err(|e| format!("Set failed after {count} items: {e}"))?;
            count += 1;
        }
        Ok(count)
    })
}}

/// Get vectors by ID.
///
/// Returns 0 on success, -1 on error. Result JSON written to `*result`
/// (caller must free with `omendb_free_string`).
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `ids_json` must be a valid, null-terminated UTF-8 string
/// - `result` must be a valid pointer to a `*mut c_char`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_get(
    db: *mut OmenDB,
    ids_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        let ids_str = read_cstr(ids_json, "ids_json")?;
        let ids: Vec<String> =
            serde_json::from_str(ids_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let results: Vec<JsonValue> = ids
            .iter()
            .filter_map(|id| {
                db.store.get(id).map(|(vector, metadata)| {
                    json!({"id": id, "vector": vector.data, "metadata": metadata})
                })
            })
            .collect();

        let json =
            serde_json::to_string(&results).map_err(|e| format!("JSON serialize error: {e}"))?;
        write_result(result, json)
    })
}}

/// Delete vectors by ID.
///
/// Returns number of vectors deleted, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `ids_json` must be a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_delete(db: *mut OmenDB, ids_json: *const c_char) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let ids_str = read_cstr(ids_json, "ids_json")?;
        let ids: Vec<String> =
            serde_json::from_str(ids_str).map_err(|e| format!("JSON parse error: {e}"))?;
        let count = db
            .store
            .delete_batch(&ids)
            .map_err(|e| format!("Delete failed: {e}"))?;
        Ok(i64::try_from(count).unwrap_or(i64::MAX))
    })
}}

/// Check if a vector ID exists.
///
/// Returns 1 if exists, 0 if not, -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `id` must be a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_exists(db: *const OmenDB, id: *const c_char) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        let id_str = read_cstr(id, "id")?;
        Ok(i32::from(db.store.contains(id_str)))
    })
}}

/// Get number of vectors in database. Returns -1 if db is NULL.
///
/// # Safety
/// - `db` must be NULL or a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_count(db: *const OmenDB) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        Ok(i64::try_from(db.store.len()).unwrap_or(i64::MAX))
    })
}}

/// Update a vector's data and/or metadata.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `id` must be a valid, null-terminated UTF-8 string
/// - `vector` must be NULL or point to at least `vector_dim` valid f32 values
/// - `metadata_json` must be NULL or a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_update(
    db: *mut OmenDB,
    id: *const c_char,
    vector: *const f32,
    vector_dim: usize,
    metadata_json: *const c_char,
) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let id_str = read_cstr(id, "id")?;

        let vector_opt = if vector.is_null() {
            None
        } else {
            let dims = db.store.dimensions();
            if vector_dim != dims {
                return Err(format!(
                    "Vector dimension mismatch: expected {dims}, got {vector_dim}"
                ));
            }
            let data = std::slice::from_raw_parts(vector, vector_dim).to_vec();
            Some(Vector::new(data))
        };

        let metadata_opt: Option<JsonValue> = if metadata_json.is_null() {
            None
        } else {
            let s = read_cstr(metadata_json, "metadata")?;
            Some(serde_json::from_str(s).map_err(|e| format!("Invalid metadata JSON: {e}"))?)
        };

        db.store
            .update(id_str, vector_opt, metadata_opt)
            .map_err(|e| format!("Update failed: {e}"))?;
        Ok(0)
    })
}}

/// Delete vectors matching a metadata filter.
///
/// Returns count of deleted vectors, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `filter_json` must be a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_delete_by_filter(
    db: *mut OmenDB,
    filter_json: *const c_char,
) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let filter_str = read_cstr(filter_json, "filter_json")?;
        let value: JsonValue =
            serde_json::from_str(filter_str).map_err(|e| format!("Invalid filter JSON: {e}"))?;
        let filter =
            MetadataFilter::from_json(&value).map_err(|e| format!("Invalid filter format: {e}"))?;
        let count = db
            .store
            .delete_by_filter(&filter)
            .map_err(|e| format!("Delete by filter failed: {e}"))?;
        Ok(i64::try_from(count).unwrap_or(i64::MAX))
    })
}}

// ─── Search ──────────────────────────────────────────────────────────────────

/// Search for similar vectors.
///
/// Returns 0 on success, -1 on error. Result JSON written to `*result`.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `query` must point to at least `query_len` valid f32 values
/// - `filter_json` must be NULL or a valid, null-terminated UTF-8 string
/// - `result` must be a valid pointer to a `*mut c_char`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_search(
    db: *mut OmenDB,
    query: *const f32,
    query_len: usize,
    k: usize,
    filter_json: *const c_char,
    result: *mut *mut c_char,
) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        if query.is_null() {
            return Err("Null query pointer".to_string());
        }
        let dims = db.store.dimensions();
        if query_len != dims {
            return Err(format!(
                "Query dimension mismatch: expected {dims}, got {query_len}"
            ));
        }

        let query_vec = std::slice::from_raw_parts(query, query_len).to_vec();
        let query = Vector::new(query_vec);
        let filter = parse_filter(filter_json)?;

        let results = db
            .store
            .search(&query, k, filter.as_ref())
            .map_err(|e| format!("Search failed: {e}"))?;

        let json_results: Vec<JsonValue> = results
            .into_iter()
            .map(|r| json!({"id": r.id, "distance": r.distance, "metadata": r.metadata}))
            .collect();

        let json = serde_json::to_string(&json_results)
            .map_err(|e| format!("JSON serialize error: {e}"))?;
        write_result(result, json)
    })
}}

// ─── Text & Hybrid Search ────────────────────────────────────────────────────

/// Enable text search (BM25) for hybrid search.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_enable_text_search(db: *mut OmenDB) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        db.store
            .enable_text_search()
            .map_err(|e| format!("Failed to enable text search: {e}"))?;
        Ok(0)
    })
}}

/// Check if text search is enabled. Returns 1 if enabled, 0 if not, -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_has_text_search(db: *const OmenDB) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        Ok(i32::from(db.store.has_text_search()))
    })
}}

/// Insert vectors with associated text for hybrid search.
///
/// Items JSON: `[{"id": "...", "vector": [...], "text": "...", "metadata": {...}}, ...]`
///
/// Returns number of vectors inserted, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `items_json` must be a valid, null-terminated UTF-8 string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_set_with_text(db: *mut OmenDB, items_json: *const c_char) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        if !db.store.has_text_search() {
            return Err(
                "Text search not enabled. Call omendb_enable_text_search first.".to_string(),
            );
        }

        let items_str = read_cstr(items_json, "items_json")?;
        let items: Vec<JsonValue> =
            serde_json::from_str(items_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let mut count = 0i64;
        for item in &items {
            let (id, data, metadata) = parse_vector_item(item)?;
            let text = item
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or("Item missing 'text' field")?;
            db.store
                .set_with_text(id, Vector::new(data), text, metadata)
                .map_err(|e| format!("Set with text failed after {count} items: {e}"))?;
            count += 1;
        }
        Ok(count)
    })
}}

/// Text-only search using BM25.
///
/// Returns 0 on success, -1 on error. Result JSON written to `*result`.
///
/// # Safety
/// - All pointer arguments must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_text_search(
    db: *mut OmenDB,
    query: *const c_char,
    k: usize,
    result: *mut *mut c_char,
) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        let query_str = read_cstr(query, "query")?;

        let search_results = db
            .store
            .search_text(query_str, k)
            .map_err(|e| format!("Text search failed: {e}"))?;

        let json_results: Vec<JsonValue> = search_results
            .into_iter()
            .map(|(id, score)| {
                let metadata = db.store.get(&id).map(|(_, meta)| meta).unwrap_or(json!({}));
                json!({"id": id, "score": score, "metadata": metadata})
            })
            .collect();

        let json = serde_json::to_string(&json_results)
            .map_err(|e| format!("JSON serialize error: {e}"))?;
        write_result(result, json)
    })
}}

/// Hybrid search combining vector similarity and BM25 text search.
///
/// Returns 0 on success, -1 on error. Result JSON written to `*result`.
///
/// # Safety
/// - All pointer arguments must be valid (except `filter_json` which can be NULL)
#[unsafe(no_mangle)]
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
) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        if query_vector.is_null() {
            return Err("Null query_vector pointer".to_string());
        }
        let dims = db.store.dimensions();
        if query_len != dims {
            return Err(format!(
                "Query dimension mismatch: expected {dims}, got {query_len}"
            ));
        }

        let text_str = read_cstr(query_text, "query_text")?;
        let query_vec = std::slice::from_raw_parts(query_vector, query_len).to_vec();
        let vector = Vector::new(query_vec);

        let alpha_opt = if alpha < 0.0 { None } else { Some(alpha) };
        let rrf_k_opt = if rrf_k == 0 { None } else { Some(rrf_k) };
        let filter = parse_filter(filter_json)?;

        let search_results = db
            .store
            .search_hybrid(&vector, text_str, k, filter.as_ref(), alpha_opt, rrf_k_opt)
            .map_err(|e| format!("Hybrid search failed: {e}"))?;

        let json_results: Vec<JsonValue> = search_results
            .into_iter()
            .map(|(id, score, metadata)| json!({"id": id, "score": score, "metadata": metadata}))
            .collect();

        let json = serde_json::to_string(&json_results)
            .map_err(|e| format!("JSON serialize error: {e}"))?;
        write_result(result, json)
    })
}}

// ─── Maintenance ─────────────────────────────────────────────────────────────

/// Flush pending changes to disk (commits text index, persists data).
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_flush(db: *mut OmenDB) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        db.store.flush().map_err(|e| format!("Flush failed: {e}"))?;
        Ok(0)
    })
}}

/// Compact the database by removing deleted records.
///
/// Returns count of compacted items, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_compact(db: *mut OmenDB) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let count = db
            .store
            .compact()
            .map_err(|e| format!("Compact failed: {e}"))?;
        Ok(i64::try_from(count).unwrap_or(i64::MAX))
    })
}}

/// Optimize index for cache-efficient search. Reorders nodes for better memory locality.
///
/// Returns count of reordered items, or -1 on error.
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_optimize(db: *mut OmenDB) -> i64 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_mut().ok_or("Null database handle")?;
        let stats = db
            .store
            .optimize()
            .map_err(|e| format!("Optimize failed: {e}"))?;
        Ok(i64::try_from(stats.vectors_reordered).unwrap_or(i64::MAX))
    })
}}

// ─── Info ────────────────────────────────────────────────────────────────────

/// Get database statistics as JSON.
///
/// Result format: `{"count": N, "dimensions": D, "quantized": false, "memory_bytes": N}`
///
/// # Safety
/// - `db` must be a valid pointer returned by `omendb_open`
/// - `result` must be a valid pointer to a `*mut c_char`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn omendb_stats(db: *const OmenDB, result: *mut *mut c_char) -> i32 { unsafe {
    ffi_boundary(-1, || {
        let db = db.as_ref().ok_or("Null database handle")?;
        let stats = json!({
            "count": db.store.len(),
            "dimensions": db.store.dimensions(),
            "quantized": db.store.is_quantized(),
            "memory_bytes": db.store.memory_usage(),
        });
        let json =
            serde_json::to_string(&stats).map_err(|e| format!("JSON serialize error: {e}"))?;
        write_result(result, json)
    })
}}

#[cfg(test)]
mod tests;
