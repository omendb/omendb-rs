use super::*;
use serde_json::Value;
use std::ffi::CString;
use std::ptr;
use tempfile::TempDir;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn c_str(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn read_and_free(ptr: *mut c_char) -> String {
    assert!(!ptr.is_null(), "expected non-null result pointer");
    let s = CStr::from_ptr(ptr).to_str().unwrap().to_string();
    omendb_free_string(ptr);
    s
}

fn last_error() -> Option<String> {
    let ptr = omendb_last_error();
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(ptr).to_str().unwrap().to_string() })
    }
}

fn open_db(dim: usize) -> (*mut OmenDB, TempDir) {
    let dir = TempDir::new().unwrap();
    let path = c_str(dir.path().to_str().unwrap());
    let db = unsafe { omendb_open(path.as_ptr(), dim, ptr::null()) };
    assert!(!db.is_null(), "open failed: {:?}", last_error());
    (db, dir)
}

fn insert(db: *mut OmenDB, id: &str, vector: &[f32], metadata: Value) -> i64 {
    let item = serde_json::json!([{"id": id, "vector": vector, "metadata": metadata}]);
    let json = c_str(&item.to_string());
    unsafe { omendb_set(db, json.as_ptr()) }
}

// ─── Version & Error ─────────────────────────────────────────────────────────

#[test]
fn version() {
    let ptr = omendb_version();
    assert!(!ptr.is_null());
    let version = unsafe { CStr::from_ptr(ptr).to_str().unwrap() };
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
}

#[test]
fn last_error_initially_null() {
    clear_last_error();
    assert!(omendb_last_error().is_null());
}

// ─── Open / Close ────────────────────────────────────────────────────────────

#[test]
fn open_and_close() {
    let (db, _dir) = open_db(4);
    unsafe { omendb_close(db) };
}

#[test]
fn open_null_path() {
    let db = unsafe { omendb_open(ptr::null(), 4, ptr::null()) };
    assert!(db.is_null());
    assert_eq!(last_error().unwrap(), "Null path pointer");
}

#[test]
fn open_with_config() {
    let dir = TempDir::new().unwrap();
    let path = c_str(dir.path().to_str().unwrap());
    let config = c_str(r#"{"m": 8, "ef_construction": 50, "ef_search": 50}"#);
    let db = unsafe { omendb_open(path.as_ptr(), 4, config.as_ptr()) };
    assert!(!db.is_null(), "open with config failed: {:?}", last_error());
    unsafe { omendb_close(db) };
}

#[test]
fn open_invalid_config_json() {
    let dir = TempDir::new().unwrap();
    let path = c_str(dir.path().to_str().unwrap());
    let config = c_str("not json");
    let db = unsafe { omendb_open(path.as_ptr(), 4, config.as_ptr()) };
    assert!(db.is_null());
    let err = last_error().unwrap();
    assert!(err.contains("Invalid config JSON"), "got: {err}");
}

#[test]
fn open_rejects_multi_vector_config() {
    let dir = TempDir::new().unwrap();
    let path = c_str(dir.path().to_str().unwrap());
    let config = c_str(r#"{"multi_vector": {"n_tokens": 32}}"#);
    let db = unsafe { omendb_open(path.as_ptr(), 4, config.as_ptr()) };
    assert!(db.is_null());
    let err = last_error().unwrap();
    assert!(err.contains("not supported"), "got: {err}");
}

#[test]
fn close_null_is_safe() {
    unsafe { omendb_close(ptr::null_mut()) };
}

// ─── Set / Get / Exists / Count ──────────────────────────────────────────────

#[test]
fn set_and_get() {
    let (db, _dir) = open_db(4);

    let count = insert(
        db,
        "doc1",
        &[1.0, 2.0, 3.0, 4.0],
        serde_json::json!({"key": "val"}),
    );
    assert_eq!(count, 1);

    let ids = c_str(r#"["doc1"]"#);
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_get(db, ids.as_ptr(), &mut result) };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["id"], "doc1");
    assert_eq!(results[0]["metadata"]["key"], "val");

    unsafe { omendb_close(db) };
}

#[test]
fn set_multiple_items() {
    let (db, _dir) = open_db(4);
    let items = serde_json::json!([
        {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}},
        {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {}},
        {"id": "c", "vector": [0.0, 0.0, 1.0, 0.0], "metadata": {}}
    ]);
    let json = c_str(&items.to_string());
    let count = unsafe { omendb_set(db, json.as_ptr()) };
    assert_eq!(count, 3);
    assert_eq!(unsafe { omendb_count(db) }, 3);

    unsafe { omendb_close(db) };
}

#[test]
fn set_null_db() {
    let items = c_str("[]");
    assert_eq!(unsafe { omendb_set(ptr::null_mut(), items.as_ptr()) }, -1);
    assert!(last_error().unwrap().contains("Null database"));
}

#[test]
fn set_null_json() {
    let (db, _dir) = open_db(4);
    assert_eq!(unsafe { omendb_set(db, ptr::null()) }, -1);
    assert!(last_error().unwrap().contains("Null items_json"));
    unsafe { omendb_close(db) };
}

#[test]
fn set_missing_id_field() {
    let (db, _dir) = open_db(4);
    let items = c_str(r#"[{"vector": [1.0, 2.0, 3.0, 4.0]}]"#);
    assert_eq!(unsafe { omendb_set(db, items.as_ptr()) }, -1);
    assert!(last_error().unwrap().contains("missing 'id'"));
    unsafe { omendb_close(db) };
}

#[test]
fn set_missing_vector_field() {
    let (db, _dir) = open_db(4);
    let items = c_str(r#"[{"id": "x"}]"#);
    assert_eq!(unsafe { omendb_set(db, items.as_ptr()) }, -1);
    assert!(last_error().unwrap().contains("missing 'vector'"));
    unsafe { omendb_close(db) };
}

#[test]
fn set_invalid_json() {
    let (db, _dir) = open_db(4);
    let items = c_str("not json");
    assert_eq!(unsafe { omendb_set(db, items.as_ptr()) }, -1);
    assert!(last_error().unwrap().contains("JSON parse error"));
    unsafe { omendb_close(db) };
}

#[test]
fn set_non_numeric_vector_element() {
    let (db, _dir) = open_db(4);
    let items = c_str(r#"[{"id": "x", "vector": [1.0, "bad", 3.0, 4.0]}]"#);
    assert_eq!(unsafe { omendb_set(db, items.as_ptr()) }, -1);
    let err = last_error().unwrap();
    assert!(err.contains("not a number"), "got: {err}");
    unsafe { omendb_close(db) };
}

#[test]
fn get_nonexistent() {
    let (db, _dir) = open_db(4);
    let ids = c_str(r#"["doesnt_exist"]"#);
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_get(db, ids.as_ptr(), &mut result) };
    assert_eq!(rc, 0);
    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert!(results.is_empty());
    unsafe { omendb_close(db) };
}

#[test]
fn get_null_output_pointer() {
    let (db, _dir) = open_db(4);
    let ids = c_str(r#"["x"]"#);
    let rc = unsafe { omendb_get(db, ids.as_ptr(), ptr::null_mut()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Output pointer"));
    unsafe { omendb_close(db) };
}

#[test]
fn delete_and_count() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));
    insert(db, "b", &[0.0, 1.0, 0.0, 0.0], serde_json::json!({}));
    assert_eq!(unsafe { omendb_count(db) }, 2);

    let ids = c_str(r#"["a"]"#);
    let deleted = unsafe { omendb_delete(db, ids.as_ptr()) };
    assert_eq!(deleted, 1);
    assert_eq!(unsafe { omendb_count(db) }, 1);

    assert_eq!(unsafe { omendb_exists(db, c_str("a").as_ptr()) }, 0);
    assert_eq!(unsafe { omendb_exists(db, c_str("b").as_ptr()) }, 1);

    unsafe { omendb_close(db) };
}

#[test]
fn exists_null_id() {
    let (db, _dir) = open_db(4);
    assert_eq!(unsafe { omendb_exists(db, ptr::null()) }, -1);
    assert!(last_error().unwrap().contains("Null id"));
    unsafe { omendb_close(db) };
}

#[test]
fn count_null_db() {
    assert_eq!(unsafe { omendb_count(ptr::null()) }, -1);
}

// ─── Update ──────────────────────────────────────────────────────────────────

#[test]
fn update_metadata_only() {
    let (db, _dir) = open_db(4);
    insert(
        db,
        "doc1",
        &[1.0, 2.0, 3.0, 4.0],
        serde_json::json!({"v": 1}),
    );

    let id = c_str("doc1");
    let meta = c_str(r#"{"v": 2}"#);
    let rc = unsafe { omendb_update(db, id.as_ptr(), ptr::null(), 0, meta.as_ptr()) };
    assert_eq!(rc, 0);

    let ids = c_str(r#"["doc1"]"#);
    let mut result: *mut c_char = ptr::null_mut();
    unsafe { omendb_get(db, ids.as_ptr(), &mut result) };
    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(results[0]["metadata"]["v"], 2);

    unsafe { omendb_close(db) };
}

#[test]
fn update_vector_only() {
    let (db, _dir) = open_db(4);
    insert(
        db,
        "doc1",
        &[1.0, 2.0, 3.0, 4.0],
        serde_json::json!({"keep": true}),
    );

    let id = c_str("doc1");
    let new_vec: Vec<f32> = vec![9.0, 8.0, 7.0, 6.0];
    let rc = unsafe { omendb_update(db, id.as_ptr(), new_vec.as_ptr(), 4, ptr::null()) };
    assert_eq!(rc, 0);

    let ids = c_str(r#"["doc1"]"#);
    let mut result: *mut c_char = ptr::null_mut();
    unsafe { omendb_get(db, ids.as_ptr(), &mut result) };
    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(results[0]["metadata"]["keep"], true);
    let vec_data: Vec<f64> = results[0]["vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(vec_data, vec![9.0, 8.0, 7.0, 6.0]);

    unsafe { omendb_close(db) };
}

#[test]
fn update_dimension_mismatch() {
    let (db, _dir) = open_db(4);
    insert(db, "doc1", &[1.0, 2.0, 3.0, 4.0], serde_json::json!({}));

    let id = c_str("doc1");
    let wrong_vec: Vec<f32> = vec![1.0, 2.0];
    let rc = unsafe { omendb_update(db, id.as_ptr(), wrong_vec.as_ptr(), 2, ptr::null()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("dimension mismatch"));

    unsafe { omendb_close(db) };
}

#[test]
fn update_null_id() {
    let (db, _dir) = open_db(4);
    let rc = unsafe { omendb_update(db, ptr::null(), ptr::null(), 0, ptr::null()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Null id"));
    unsafe { omendb_close(db) };
}

// ─── Search ──────────────────────────────────────────────────────────────────

#[test]
fn search_basic() {
    let (db, _dir) = open_db(4);
    let items = serde_json::json!([
        {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"label": "x"}},
        {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {"label": "y"}},
        {"id": "c", "vector": [0.0, 0.0, 1.0, 0.0], "metadata": {"label": "z"}}
    ]);
    let json = c_str(&items.to_string());
    unsafe { omendb_set(db, json.as_ptr()) };

    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, query.as_ptr(), 4, 2, ptr::null(), &mut result) };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0]["id"], "a");

    unsafe { omendb_close(db) };
}

#[test]
fn search_dimension_mismatch() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));

    let query: Vec<f32> = vec![1.0, 0.0];
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, query.as_ptr(), 2, 1, ptr::null(), &mut result) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("dimension mismatch"));

    unsafe { omendb_close(db) };
}

#[test]
fn search_null_query() {
    let (db, _dir) = open_db(4);
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, ptr::null(), 4, 1, ptr::null(), &mut result) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Null query"));
    unsafe { omendb_close(db) };
}

#[test]
fn search_null_result_pointer() {
    let (db, _dir) = open_db(4);
    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let rc = unsafe { omendb_search(db, query.as_ptr(), 4, 1, ptr::null(), ptr::null_mut()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Output pointer"));
    unsafe { omendb_close(db) };
}

#[test]
fn search_with_filter() {
    let (db, _dir) = open_db(4);
    let items = serde_json::json!([
        {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"category": "x"}},
        {"id": "b", "vector": [0.9, 0.1, 0.0, 0.0], "metadata": {"category": "y"}},
        {"id": "c", "vector": [0.8, 0.2, 0.0, 0.0], "metadata": {"category": "x"}}
    ]);
    let json = c_str(&items.to_string());
    unsafe { omendb_set(db, json.as_ptr()) };

    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let filter = c_str(r#"{"category": {"$eq": "x"}}"#);
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, query.as_ptr(), 4, 10, filter.as_ptr(), &mut result) };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r["metadata"]["category"], "x");
    }

    unsafe { omendb_close(db) };
}

#[test]
fn search_invalid_filter_json() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));

    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let filter = c_str("not json");
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, query.as_ptr(), 4, 1, filter.as_ptr(), &mut result) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Invalid filter JSON"));

    unsafe { omendb_close(db) };
}

#[test]
fn search_empty_db() {
    let (db, _dir) = open_db(4);
    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_search(db, query.as_ptr(), 4, 5, ptr::null(), &mut result) };
    assert_eq!(rc, 0);
    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert!(results.is_empty());
    unsafe { omendb_close(db) };
}

// ─── Text & Hybrid Search ────────────────────────────────────────────────────

#[test]
fn has_text_search_toggle() {
    let (db, _dir) = open_db(4);
    assert_eq!(unsafe { omendb_has_text_search(db) }, 0);
    assert_eq!(unsafe { omendb_enable_text_search(db) }, 0);
    assert_eq!(unsafe { omendb_has_text_search(db) }, 1);
    unsafe { omendb_close(db) };
}

#[test]
fn set_with_text_requires_enable() {
    let (db, _dir) = open_db(4);
    let items = c_str(r#"[{"id":"a","vector":[1,0,0,0],"text":"hello","metadata":{}}]"#);
    let rc = unsafe { omendb_set_with_text(db, items.as_ptr()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("not enabled"));
    unsafe { omendb_close(db) };
}

#[test]
fn set_with_text_missing_text_field() {
    let (db, _dir) = open_db(4);
    unsafe { omendb_enable_text_search(db) };
    let items = c_str(r#"[{"id":"a","vector":[1,0,0,0],"metadata":{}}]"#);
    let rc = unsafe { omendb_set_with_text(db, items.as_ptr()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("missing 'text'"));
    unsafe { omendb_close(db) };
}

#[test]
fn text_search() {
    let (db, _dir) = open_db(4);
    unsafe { omendb_enable_text_search(db) };

    let items = serde_json::json!([
        {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "text": "the quick brown fox", "metadata": {}},
        {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0], "text": "lazy dog sleeps all day", "metadata": {}}
    ]);
    let json = c_str(&items.to_string());
    unsafe { omendb_set_with_text(db, json.as_ptr()) };
    unsafe { omendb_flush(db) };

    let query = c_str("fox");
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_text_search(db, query.as_ptr(), 5, &mut result) };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0]["id"], "a");

    unsafe { omendb_close(db) };
}

#[test]
fn text_search_null_query() {
    let (db, _dir) = open_db(4);
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_text_search(db, ptr::null(), 5, &mut result) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Null query"));
    unsafe { omendb_close(db) };
}

#[test]
fn hybrid_search() {
    let (db, _dir) = open_db(4);
    unsafe { omendb_enable_text_search(db) };

    let items = serde_json::json!([
        {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "text": "machine learning models", "metadata": {}},
        {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0], "text": "deep neural networks", "metadata": {}}
    ]);
    let json = c_str(&items.to_string());
    unsafe { omendb_set_with_text(db, json.as_ptr()) };
    unsafe { omendb_flush(db) };

    let query_vec: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let query_text = c_str("machine learning");
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe {
        omendb_hybrid_search(
            db,
            query_vec.as_ptr(),
            4,
            query_text.as_ptr(),
            5,
            -1.0,
            0,
            ptr::null(),
            &mut result,
        )
    };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let results: Vec<Value> = serde_json::from_str(&json_str).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0]["id"], "a");

    unsafe { omendb_close(db) };
}

#[test]
fn hybrid_search_dimension_mismatch() {
    let (db, _dir) = open_db(4);
    unsafe { omendb_enable_text_search(db) };

    let query_vec: Vec<f32> = vec![1.0, 0.0];
    let query_text = c_str("test");
    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe {
        omendb_hybrid_search(
            db,
            query_vec.as_ptr(),
            2,
            query_text.as_ptr(),
            5,
            -1.0,
            0,
            ptr::null(),
            &mut result,
        )
    };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("dimension mismatch"));

    unsafe { omendb_close(db) };
}

// ─── Maintenance ─────────────────────────────────────────────────────────────

#[test]
fn save_and_reopen() {
    let dir = TempDir::new().unwrap();
    let path_str = dir.path().to_str().unwrap();

    {
        let path = c_str(path_str);
        let db = unsafe { omendb_open(path.as_ptr(), 4, ptr::null()) };
        assert!(!db.is_null());
        insert(
            db,
            "persist",
            &[1.0, 2.0, 3.0, 4.0],
            serde_json::json!({"saved": true}),
        );
        assert_eq!(unsafe { omendb_save(db) }, 0);
        unsafe { omendb_close(db) };
    }

    {
        let path = c_str(path_str);
        let db = unsafe { omendb_open(path.as_ptr(), 4, ptr::null()) };
        assert!(!db.is_null());
        assert_eq!(unsafe { omendb_count(db) }, 1);
        assert_eq!(unsafe { omendb_exists(db, c_str("persist").as_ptr()) }, 1);
        unsafe { omendb_close(db) };
    }
}

#[test]
fn compact() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));
    insert(db, "b", &[0.0, 1.0, 0.0, 0.0], serde_json::json!({}));

    let ids = c_str(r#"["a"]"#);
    unsafe { omendb_delete(db, ids.as_ptr()) };

    let compacted = unsafe { omendb_compact(db) };
    assert!(compacted >= 0);
    assert_eq!(unsafe { omendb_count(db) }, 1);

    unsafe { omendb_close(db) };
}

#[test]
fn optimize() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));
    insert(db, "b", &[0.0, 1.0, 0.0, 0.0], serde_json::json!({}));

    let count = unsafe { omendb_optimize(db) };
    assert!(count >= 0);

    unsafe { omendb_close(db) };
}

#[test]
fn flush() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));
    assert_eq!(unsafe { omendb_flush(db) }, 0);
    unsafe { omendb_close(db) };
}

// ─── Stats ───────────────────────────────────────────────────────────────────

#[test]
fn stats() {
    let (db, _dir) = open_db(4);
    insert(db, "a", &[1.0, 0.0, 0.0, 0.0], serde_json::json!({}));

    let mut result: *mut c_char = ptr::null_mut();
    let rc = unsafe { omendb_stats(db, &mut result) };
    assert_eq!(rc, 0);

    let json_str = unsafe { read_and_free(result) };
    let stats: Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(stats["count"], 1);
    assert_eq!(stats["dimensions"], 4);
    assert_eq!(stats["quantized"], false);
    assert!(stats["memory_bytes"].as_u64().unwrap() > 0);

    unsafe { omendb_close(db) };
}

#[test]
fn stats_null_result_pointer() {
    let (db, _dir) = open_db(4);
    let rc = unsafe { omendb_stats(db, ptr::null_mut()) };
    assert_eq!(rc, -1);
    assert!(last_error().unwrap().contains("Output pointer"));
    unsafe { omendb_close(db) };
}

// ─── Delete by Filter ───────────────────────────────────────────────────────

#[test]
fn delete_by_filter() {
    let (db, _dir) = open_db(4);
    insert(
        db,
        "a",
        &[1.0, 0.0, 0.0, 0.0],
        serde_json::json!({"type": "keep"}),
    );
    insert(
        db,
        "b",
        &[0.0, 1.0, 0.0, 0.0],
        serde_json::json!({"type": "remove"}),
    );
    insert(
        db,
        "c",
        &[0.0, 0.0, 1.0, 0.0],
        serde_json::json!({"type": "remove"}),
    );

    let filter = c_str(r#"{"type": {"$eq": "remove"}}"#);
    let deleted = unsafe { omendb_delete_by_filter(db, filter.as_ptr()) };
    assert_eq!(deleted, 2);
    assert_eq!(unsafe { omendb_count(db) }, 1);
    assert_eq!(unsafe { omendb_exists(db, c_str("a").as_ptr()) }, 1);

    unsafe { omendb_close(db) };
}

#[test]
fn delete_by_filter_null_json() {
    let (db, _dir) = open_db(4);
    assert_eq!(unsafe { omendb_delete_by_filter(db, ptr::null()) }, -1);
    assert!(last_error().unwrap().contains("Null filter_json"));
    unsafe { omendb_close(db) };
}

#[test]
fn delete_by_filter_invalid_json() {
    let (db, _dir) = open_db(4);
    let filter = c_str("not json");
    assert_eq!(unsafe { omendb_delete_by_filter(db, filter.as_ptr()) }, -1);
    assert!(last_error().unwrap().contains("Invalid filter JSON"));
    unsafe { omendb_close(db) };
}

// ─── Free String ─────────────────────────────────────────────────────────────

#[test]
fn free_string_null_is_safe() {
    unsafe { omendb_free_string(ptr::null_mut()) };
}
