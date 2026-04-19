use super::*;
use std::sync::Arc;
use std::thread;

#[test]
#[ignore = "stress lane; run with `cargo test --lib test_write_contention_dense_inserts --release -- --ignored`"]
fn test_write_contention_dense_inserts() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("contention_dense");
    let dim = 128;

    let store = Arc::new(VectorStore::create(&db_path, dense_schema(dim)).unwrap());

    let num_threads = 8;
    let inserts_per_thread = 100;

    let mut handles = vec![];
    for t in 0..num_threads {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..inserts_per_thread {
                let id = format!("vec_{}_{}", t, i);
                let vec = random_vector(dim as usize, (t * 1000 + i) as usize);
                store
                    .set(&id, vec, serde_json::json!({"t": t, "i": i}))
                    .unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(store.len(), (num_threads * inserts_per_thread));

    // Verify a random sample
    let test_id = format!("vec_{}_{}", num_threads / 2, inserts_per_thread / 2);
    let record = store.get(&test_id).unwrap();
    assert_eq!(record.1["t"], num_threads / 2);
}

#[test]
#[ignore = "stress lane; run with `cargo test --lib test_write_contention_mixed_ops --release -- --ignored`"]
fn test_write_contention_mixed_ops() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("contention_mixed");
    let dim = 4;

    let store = Arc::new(VectorStore::create(&db_path, dense_schema(dim)).unwrap());

    let num_threads = 4;
    let ops_per_thread = 50;

    let mut handles = vec![];
    for t in 0..num_threads {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..ops_per_thread {
                let id = format!("vec_{}_{}", t, i);
                let vec = random_vector(dim as usize, (t * 1000 + i) as usize);
                store
                    .set(&id, vec, serde_json::json!({"op": "set"}))
                    .unwrap();

                if i % 5 == 0 {
                    let _ = store.delete(&id);
                }

                if i % 10 == 0 {
                    let _ = store.checkpoint_wal();
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Some were deleted, some remain. Just ensure it didn't crash/deadlock.
    assert!(store.len() > 0);
}
