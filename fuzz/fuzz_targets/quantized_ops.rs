#![no_main]

//! Fuzz target for quantized VectorStore operations.
//!
//! This target specifically tests SQ8 quantization mode,
//! which had 10 bugs fixed in v0.0.12. High-risk area for regressions.

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use omendb::{Vector, VectorStoreOptions};
use serde_json::json;
use tempfile::TempDir;

const DIMENSIONS: usize = 32;

#[derive(Debug, Clone)]
enum QuantizedOp {
    Insert {
        id: String,
        vector: Vec<f32>,
    },
    Search {
        vector: Vec<f32>,
        k: usize,
    },
    GetById {
        id: String,
    },
    Items,
    Count,
    Flush,
    // Specifically test problematic patterns from v0.0.12 bugs
    InsertThenGetById {
        id: String,
        vector: Vec<f32>,
    },
    InsertThenSearch {
        id: String,
        vector: Vec<f32>,
        k: usize,
    },
}

impl<'a> Arbitrary<'a> for QuantizedOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op_type: u8 = u.int_in_range(0..=7)?;

        match op_type {
            0 => {
                let id = generate_id(u)?;
                let vector = generate_vector(u)?;
                Ok(QuantizedOp::Insert { id, vector })
            }
            1 => {
                let vector = generate_vector(u)?;
                let k: usize = u.int_in_range(1..=50)?;
                Ok(QuantizedOp::Search { vector, k })
            }
            2 => {
                let id = generate_id(u)?;
                Ok(QuantizedOp::GetById { id })
            }
            3 => Ok(QuantizedOp::Items),
            4 => Ok(QuantizedOp::Count),
            5 => Ok(QuantizedOp::Flush),
            6 => {
                // InsertThenGetById - tests ID mapping consistency (bug area)
                let id = generate_id(u)?;
                let vector = generate_vector(u)?;
                Ok(QuantizedOp::InsertThenGetById { id, vector })
            }
            7 => {
                // InsertThenSearch - tests search returns inserted vectors (bug area)
                let id = generate_id(u)?;
                let vector = generate_vector(u)?;
                let k: usize = u.int_in_range(1..=10)?;
                Ok(QuantizedOp::InsertThenSearch { id, vector, k })
            }
            _ => unreachable!(),
        }
    }
}

fn generate_id(u: &mut Unstructured<'_>) -> arbitrary::Result<String> {
    let id_len: usize = u.int_in_range(1..=32)?;
    let id: String = (0..id_len)
        .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
        .collect();
    Ok(id)
}

fn generate_vector(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<f32>> {
    let vector: Vec<f32> = (0..DIMENSIONS)
        .map(|_| u.arbitrary::<f32>().unwrap_or(0.0))
        .collect();
    Ok(vector)
}

fn sanitize_vector(v: &mut [f32]) {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

fn run_with_sq8(ops: Vec<QuantizedOp>) {
    let temp_dir = match TempDir::new() {
        Ok(dir) => dir,
        Err(_) => return,
    };

    let mut store = match VectorStoreOptions::default()
        .dimensions(DIMENSIONS)
        .quantization(true)
        .open(temp_dir.path())
    {
        Ok(s) => s,
        Err(_) => return,
    };

    for op in ops {
        match op {
            QuantizedOp::Insert { id, mut vector } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector);
                let _ = store.set(&id, v, json!({"test": true}));
            }
            QuantizedOp::Search { mut vector, k } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector);
                let _ = store.search(&v, k.min(100), None);
            }
            QuantizedOp::GetById { id } => {
                // Should not panic, may return None
                let _ = store.get(&id);
            }
            QuantizedOp::Items => {
                // Should not panic, should return valid Vec
                let items = store.items();
                // Items count should match store.len()
                assert_eq!(items.len(), store.len());
            }
            QuantizedOp::Count => {
                let _ = store.len();
            }
            QuantizedOp::Flush => {
                let _ = store.flush();
            }
            QuantizedOp::InsertThenGetById { id, mut vector } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector.clone());
                if store.set(&id, v, json!({})).is_ok() {
                    // After successful insert, get() MUST return Some
                    let result = store.get(&id);
                    assert!(
                        result.is_some(),
                        "get() returned None after successful insert for id: {}",
                        id
                    );
                }
            }
            QuantizedOp::InsertThenSearch { id, mut vector, k } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector.clone());
                if store.set(&id, v.clone(), json!({})).is_ok() {
                    // Search for the exact vector should return it
                    if let Ok(results) = store.search(&v, k.min(100), None) {
                        // The inserted vector should be in top-k results
                        // (unless k is very small and there are many vectors)
                        // At minimum, results should not be empty if store is not empty
                        if store.len() > 0 {
                            assert!(
                                !results.is_empty() || k == 0,
                                "Search returned empty results on non-empty store"
                            );
                        }
                    }
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);

    let ops: Vec<QuantizedOp> = match u.arbitrary() {
        Ok(ops) => ops,
        Err(_) => return,
    };

    if ops.is_empty() {
        return;
    }

    // Test with SQ8 quantization
    run_with_sq8(ops);
});
