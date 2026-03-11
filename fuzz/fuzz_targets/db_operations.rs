#![no_main]

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use omendb::{Vector, VectorStoreOptions};
use serde_json::json;
use tempfile::TempDir;

const DIMENSIONS: usize = 32;

#[derive(Debug, Clone)]
enum VectorOp {
    Insert { id: String, vector: Vec<f32> },
    Search { vector: Vec<f32>, k: usize },
    Get { id: String },
    Delete { id: String },
    Flush,
    Count,
}

impl<'a> Arbitrary<'a> for VectorOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op_type: u8 = u.int_in_range(0..=5)?;

        match op_type {
            0 => {
                let id_len: usize = u.int_in_range(1..=32)?;
                let id: String = (0..id_len)
                    .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
                    .collect();
                let vector: Vec<f32> = (0..DIMENSIONS)
                    .map(|_| u.arbitrary::<f32>().unwrap_or(0.0))
                    .collect();
                Ok(VectorOp::Insert { id, vector })
            }
            1 => {
                let vector: Vec<f32> = (0..DIMENSIONS)
                    .map(|_| u.arbitrary::<f32>().unwrap_or(0.0))
                    .collect();
                let k: usize = u.int_in_range(1..=100)?;
                Ok(VectorOp::Search { vector, k })
            }
            2 => {
                let id_len: usize = u.int_in_range(1..=32)?;
                let id: String = (0..id_len)
                    .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
                    .collect();
                Ok(VectorOp::Get { id })
            }
            3 => {
                let id_len: usize = u.int_in_range(1..=32)?;
                let id: String = (0..id_len)
                    .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
                    .collect();
                Ok(VectorOp::Delete { id })
            }
            4 => Ok(VectorOp::Flush),
            5 => Ok(VectorOp::Count),
            _ => unreachable!(),
        }
    }
}

fn sanitize_vector(v: &mut [f32]) {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);

    let ops: Vec<VectorOp> = match u.arbitrary() {
        Ok(ops) => ops,
        Err(_) => return,
    };

    if ops.is_empty() {
        return;
    }

    let temp_dir = match TempDir::new() {
        Ok(dir) => dir,
        Err(_) => return,
    };

    let mut store = match VectorStoreOptions::default()
        .dimensions(DIMENSIONS)
        .open(temp_dir.path())
    {
        Ok(s) => s,
        Err(_) => return,
    };

    for op in ops {
        match op {
            VectorOp::Insert { id, mut vector } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector);
                let _ = store.set(&id, v, json!({}));
            }
            VectorOp::Search { mut vector, k } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector);
                let _ = store.search(&v, k.min(100), None);
            }
            VectorOp::Get { id } => {
                let _ = store.get(&id);
            }
            VectorOp::Delete { id } => {
                let _ = store.delete(&id);
            }
            VectorOp::Flush => {
                let _ = store.flush();
            }
            VectorOp::Count => {
                let _ = store.len();
            }
        }
    }
});
