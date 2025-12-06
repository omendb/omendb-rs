#![no_main]

use libfuzzer_sys::arbitrary::{self, Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use omendb::text::TextIndex;
use omendb::{Vector, VectorStoreOptions};
use serde_json::json;

const DIMENSIONS: usize = 32;

#[derive(Debug, Clone)]
enum TextOp {
    IndexDocument {
        id: String,
        text: String,
    },
    Search {
        query: String,
        k: usize,
    },
    Delete {
        id: String,
    },
    Commit,
    HybridSearch {
        vector: Vec<f32>,
        text: String,
        k: usize,
    },
}

impl<'a> Arbitrary<'a> for TextOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let op_type: u8 = u.int_in_range(0..=4)?;

        match op_type {
            0 => {
                let id_len: usize = u.int_in_range(1..=32)?;
                let id: String = (0..id_len)
                    .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
                    .collect();
                let text_len: usize = u.int_in_range(1..=200)?;
                let text: String = (0..text_len)
                    .map(|_| {
                        let c = u.int_in_range(b' '..=b'~').unwrap_or(b' ');
                        c as char
                    })
                    .collect();
                Ok(TextOp::IndexDocument { id, text })
            }
            1 => {
                let query_len: usize = u.int_in_range(0..=50)?;
                let query: String = (0..query_len)
                    .map(|_| {
                        let c = u.int_in_range(b' '..=b'~').unwrap_or(b' ');
                        c as char
                    })
                    .collect();
                let k: usize = u.int_in_range(1..=100)?;
                Ok(TextOp::Search { query, k })
            }
            2 => {
                let id_len: usize = u.int_in_range(1..=32)?;
                let id: String = (0..id_len)
                    .map(|_| u.int_in_range(b'a'..=b'z').unwrap_or(b'a') as char)
                    .collect();
                Ok(TextOp::Delete { id })
            }
            3 => Ok(TextOp::Commit),
            4 => {
                let vector: Vec<f32> = (0..DIMENSIONS)
                    .map(|_| u.arbitrary::<f32>().unwrap_or(0.0))
                    .collect();
                let text_len: usize = u.int_in_range(0..=50)?;
                let text: String = (0..text_len)
                    .map(|_| {
                        let c = u.int_in_range(b' '..=b'~').unwrap_or(b' ');
                        c as char
                    })
                    .collect();
                let k: usize = u.int_in_range(1..=100)?;
                Ok(TextOp::HybridSearch { vector, text, k })
            }
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

    let ops: Vec<TextOp> = match u.arbitrary() {
        Ok(ops) => ops,
        Err(_) => return,
    };

    if ops.is_empty() {
        return;
    }

    // Test standalone TextIndex
    let mut text_index = match TextIndex::open_in_memory() {
        Ok(idx) => idx,
        Err(_) => return,
    };

    // Test hybrid search with VectorStore
    let mut store = match VectorStoreOptions::default()
        .dimensions(DIMENSIONS)
        .text_search(true)
        .build()
    {
        Ok(s) => s,
        Err(_) => return,
    };

    for op in ops {
        match op {
            TextOp::IndexDocument { id, text } => {
                let _ = text_index.index_document(&id, &text);
                // Also add to vector store for hybrid search
                let mut vector: Vec<f32> = (0..DIMENSIONS).map(|i| (i as f32) * 0.1).collect();
                sanitize_vector(&mut vector);
                let _ = store.set_with_text(id, Vector::new(vector), &text, json!({}));
            }
            TextOp::Search { query, k } => {
                let _ = text_index.search(&query, k.min(100));
            }
            TextOp::Delete { id } => {
                let _ = text_index.delete_document(&id);
            }
            TextOp::Commit => {
                let _ = text_index.commit();
                let _ = store.flush();
            }
            TextOp::HybridSearch {
                mut vector,
                text,
                k,
            } => {
                sanitize_vector(&mut vector);
                let v = Vector::new(vector);
                let _ = store.hybrid_search(&v, &text, k.min(100), None);
            }
        }
    }
});
