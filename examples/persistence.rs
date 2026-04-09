//! Persistence example - saving and loading from disk
//!
//! Demonstrates how to persist vectors to disk and reload them.
//!
//! Run with: cargo run --example persistence

use omendb::catalog::{
    CollectionSchema, DenseSchema, FrozenDenseIndexKind, MutableDenseIndexKind, QuantizationMode,
};
use omendb::{Metric, Vector, VectorStore};
use serde_json::json;
use tempfile::TempDir;

fn main() -> anyhow::Result<()> {
    // Use a temp directory for this example
    let tmp = TempDir::new()?;
    let db_path = tmp.path().join("vectors");

    // Create and populate store
    {
        let store = VectorStore::create(
            &db_path,
            CollectionSchema {
                name: String::new(),
                metric: Metric::L2,
                dense: Some(DenseSchema {
                    dim: 64,
                    quantization: QuantizationMode::None,
                    mutable_index: MutableDenseIndexKind::Hnsw,
                    frozen_index: FrozenDenseIndexKind::Hnsw,
                }),
                sparse: None,
                multi: None,
                text: None,
            },
        )?;

        store.set("a", Vector::new(vec![1.0; 64]), json!({"name": "first"}))?;
        store.set("b", Vector::new(vec![2.0; 64]), json!({"name": "second"}))?;
        store.set("c", Vector::new(vec![3.0; 64]), json!({"name": "third"}))?;

        store.flush()?;
        println!("Saved {} vectors to {:?}", store.len(), db_path);
    }

    // Reopen and verify
    {
        let store = VectorStore::open(&db_path)?;
        println!("Reopened: {} vectors", store.len());

        if let Some((vec, meta)) = store.get("a") {
            println!("Found 'a': {} dims, name={}", vec.data.len(), meta["name"]);
        }
    }

    Ok(())
}
