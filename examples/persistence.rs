//! Persistence example - saving and loading from disk
//!
//! Demonstrates how to persist vectors to disk and reload them.
//!
//! Run with: cargo run --example persistence

use omendb::{Vector, VectorStore};
use serde_json::json;
use tempfile::TempDir;

fn main() -> anyhow::Result<()> {
    // Use a temp directory for this example
    let tmp = TempDir::new()?;
    let db_path = tmp.path().join("vectors");

    // Create and populate store
    {
        let mut store = VectorStore::open_with_dimensions(&db_path, 64)?;

        store.set(
            "a".into(),
            Vector::new(vec![1.0; 64]),
            json!({"name": "first"}),
        )?;
        store.set(
            "b".into(),
            Vector::new(vec![2.0; 64]),
            json!({"name": "second"}),
        )?;
        store.set(
            "c".into(),
            Vector::new(vec![3.0; 64]),
            json!({"name": "third"}),
        )?;

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
