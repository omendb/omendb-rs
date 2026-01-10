//! Basic OmenDB usage example
//!
//! Demonstrates core operations: insert, search, get, delete.
//!
//! Run with: cargo run --example basic

use omendb::{Vector, VectorStore};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    // Create an in-memory store (128-dimensional vectors)
    let mut store = VectorStore::new(128);

    // Insert vectors with metadata
    let v1 = Vector::new(vec![1.0; 128]);
    let v2 = Vector::new(vec![0.9; 128]);
    let v3 = Vector::new(vec![0.0; 128]);

    store.set("doc1".into(), v1, json!({"type": "article", "year": 2024}))?;
    store.set("doc2".into(), v2, json!({"type": "article", "year": 2023}))?;
    store.set("doc3".into(), v3, json!({"type": "note", "year": 2024}))?;

    println!("Inserted {} vectors", store.len());

    // Search for similar vectors
    let query = Vector::new(vec![1.0; 128]);
    let results = store.knn_search(&query, 2)?;

    println!("\nTop 2 similar vectors:");
    for (idx, distance) in &results {
        println!("  idx={}, distance={:.4}", idx, distance);
    }

    // Get by ID
    if let Some((vec, metadata)) = store.get("doc1") {
        println!(
            "\nGet 'doc1': {} dims, metadata={}",
            vec.data.len(),
            metadata
        );
    }

    // Delete
    store.delete("doc3")?;
    println!("\nAfter delete: {} vectors", store.len());

    Ok(())
}
