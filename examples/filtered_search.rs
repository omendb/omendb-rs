//! Filtered search with ACORN-1 algorithm
//!
//! Demonstrates metadata filtering during vector search.
//!
//! Run with: cargo run --example filtered_search

use omendb_core::{MetadataFilter, Vector, VectorStore};
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let mut store = VectorStore::new(64);

    // Insert papers with metadata
    let papers = vec![
        (
            "hnsw",
            vec![0.1; 64],
            json!({"year": 2018, "venue": "PAMI", "citations": 1500}),
        ),
        (
            "rabitq",
            vec![0.2; 64],
            json!({"year": 2024, "venue": "SIGMOD", "citations": 50}),
        ),
        (
            "diskann",
            vec![0.3; 64],
            json!({"year": 2019, "venue": "NeurIPS", "citations": 800}),
        ),
        (
            "acorn",
            vec![0.4; 64],
            json!({"year": 2023, "venue": "VLDB", "citations": 120}),
        ),
        (
            "faiss",
            vec![0.5; 64],
            json!({"year": 2017, "venue": "arXiv", "citations": 2000}),
        ),
    ];

    for (id, vec, meta) in papers {
        store.set(id.into(), Vector::new(vec), meta)?;
    }

    let query = Vector::new(vec![0.3; 64]);

    // Filter: year >= 2020
    let filter = MetadataFilter::Gte("year".into(), 2020.0);
    let results = store.knn_search_with_filter(&query, 10, &filter)?;
    println!(
        "year >= 2020: {:?}",
        results.iter().map(|(i, _, _)| *i).collect::<Vec<_>>()
    );

    // Filter: venue in ["VLDB", "SIGMOD"]
    let filter = MetadataFilter::In("venue".into(), vec![json!("VLDB"), json!("SIGMOD")]);
    let results = store.knn_search_with_filter(&query, 10, &filter)?;
    println!(
        "venue in [VLDB, SIGMOD]: {:?}",
        results.iter().map(|(i, _, _)| *i).collect::<Vec<_>>()
    );

    // Filter: citations > 500
    let filter = MetadataFilter::Gt("citations".into(), 500.0);
    let results = store.knn_search_with_filter(&query, 10, &filter)?;
    println!(
        "citations > 500: {:?}",
        results.iter().map(|(i, _, _)| *i).collect::<Vec<_>>()
    );

    // Combined: year >= 2018 AND citations > 100
    let filter = MetadataFilter::And(vec![
        MetadataFilter::Gte("year".into(), 2018.0),
        MetadataFilter::Gt("citations".into(), 100.0),
    ]);
    let results = store.knn_search_with_filter(&query, 10, &filter)?;
    println!(
        "year >= 2018 AND citations > 100: {:?}",
        results.iter().map(|(i, _, _)| *i).collect::<Vec<_>>()
    );

    Ok(())
}
