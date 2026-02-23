//! OmenDB Node.js/Bun bindings via napi-rs
//!
//! Fast embedded vector database with HNSW indexing.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]

mod collections;
mod conversions;
mod database;
mod edges;
mod filters;
mod hybrid;
mod open;
mod search;
mod sparse;
mod types;
