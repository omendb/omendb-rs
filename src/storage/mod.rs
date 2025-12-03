//! Storage modules for vector data, node metadata, and graph edges.

pub mod edge_storage;
pub mod node_metadata;
pub mod vector_metadata;

pub use edge_storage::EdgeStorage;
pub use node_metadata::NodeMetadataStorage;
pub use vector_metadata::VectorMetadataStorage;
