// HNSW Storage Module
//
// This module contains all storage implementations for the HNSW index:
// - NeighborLists: Lock-free neighbor storage with ArcSwap (legacy)
// - VectorStorage: Full precision and SQ8 quantized vector storage
// - Level0Storage: Atomic slot storage for level 0 neighbors
// - UpperLevelStorage: Sparse storage for upper level neighbors
// - NeighborStorage: Unified facade over Level0 + Upper levels

mod level0;
mod neighbor_lists;
mod neighbor_storage;
mod upper_levels;
mod vector_storage;

// Re-export main types (NeighborStorage uses Level0/Upper internally)
pub use neighbor_lists::NeighborLists;
pub use neighbor_storage::NeighborStorage;
pub use vector_storage::VectorStorage;
