//! Query request contracts.

pub mod request;

pub use request::{
    DenseSearchRequest, HybridSearchRequest, MultiSearchRequest, SearchRequest, SparseSearchRequest,
    TextSearchRequest,
};
