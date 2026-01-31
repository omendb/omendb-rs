//! Input types for unified VectorStore API.
//!
//! Provides traits for type-based dispatch so `set()` and `search()` work
//! with both single vectors and multi-vector token embeddings.

use super::MetadataFilter;
use serde_json::Value as JsonValue;

// === Search Options ===

/// Reranking mode for multi-vector search.
#[derive(Debug, Clone, Default)]
pub enum Rerank {
    /// No reranking - fast approximate search.
    Off,
    /// Rerank with default factor (32x candidates).
    #[default]
    On,
    /// Rerank with custom factor.
    Factor(usize),
}

/// Options for search operations.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// HNSW ef parameter override (default: 2*k).
    pub ef: Option<usize>,
    /// Metadata filter.
    pub filter: Option<MetadataFilter>,
    /// Maximum distance threshold.
    pub max_distance: Option<f32>,
    /// Reranking for multi-vector (ignored for regular stores).
    pub rerank: Rerank,
}

/// Alias for `SearchOptions` - parameters for vector search.
pub type SearchParams = SearchOptions;

impl SearchOptions {
    /// Create a new SearchOptions with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with filter only.
    #[must_use]
    pub fn with_filter(filter: MetadataFilter) -> Self {
        Self {
            filter: Some(filter),
            ..Default::default()
        }
    }

    /// Create options with ef override.
    #[must_use]
    pub fn with_ef(ef: usize) -> Self {
        Self {
            ef: Some(ef),
            ..Default::default()
        }
    }

    /// Builder: set filter.
    #[must_use]
    pub fn filter(mut self, f: MetadataFilter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Builder: set ef.
    #[must_use]
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Builder: set max distance.
    #[must_use]
    pub fn max_distance(mut self, d: f32) -> Self {
        self.max_distance = Some(d);
        self
    }

    /// Builder: set rerank mode.
    #[must_use]
    pub fn rerank(mut self, r: Rerank) -> Self {
        self.rerank = r;
        self
    }

    /// Builder: disable reranking.
    #[must_use]
    pub fn no_rerank(mut self) -> Self {
        self.rerank = Rerank::Off;
        self
    }
}

/// Parameters for hybrid search (vector + text).
#[derive(Debug, Clone)]
pub struct HybridParams {
    /// Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5).
    pub alpha: f32,
    /// RRF constant (default=60, higher reduces rank influence).
    pub rrf_k: usize,
    /// Optional metadata filter.
    pub filter: Option<MetadataFilter>,
    /// Return separate keyword_score and semantic_score.
    pub subscores: bool,
    /// HNSW ef parameter override.
    pub ef: Option<usize>,
}

impl Default for HybridParams {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            rrf_k: 60,
            filter: None,
            subscores: false,
            ef: None,
        }
    }
}

impl HybridParams {
    /// Create with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set alpha.
    #[must_use]
    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    /// Builder: set rrf_k.
    #[must_use]
    pub fn rrf_k(mut self, k: usize) -> Self {
        self.rrf_k = k;
        self
    }

    /// Builder: set filter.
    #[must_use]
    pub fn filter(mut self, f: MetadataFilter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Builder: enable subscores.
    #[must_use]
    pub fn subscores(mut self) -> Self {
        self.subscores = true;
        self
    }

    /// Builder: set ef.
    #[must_use]
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }
}

/// Internal representation of stored data.
#[derive(Debug, Clone)]
pub enum VectorData {
    /// Single vector (regular store).
    Single(Vec<f32>),
    /// Multi-vector tokens (multi-vector store).
    Multi(Vec<Vec<f32>>),
}

impl VectorData {
    /// Dimension of single vector or token vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        match self {
            Self::Single(v) => v.len(),
            Self::Multi(tokens) => tokens.first().map_or(0, Vec::len),
        }
    }

    /// Number of items (1 for single, token count for multi).
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Multi(tokens) => tokens.len(),
        }
    }

    /// Is this single vector data?
    #[must_use]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single(_))
    }

    /// Is this multi-vector data?
    #[must_use]
    pub fn is_multi(&self) -> bool {
        matches!(self, Self::Multi(_))
    }

    /// Get as single vector, if applicable.
    #[must_use]
    pub fn as_single(&self) -> Option<&[f32]> {
        match self {
            Self::Single(v) => Some(v),
            Self::Multi(_) => None,
        }
    }

    /// Get as token embeddings, if applicable.
    #[must_use]
    pub fn as_multi(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::Single(_) => None,
            Self::Multi(tokens) => Some(tokens),
        }
    }
}

/// Data that can be stored in a VectorStore.
///
/// Implemented for single vectors (`&[f32]`, `Vec<f32>`) and
/// multi-vectors (`&[&[f32]]`, `Vec<Vec<f32>>`, etc.).
pub trait VectorInput {
    /// Convert to internal representation.
    fn into_vector_data(self) -> VectorData;
}

// === Single vector implementations ===

impl VectorInput for Vec<f32> {
    fn into_vector_data(self) -> VectorData {
        VectorData::Single(self)
    }
}

impl VectorInput for &[f32] {
    fn into_vector_data(self) -> VectorData {
        VectorData::Single(self.to_vec())
    }
}

impl VectorInput for &Vec<f32> {
    fn into_vector_data(self) -> VectorData {
        VectorData::Single(self.clone())
    }
}

// === Multi-vector implementations ===

impl VectorInput for Vec<Vec<f32>> {
    fn into_vector_data(self) -> VectorData {
        VectorData::Multi(self)
    }
}

impl VectorInput for &Vec<Vec<f32>> {
    fn into_vector_data(self) -> VectorData {
        VectorData::Multi(self.clone())
    }
}

impl VectorInput for &[Vec<f32>] {
    fn into_vector_data(self) -> VectorData {
        VectorData::Multi(self.to_vec())
    }
}

impl VectorInput for &[&[f32]] {
    fn into_vector_data(self) -> VectorData {
        VectorData::Multi(self.iter().map(|t| t.to_vec()).collect())
    }
}

impl VectorInput for Vec<&[f32]> {
    fn into_vector_data(self) -> VectorData {
        VectorData::Multi(self.iter().map(|t| t.to_vec()).collect())
    }
}

// === Query data (borrowed, for search) ===

/// Query representation for search operations.
#[derive(Debug, Clone)]
pub enum QueryData<'a> {
    /// Single vector query.
    Single(&'a [f32]),
    /// Multi-vector query tokens.
    Multi(Vec<&'a [f32]>),
}

impl QueryData<'_> {
    /// Dimension of query vector(s).
    #[must_use]
    pub fn dim(&self) -> usize {
        match self {
            Self::Single(v) => v.len(),
            Self::Multi(tokens) => tokens.first().map_or(0, |t| t.len()),
        }
    }

    /// Is this a single vector query?
    #[must_use]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single(_))
    }

    /// Is this a multi-vector query?
    #[must_use]
    pub fn is_multi(&self) -> bool {
        matches!(self, Self::Multi(_))
    }
}

/// Data that can be used as a search query.
///
/// Implemented for single vectors and multi-vector token sets.
pub trait QueryInput {
    /// Convert to query data (borrows self).
    fn to_query_data(&self) -> QueryData<'_>;
}

// === Single query implementations ===

impl QueryInput for [f32] {
    fn to_query_data(&self) -> QueryData<'_> {
        QueryData::Single(self)
    }
}

impl QueryInput for Vec<f32> {
    fn to_query_data(&self) -> QueryData<'_> {
        QueryData::Single(self.as_slice())
    }
}

// === Multi query implementations ===

impl QueryInput for [Vec<f32>] {
    fn to_query_data(&self) -> QueryData<'_> {
        QueryData::Multi(self.iter().map(Vec::as_slice).collect())
    }
}

impl QueryInput for Vec<Vec<f32>> {
    fn to_query_data(&self) -> QueryData<'_> {
        QueryData::Multi(self.iter().map(Vec::as_slice).collect())
    }
}

impl<'a> QueryInput for [&'a [f32]] {
    fn to_query_data(&self) -> QueryData<'a> {
        QueryData::Multi(self.to_vec())
    }
}

impl<'a> QueryInput for Vec<&'a [f32]> {
    fn to_query_data(&self) -> QueryData<'a> {
        QueryData::Multi(self.clone())
    }
}

// === Batch input ===

/// Single item in a batch operation.
pub trait BatchItem {
    /// Extract (id, data, metadata) components.
    fn into_parts(self) -> (String, VectorData, JsonValue);
}

// Tuple implementations for batch - single vectors
impl BatchItem for (String, Vec<f32>, JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        (self.0, VectorData::Single(self.1), self.2)
    }
}

impl BatchItem for (&str, Vec<f32>, JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        (self.0.to_string(), VectorData::Single(self.1), self.2)
    }
}

impl BatchItem for (&str, &[f32], JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        (
            self.0.to_string(),
            VectorData::Single(self.1.to_vec()),
            self.2,
        )
    }
}

// Tuple implementations for batch - multi-vectors
impl BatchItem for (String, Vec<Vec<f32>>, JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        (self.0, VectorData::Multi(self.1), self.2)
    }
}

impl BatchItem for (&str, Vec<Vec<f32>>, JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        (self.0.to_string(), VectorData::Multi(self.1), self.2)
    }
}

impl<'a> BatchItem for (&str, &'a [&'a [f32]], JsonValue) {
    fn into_parts(self) -> (String, VectorData, JsonValue) {
        let tokens = self.1.iter().map(|t| t.to_vec()).collect();
        (self.0.to_string(), VectorData::Multi(tokens), self.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_data_single() {
        let data = VectorData::Single(vec![1.0, 2.0, 3.0]);
        assert!(data.is_single());
        assert!(!data.is_multi());
        assert_eq!(data.dim(), 3);
        assert_eq!(data.count(), 1);
    }

    #[test]
    fn test_vector_data_multi() {
        let data = VectorData::Multi(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert!(!data.is_single());
        assert!(data.is_multi());
        assert_eq!(data.dim(), 2);
        assert_eq!(data.count(), 3);
    }

    #[test]
    fn test_vector_input_vec_f32() {
        let v = vec![1.0, 2.0, 3.0];
        let data = v.into_vector_data();
        assert!(data.is_single());
    }

    #[test]
    fn test_vector_input_slice_f32() {
        let v = [1.0, 2.0, 3.0];
        let data = v.as_slice().into_vector_data();
        assert!(data.is_single());
    }

    #[test]
    fn test_vector_input_vec_vec_f32() {
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let data = v.into_vector_data();
        assert!(data.is_multi());
    }

    #[test]
    fn test_vector_input_slice_slice() {
        let t1 = [1.0, 2.0];
        let t2 = [3.0, 4.0];
        let tokens: &[&[f32]] = &[&t1, &t2];
        let data = tokens.into_vector_data();
        assert!(data.is_multi());
    }

    #[test]
    fn test_query_input_single() {
        let v = vec![1.0, 2.0, 3.0];
        let query = v.to_query_data();
        assert!(query.is_single());
        assert_eq!(query.dim(), 3);
    }

    #[test]
    fn test_query_input_multi() {
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let query = v.to_query_data();
        assert!(query.is_multi());
        assert_eq!(query.dim(), 2);
    }

    #[test]
    fn test_batch_item_single() {
        let item = ("id", vec![1.0, 2.0], serde_json::json!({}));
        let (id, data, _) = item.into_parts();
        assert_eq!(id, "id");
        assert!(data.is_single());
    }

    #[test]
    fn test_batch_item_multi() {
        let item = (
            "id",
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            serde_json::json!({}),
        );
        let (id, data, _) = item.into_parts();
        assert_eq!(id, "id");
        assert!(data.is_multi());
    }
}
