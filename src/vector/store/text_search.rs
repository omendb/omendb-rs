//! Text and hybrid search operations for VectorStore.
//!
//! This module contains methods for BM25 text search and hybrid search
//! that combines vector similarity with text relevance using RRF fusion.

use super::helpers;
use super::record_store::RecordStore;
use super::{MetadataFilter, VectorStore};
use crate::text::{
    weighted_reciprocal_rank_fusion, weighted_reciprocal_rank_fusion_with_subscores, HybridResult,
    TextIndex, TextSearchConfig, DEFAULT_RRF_K,
};
use crate::vector::types::Vector;
use anyhow::Result;
use serde_json::Value as JsonValue;

impl VectorStore {
    // ========================================================================
    // Text Search Setup
    // ========================================================================

    /// Enable text search on this store.
    ///
    /// Creates a text index for BM25 keyword search. Must be called before
    /// using `set_with_text()`, `text_search()`, or `hybrid_search()`.
    pub fn enable_text_search(&mut self) -> Result<()> {
        self.enable_text_search_with_config(None)
    }

    /// Enable text search with custom configuration.
    ///
    /// # Arguments
    /// * `config` - Optional text search configuration (language, stopwords, etc.)
    pub fn enable_text_search_with_config(
        &mut self,
        config: Option<TextSearchConfig>,
    ) -> Result<()> {
        if self.text_index.is_some() {
            return Ok(());
        }

        let config = config
            .or_else(|| self.text_search_config.clone())
            .unwrap_or_default();

        self.text_index = if let Some(ref path) = self.storage_path {
            let text_path = path.join("text_index");
            Some(TextIndex::open_with_config(&text_path, &config)?)
        } else {
            Some(TextIndex::open_in_memory_with_config(&config)?)
        };

        Ok(())
    }

    /// Check if text search is enabled.
    #[must_use]
    pub fn has_text_search(&self) -> bool {
        self.text_index.is_some()
    }

    // ========================================================================
    // Insert with Text
    // ========================================================================

    /// Upsert vector with text content for hybrid search.
    ///
    /// Indexes the text for BM25 search and stores the vector for similarity search.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this document
    /// * `vector` - Embedding vector
    /// * `text` - Text content to index for keyword search
    /// * `metadata` - Optional metadata
    pub fn set_with_text(
        &mut self,
        id: String,
        vector: Vector,
        text: &str,
        metadata: JsonValue,
    ) -> Result<usize> {
        let Some(ref mut text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        text_index.index_document(&id, text)?;
        self.set(id, vector, metadata)
    }

    /// Batch upsert vectors with text content for hybrid search.
    ///
    /// # Arguments
    /// * `batch` - Vector of (id, vector, text, metadata) tuples
    pub fn set_batch_with_text(
        &mut self,
        batch: Vec<(String, Vector, String, JsonValue)>,
    ) -> Result<Vec<usize>> {
        let Some(ref mut text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        for (id, _, text, _) in &batch {
            text_index.index_document(id, text)?;
        }

        let vector_batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, _, metadata)| (id, vector, metadata))
            .collect();

        self.set_batch(vector_batch)
    }

    // ========================================================================
    // Text-Only Search
    // ========================================================================

    /// Search text index only (BM25 scoring).
    ///
    /// Returns documents ranked by keyword relevance without considering
    /// vector similarity.
    ///
    /// # Arguments
    /// * `query` - Text query
    /// * `k` - Number of results to return
    pub fn text_search(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        let Some(ref text_index) = self.text_index else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        text_index.search(query, k)
    }

    // ========================================================================
    // Hybrid Search (Vector + Text)
    // ========================================================================

    /// Hybrid search combining vector similarity and BM25 text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine results from vector
    /// and text search with configurable weighting.
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding
    /// * `query_text` - Text query for BM25
    /// * `k` - Number of results to return
    /// * `alpha` - Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5)
    pub fn hybrid_search(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_rrf_k(query_vector, query_text, k, alpha, None)
    }

    /// Hybrid search with configurable RRF k constant.
    ///
    /// # Arguments
    /// * `rrf_k` - RRF constant (default=60, higher reduces rank influence)
    pub fn hybrid_search_with_rrf_k(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.validate_hybrid_search_preconditions(query_vector)?;

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results = self.convert_knn_results_to_id_scores(vector_results);

        let text_results = self.text_search(query_text, fetch_k)?;

        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata(&self.records, fused))
    }

    /// Hybrid search with metadata filter.
    pub fn hybrid_search_with_filter(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.hybrid_search_with_filter_rrf_k(query_vector, query_text, k, filter, alpha, None)
    }

    /// Hybrid search with filter and configurable RRF k constant.
    pub fn hybrid_search_with_filter_rrf_k(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.validate_hybrid_search_preconditions(query_vector)?;

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
        let text_results = filter_text_results_by_metadata(&self.records, text_results, filter);

        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata(&self.records, fused))
    }

    // ========================================================================
    // Hybrid Search with Subscores
    // ========================================================================

    /// Hybrid search returning separate keyword and semantic scores.
    ///
    /// Returns [`HybridResult`] with `keyword_score` (BM25) and `semantic_score`
    /// (vector distance) for each result, enabling custom post-processing.
    pub fn hybrid_search_with_subscores(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
        self.validate_hybrid_search_preconditions(query_vector)?;

        let fetch_k = k * 2;

        let vector_results = self.knn_search(query_vector, fetch_k)?;
        let vector_results = self.convert_knn_results_to_id_scores(vector_results);

        let text_results = self.text_search(query_text, fetch_k)?;

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata_to_hybrid_results(&self.records, fused))
    }

    /// Hybrid search with filter returning separate keyword and semantic scores.
    pub fn hybrid_search_with_filter_subscores(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: &MetadataFilter,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
        self.validate_hybrid_search_preconditions(query_vector)?;

        let fetch_k = k * 4;

        let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
        let vector_results: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect();

        let text_results = self.text_search(query_text, fetch_k)?;
        let text_results = filter_text_results_by_metadata(&self.records, text_results, filter);

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata_to_hybrid_results(&self.records, fused))
    }

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Validate preconditions for hybrid search.
    fn validate_hybrid_search_preconditions(&self, query_vector: &Vector) -> Result<()> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.is_none() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }
        Ok(())
    }

    /// Convert KNN results (index, distance) to (id, distance).
    fn convert_knn_results_to_id_scores(&self, results: Vec<(usize, f32)>) -> Vec<(String, f32)> {
        results
            .into_iter()
            .filter_map(|(idx, distance)| {
                self.records
                    .get_id(idx as u32)
                    .map(|id| (id.to_string(), distance))
            })
            .collect()
    }
}

// ============================================================================
// Standalone Helper Functions
// ============================================================================

/// Attach metadata to fused results.
fn attach_metadata(
    records: &RecordStore,
    results: Vec<(String, f32)>,
) -> Vec<(String, f32, JsonValue)> {
    results
        .into_iter()
        .map(|(id, score)| {
            let metadata = records
                .get(&id)
                .and_then(|r| r.metadata.clone())
                .unwrap_or_else(helpers::default_metadata);
            (id, score, metadata)
        })
        .collect()
}

/// Attach metadata to hybrid results with subscores.
fn attach_metadata_to_hybrid_results(
    records: &RecordStore,
    results: Vec<HybridResult>,
) -> Vec<(HybridResult, JsonValue)> {
    results
        .into_iter()
        .map(|result| {
            let metadata = records
                .get(&result.id)
                .and_then(|r| r.metadata.clone())
                .unwrap_or_else(helpers::default_metadata);
            (result, metadata)
        })
        .collect()
}

/// Filter text results by metadata filter.
fn filter_text_results_by_metadata(
    records: &RecordStore,
    results: Vec<(String, f32)>,
    filter: &MetadataFilter,
) -> Vec<(String, f32)> {
    results
        .into_iter()
        .filter(|(id, _)| {
            records
                .get(id)
                .and_then(|r| r.metadata.as_ref())
                .is_some_and(|meta| filter.matches(meta))
        })
        .collect()
}
