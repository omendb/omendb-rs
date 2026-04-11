//! Text and hybrid search operations for VectorStore.
//!
//! This module contains methods for BM25 text search and hybrid search
//! that combines vector similarity with text relevance using RRF fusion.

use super::helpers;
use super::planner::QueryPlanner;
use super::record_store::RecordStore;
use super::{MetadataFilter, VectorStore};
use crate::text::{
    DEFAULT_RRF_K, HybridResult, TextEngine, TextIndex, TextSearchConfig,
    weighted_reciprocal_rank_fusion, weighted_reciprocal_rank_fusion_with_subscores,
};
use crate::vector::store::input::HybridParams;
use crate::vector::types::Vector;
use anyhow::Result;
use serde_json::Value as JsonValue;

/// ID-score pairs from vector or text search, before fusion.
type ScoredIds = Vec<(String, f32)>;

impl VectorStore {
    /// Enable text search on this store.
    ///
    /// Creates a text index for BM25 keyword search. Must be called before
    /// using `set_with_text()`, `search_text()`, or `search_hybrid()`.
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
        let config = config
            .or_else(|| self.text_search_config.read().clone())
            .unwrap_or_default();
        *self.text_search_config.write() = Some(config.clone());

        if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            storage.put_config("text_writer_buffer_mb", config.writer_buffer_mb as u64)?;
            storage.put_config(
                "text_tokenizer",
                match config.tokenizer {
                    crate::text::TokenizerPreset::Default => 0,
                    crate::text::TokenizerPreset::Code => 1,
                    crate::text::TokenizerPreset::Raw => 2,
                },
            )?;
        }

        if self.text_index.read().is_some() {
            return Ok(());
        }

        *self.text_index.write() = if let Some(ref path) = self.storage_path {
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
        self.text_index.read().is_some()
    }

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
        id: &str,
        vector: Vector,
        text: &str,
        metadata: JsonValue,
    ) -> Result<usize> {
        self.require_dense_schema("set_with_text")?;
        self.require_text_schema("set_with_text")?;

        let mut text_index = self.text_index.write();
        let Some(text_index) = text_index.as_mut() else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        text_index.index_document(id, text)?;
        self.set(id, vector, metadata)
    }

    /// Batch upsert vectors with text content for hybrid search.
    ///
    /// # Arguments
    /// * `batch` - Vector of (id, vector, text, metadata) tuples
    pub fn set_batch_with_text<S: Into<String>>(
        &mut self,
        batch: Vec<(S, Vector, S, JsonValue)>,
    ) -> Result<Vec<usize>> {
        self.require_dense_schema("set_batch_with_text")?;
        self.require_text_schema("set_batch_with_text")?;

        let mut text_index = self.text_index.write();
        let Some(text_index) = text_index.as_mut() else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        // Convert IDs and text to String up front
        let batch: Vec<(String, Vector, String, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, text, metadata)| (id.into(), vector, text.into(), metadata))
            .collect();

        for (id, _, text, _) in &batch {
            text_index.index_document(id, text)?;
        }

        let vector_batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .map(|(id, vector, _, metadata)| (id, vector, metadata))
            .collect();

        self.set_batch(vector_batch)
    }

    /// Index text content for BM25 search without storing a vector.
    ///
    /// Use this with `store()` for multi-vector stores where `set_with_text()`
    /// is not applicable. Call after `store()` with the same id.
    ///
    /// # Arguments
    /// * `id` - Document identifier (must match the id used in `store()`)
    /// * `text` - Text content to index for keyword search
    pub fn index_text(&mut self, id: &str, text: &str) -> Result<()> {
        self.require_text_schema("index_text")?;

        let mut text_index = self.text_index.write();
        let Some(text_index) = text_index.as_mut() else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };
        text_index.index_document(id, text)
    }

    /// Search text index only (BM25 scoring).
    ///
    /// Returns documents ranked by keyword relevance without considering
    /// vector similarity.
    ///
    /// # Arguments
    /// * `query` - Text query
    /// * `k` - Number of results to return
    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<(String, f32)>> {
        self.require_text_schema("search_text")?;

        let text_index = self.text_index.read();
        let Some(text_index) = text_index.as_ref() else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };

        let results = text_index.search(query, k)?;
        Ok(results.into_iter().map(|r| (r.id, r.score)).collect())
    }

    /// Hybrid search combining vector similarity and BM25 text relevance.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine results from vector
    /// and text search with configurable weighting.
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding
    /// * `query_text` - Text query for BM25
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter
    /// * `alpha` - Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5)
    /// * `rrf_k` - RRF constant (default=60, higher reduces rank influence)
    pub fn search_hybrid(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: Option<&MetadataFilter>,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(String, f32, JsonValue)>> {
        self.require_dense_schema("search_hybrid")?;
        self.require_text_schema("search_hybrid")?;

        self.validate_hybrid_search_preconditions(query_vector)?;

        if let Some(engine) = self.published_view.load().as_ref() {
            let text_index_guard = self.text_index.read();
            let metadata_index_guard = self.metadata_index.read();
            let graph_schema = self.graph_schema.read().clone();
            let edge_store_guard = self.edge_store.read();
            let planner = QueryPlanner::new(
                &self.records,
                engine.as_ref(),
                text_index_guard.as_ref().map(|ti| ti as &dyn TextEngine),
                Some(&metadata_index_guard),
                graph_schema,
                edge_store_guard.as_ref(),
            );

            let mut params = HybridParams::new()
                .alpha(alpha.unwrap_or(0.5))
                .rrf_k(rrf_k.unwrap_or(DEFAULT_RRF_K));
            if let Some(f) = filter {
                params = params.filter(f.clone());
            }

            let results = planner.search_hybrid(&query_vector.data, query_text, k, &params)?;
            return Ok(results
                .into_iter()
                .map(|r| (r.id, r.distance, r.metadata))
                .collect());
        }

        let (vector_results, text_results) =
            self.fetch_hybrid_candidates(query_vector, query_text, k, filter)?;

        let fused = weighted_reciprocal_rank_fusion(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata(&self.records, fused))
    }

    /// Hybrid search returning separate keyword and semantic scores.
    ///
    /// Returns [`HybridResult`] with `keyword_score` (BM25) and `semantic_score`
    /// (vector distance) for each result, enabling custom post-processing.
    ///
    /// # Arguments
    /// * `query_vector` - Query embedding
    /// * `query_text` - Text query for BM25
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter
    /// * `alpha` - Weight for vector vs text (0.0=text only, 1.0=vector only, default=0.5)
    /// * `rrf_k` - RRF constant (default=60, higher reduces rank influence)
    pub fn search_hybrid_with_subscores(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: Option<&MetadataFilter>,
        alpha: Option<f32>,
        rrf_k: Option<usize>,
    ) -> Result<Vec<(HybridResult, JsonValue)>> {
        self.require_dense_schema("search_hybrid")?;
        self.require_text_schema("search_hybrid")?;

        self.validate_hybrid_search_preconditions(query_vector)?;

        if let Some(engine) = self.published_view.load().as_ref() {
            let text_index_guard = self.text_index.read();
            let metadata_index_guard = self.metadata_index.read();
            let graph_schema = self.graph_schema.read().clone();
            let edge_store_guard = self.edge_store.read();
            let planner = QueryPlanner::new(
                &self.records,
                engine.as_ref(),
                text_index_guard.as_ref().map(|ti| ti as &dyn TextEngine),
                Some(&metadata_index_guard),
                graph_schema,
                edge_store_guard.as_ref(),
            );

            let mut params = HybridParams::new()
                .alpha(alpha.unwrap_or(0.5))
                .rrf_k(rrf_k.unwrap_or(DEFAULT_RRF_K));
            if let Some(f) = filter {
                params = params.filter(f.clone());
            }

            return planner.search_hybrid_with_subscores(
                &query_vector.data,
                query_text,
                k,
                &params,
            );
        }

        let (vector_results, text_results) =
            self.fetch_hybrid_candidates(query_vector, query_text, k, filter)?;

        let fused = weighted_reciprocal_rank_fusion_with_subscores(
            vector_results,
            text_results,
            k,
            rrf_k.unwrap_or(DEFAULT_RRF_K),
            alpha.unwrap_or(0.5),
        );

        Ok(attach_metadata_to_hybrid_results(&self.records, fused))
    }

    /// Fetch vector and text candidates for hybrid search, optionally applying a metadata filter.
    fn fetch_hybrid_candidates(
        &self,
        query_vector: &Vector,
        query_text: &str,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<(ScoredIds, ScoredIds)> {
        if let Some(filter) = filter {
            let fetch_k = k * 4;

            let vector_results = self.knn_search_with_filter(query_vector, fetch_k, filter)?;
            let vector_results: ScoredIds = vector_results
                .into_iter()
                .map(|r| (r.id, r.distance))
                .collect();

            let text_results = self.search_text(query_text, fetch_k)?;
            let text_results = filter_text_results_by_metadata(&self.records, text_results, filter);

            Ok((vector_results, text_results))
        } else {
            let fetch_k = k * 2;

            let vector_results = self.knn_search(query_vector, fetch_k)?;
            let vector_results = self.convert_knn_results_to_id_scores(vector_results);

            let text_results = self.search_text(query_text, fetch_k)?;

            Ok((vector_results, text_results))
        }
    }

    /// Validate preconditions for hybrid search.
    fn validate_hybrid_search_preconditions(&self, query_vector: &Vector) -> Result<()> {
        if query_vector.data.len() != self.dimensions() {
            anyhow::bail!(
                "Query vector dimension {} does not match store dimension {}",
                query_vector.data.len(),
                self.dimensions()
            );
        }
        if self.text_index.read().is_none() {
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
                    .map(|id| (id.clone(), distance))
            })
            .collect()
    }
}

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
                .and_then(|r| r.metadata)
                .is_some_and(|meta| filter.matches(&meta))
        })
        .collect()
}
