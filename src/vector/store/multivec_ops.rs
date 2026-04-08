//! Multi-vector operations for VectorStore.
//!
//! This module contains methods for ColBERT-style token embedding storage and search.
//! Uses MUVERA (Multi-Vector Retrieval via Fixed Dimensional Encodings) for efficient
//! approximate nearest neighbor search with MaxSim reranking.

use super::helpers;
use super::input::{BatchItem, QueryData, Rerank, SearchOptions, VectorData, VectorInput};
use super::{SearchResult, VectorStore};
use crate::vector::muvera::MuveraEncoder;
use crate::vector::types::Vector;
use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value as JsonValue;

impl VectorStore {
    /// Check if this store is configured for multi-vector documents.
    #[must_use]
    pub fn is_multi_vector(&self) -> bool {
        self.muvera_encoder.is_some()
    }

    /// Get the token dimension (for multi-vector stores).
    #[must_use]
    pub fn token_dimension(&self) -> Option<usize> {
        self.muvera_encoder
            .as_ref()
            .map(MuveraEncoder::token_dimension)
    }

    /// Get the encoded dimension (for multi-vector stores).
    #[must_use]
    pub fn encoded_dimension(&self) -> Option<usize> {
        self.muvera_encoder
            .as_ref()
            .map(MuveraEncoder::fde_dimension)
    }

    /// Insert a document with token embeddings.
    ///
    /// Use this for ColBERT-style retrieval where documents are represented as
    /// sets of token embeddings rather than single vectors.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique document identifier
    /// * `tokens` - Token embeddings (each token must have the configured dimension)
    /// * `metadata` - Optional JSON metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use omendb::VectorStore;
    ///
    /// let mut store = VectorStore::multi_vector(128);
    ///
    /// // Document with 10 token embeddings
    /// let tokens: Vec<Vec<f32>> = vec![vec![0.1; 128]; 10];
    /// let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
    ///
    /// store.set_multi("doc1", &refs, serde_json::json!({"title": "Example"})).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Store is not configured for multi-vector (use `VectorStore::multi_vector()`)
    /// - Token array is empty
    /// - Token dimension doesn't match configured dimension
    /// - Token count exceeds `max_tokens` (default 512)
    #[allow(dead_code)]
    pub(crate) fn set_multi(
        &mut self,
        id: &str,
        tokens: &[&[f32]],
        metadata: JsonValue,
    ) -> Result<()> {
        self.store_multi_internal(tokens, id, metadata)
    }

    /// Batch insert documents with token embeddings.
    ///
    /// More efficient than calling `set_multi()` in a loop because it
    /// encodes all documents in parallel and uses batch HNSW insertion.
    ///
    /// # Arguments
    ///
    /// * `batch` - Vector of (id, tokens, metadata) tuples
    ///
    /// # Errors
    ///
    /// Returns an error if any document fails validation (same rules as `set_multi`).
    #[allow(dead_code)]
    pub(crate) fn set_multi_batch(
        &mut self,
        batch: Vec<(&str, Vec<Vec<f32>>, JsonValue)>,
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // Validate MUVERA is enabled
        let encoder = self.muvera_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Store not configured for multi-vector. Use VectorStore::new_muvera()")
        })?;

        let token_dim = encoder.token_dimension();

        // Get config limits before validation
        let pool_factor = encoder.config().pool_factor;
        let max_tokens = encoder.config().max_tokens;

        // Validate all documents first
        for (i, (id, tokens, _)) in batch.iter().enumerate() {
            if tokens.is_empty() {
                anyhow::bail!("Document '{id}' (index {i}) has empty tokens");
            }
            if let Some(limit) = max_tokens {
                // Adjust limit for pool_factor so we don't run k-means just to check
                let pre_pool_limit = pool_factor.map_or(limit, |pf| limit * pf as usize);
                if tokens.len() > pre_pool_limit {
                    anyhow::bail!(
                        "Document '{}' has {} tokens, exceeds max_tokens {}",
                        id,
                        tokens.len(),
                        limit
                    );
                }
            }
            for (j, token) in tokens.iter().enumerate() {
                if token.len() != token_dim {
                    anyhow::bail!(
                        "Document '{}' token {} has dimension {} but expected {}",
                        id,
                        j,
                        token.len(),
                        token_dim
                    );
                }
            }
        }

        // Encode all documents to FDEs in parallel, applying pooling if configured
        let pooled_and_fdes: Vec<(Vec<Vec<f32>>, Vec<f32>)> = batch
            .par_iter()
            .map(|(_, tokens, _)| {
                let token_refs: Vec<&[f32]> = tokens.iter().map(std::vec::Vec::as_slice).collect();

                // Apply pooling if configured, otherwise use original tokens
                let (pooled_tokens, fde) = if let Some(pf) = pool_factor {
                    let pooled = crate::vector::muvera::pool_tokens(&token_refs, pf);
                    let final_refs: Vec<&[f32]> =
                        pooled.iter().map(std::vec::Vec::as_slice).collect();
                    let fde = encoder.encode_document(&final_refs);
                    (pooled, fde)
                } else {
                    // No pooling - encode directly from original tokens, no clone needed
                    let fde = encoder.encode_document(&token_refs);
                    (tokens.clone(), fde) // Clone only for return value storage
                };
                (pooled_tokens, fde)
            })
            .collect();

        // Determine update vs insert order to match set_batch processing
        // set_batch processes updates first, then inserts, so we must store tokens in that order
        let mut update_indices = Vec::new();
        let mut insert_indices = Vec::new();
        for (i, (id, _, _)) in batch.iter().enumerate() {
            if self.records.get_slot(id).is_some() {
                update_indices.push(i);
            } else {
                insert_indices.push(i);
            }
        }

        // Store pooled tokens in set_batch's processing order (updates first, then inserts)
        let mut multivec_storage_guard = self.multivec_storage.write();
        let multivec_storage = multivec_storage_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;

        for &i in update_indices.iter().chain(insert_indices.iter()) {
            let token_refs: Vec<&[f32]> = pooled_and_fdes[i]
                .0
                .iter()
                .map(std::vec::Vec::as_slice)
                .collect();
            multivec_storage
                .add(&token_refs)
                .map_err(anyhow::Error::msg)?;
        }

        // Prepare batch for set_batch (maintains original batch order - set_batch reorders internally)
        let fde_batch: Vec<(String, Vector, JsonValue)> = batch
            .into_iter()
            .zip(pooled_and_fdes.iter())
            .map(|((id, _, metadata), (_, fde))| {
                (id.to_string(), Vector::new(fde.clone()), metadata)
            })
            .collect();

        // Use existing set_batch for efficient HNSW insertion
        let result_slots = self.set_batch(fde_batch)?;
        for (slot, &i) in result_slots
            .iter()
            .zip(update_indices.iter().chain(insert_indices.iter()))
        {
            self.records
                .update_multi(*slot as u32, Some(pooled_and_fdes[i].0.clone()))?;
        }

        Ok(())
    }

    /// Search for similar documents using query tokens.
    ///
    /// This is the recommended search method for multi-vector stores. It:
    /// 1. Finds candidate documents using fast approximate search
    /// 2. Reranks candidates using exact MaxSim scoring
    /// 3. Returns top-k results ordered by similarity
    ///
    /// # Arguments
    ///
    /// * `query_tokens` - Query token embeddings
    /// * `k` - Number of results to return
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use omendb::VectorStore;
    ///
    /// let store = VectorStore::multi_vector(128);
    /// // ... insert documents ...
    ///
    /// let query: Vec<Vec<f32>> = vec![vec![0.1; 128]; 5]; // 5 query tokens
    /// let refs: Vec<&[f32]> = query.iter().map(|t| t.as_slice()).collect();
    ///
    /// let results = store.search_multi(&refs, 10).unwrap();
    /// ```
    #[allow(dead_code)]
    pub(crate) fn search_multi(
        &self,
        query_tokens: &[&[f32]],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Default rerank factor of 20 (fetch 20x candidates, rerank to k)
        self.search_multi_rerank(query_tokens, k, 20, false)
    }

    /// Fast approximate search without MaxSim reranking.
    ///
    /// Use this when search speed is more important than quality. Returns
    /// results based on FDE (Fixed Dimensional Encoding) similarity only.
    ///
    /// For higher quality results, use `search_multi()` which includes reranking.
    pub(crate) fn search_multi_approx(
        &self,
        query_tokens: &[&[f32]],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Validate multi-vector is enabled
        let encoder = self.muvera_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Store not configured for multi-vector. Use VectorStore::multi_vector()"
            )
        })?;

        if query_tokens.is_empty() {
            anyhow::bail!("Cannot search with empty query tokens");
        }

        let token_dim = encoder.token_dimension();
        for (i, token) in query_tokens.iter().enumerate() {
            if token.len() != token_dim {
                anyhow::bail!(
                    "Query token {} has dimension {} but expected {}",
                    i,
                    token.len(),
                    token_dim
                );
            }
        }

        // Encode query tokens to FDE (query mode = SUM)
        let query_fde = encoder.encode_query(query_tokens);

        // Auto-boost ef_search for high-dimensional FDE vectors
        // Weaviate uses ef >= FDE_dim; we use fde_dim / 16 as a reasonable default
        let fde_dim = encoder.fde_dimension();
        let boosted_ef = (fde_dim / 16).max(self.ef_search());

        // Search HNSW with FDE using boosted ef
        let query_vec = Vector::new(query_fde);
        let results = self.knn_search_with_ef(&query_vec, k, Some(boosted_ef))?;

        // Convert to SearchResult (filter deleted nodes)
        let search_results = results
            .into_iter()
            .filter(|(slot, _)| self.records.is_live(*slot as u32))
            .map(|(slot, distance)| {
                let record = self.records.get_by_slot(slot as u32);
                let id = record
                    .as_ref()
                    .map_or_else(|| format!("__slot_{slot}"), |r| r.id.clone());
                let metadata = record
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or(JsonValue::Null);
                SearchResult::new(id, distance, metadata)
            })
            .collect();

        Ok(search_results)
    }

    /// Search with custom rerank factor.
    ///
    /// Lower factor = faster but lower quality.
    /// Higher factor = slower but higher quality.
    ///
    /// # Arguments
    ///
    /// * `query_tokens` - Query token embeddings
    /// * `k` - Number of results to return
    /// * `rerank_factor` - Fetch this many times k candidates before reranking
    pub(crate) fn search_multi_rerank(
        &self,
        query_tokens: &[&[f32]],
        k: usize,
        rerank_factor: usize,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        // Validate multi-vector is enabled
        let encoder = self.muvera_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Store not configured for multi-vector. Use VectorStore::multi_vector()"
            )
        })?;
        let multivec_storage_guard = self.multivec_storage.read();
        let multivec_storage = multivec_storage_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;

        if query_tokens.is_empty() {
            anyhow::bail!("Cannot search with empty query tokens");
        }

        let token_dim = encoder.token_dimension();
        for (i, token) in query_tokens.iter().enumerate() {
            if token.len() != token_dim {
                anyhow::bail!(
                    "Query token {} has dimension {} but expected {}",
                    i,
                    token.len(),
                    token_dim
                );
            }
        }

        // Step 1: Get candidates using FDE search
        let num_candidates = k * rerank_factor;

        // Auto-boost ef_search for high-dimensional FDE vectors
        let fde_dim = encoder.fde_dimension();
        let boosted_ef = (fde_dim / 16).max(self.ef_search()).max(num_candidates);

        let query_fde = encoder.encode_query(query_tokens);
        let query_vec = Vector::new(query_fde);
        let candidates = self.knn_search_with_ef(&query_vec, num_candidates, Some(boosted_ef))?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Collect original tokens for each candidate (filter deleted nodes)
        let mut candidate_data: Vec<(usize, String, JsonValue, Vec<&[f32]>)> = Vec::new();

        for (slot, _fde_distance) in &candidates {
            let slot_u32 = *slot as u32;

            // Skip deleted nodes (HNSW may return stale references)
            if !self.records.is_live(slot_u32) {
                continue;
            }

            // Get document's original tokens from MultiVecStorage
            if let Some(doc_tokens) = multivec_storage.get_tokens(slot_u32) {
                let record = self.records.get_by_slot(slot_u32);
                let id = record
                    .as_ref()
                    .map_or_else(|| format!("__slot_{slot}"), |r| r.id.clone());
                let metadata = record
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or(JsonValue::Null);
                candidate_data.push((*slot, id, metadata, doc_tokens));
            }
        }

        if candidate_data.is_empty() {
            return Ok(Vec::new());
        }

        // Step 3: Compute MaxSim scores (and explanations if requested)
        let results: Vec<SearchResult> = if explain {
            let mut explained = Vec::with_capacity(candidate_data.len());
            for (_, id, metadata, tokens) in candidate_data {
                let explanation = crate::vector::muvera::maxsim_explain(query_tokens, &tokens);
                let score = explanation.score;
                let explanation_json = serde_json::to_value(explanation)?;
                explained.push(SearchResult::with_explanation(
                    id,
                    -score,
                    metadata,
                    explanation_json,
                ));
            }
            explained.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            explained.into_iter().take(k).collect()
        } else {
            let doc_tokens_refs: Vec<&Vec<&[f32]>> = candidate_data
                .iter()
                .map(|(_, _, _, tokens)| tokens)
                .collect();

            let maxsim_scores = super::super::muvera::maxsim_batch(query_tokens, &doc_tokens_refs);

            let mut scored: Vec<(String, JsonValue, f32)> = candidate_data
                .into_iter()
                .zip(maxsim_scores.into_iter())
                .map(|((_, id, metadata, _), score)| (id, metadata, score))
                .collect();

            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            scored
                .into_iter()
                .take(k)
                .map(|(id, metadata, score)| SearchResult::new(id, -score, metadata))
                .collect()
        };

        Ok(results)
    }

    /// Store vector or token embeddings.
    ///
    /// This unified method works for both regular and multi-vector stores:
    /// - Regular stores: pass `&[f32]` or `Vec<f32>`
    /// - Multi-vector stores: pass `&[&[f32]]` or `Vec<Vec<f32>>`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use omendb::VectorStore;
    /// use serde_json::json;
    ///
    /// // Regular store
    /// let mut store = VectorStore::new(3);
    /// store.store("doc1", vec![1.0, 2.0, 3.0], json!({})).unwrap();
    ///
    /// // Multi-vector store
    /// let mut store = VectorStore::multi_vector(3);
    /// let tokens = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    /// store.store("doc1", tokens, json!({})).unwrap();
    /// ```
    pub fn store<V: VectorInput>(&mut self, id: &str, data: V, metadata: JsonValue) -> Result<()> {
        let data = data.into_vector_data();

        match (&self.muvera_encoder, data) {
            // Regular store, single vector
            (None, VectorData::Single(vec)) => {
                self.set(id, Vector::new(vec), metadata)?;
                Ok(())
            }
            // Multi-vector store, multi-vector data
            (Some(_), VectorData::Multi(tokens)) => {
                let token_refs: Vec<&[f32]> = tokens.iter().map(std::vec::Vec::as_slice).collect();
                self.store_multi_internal(&token_refs, id, metadata)
            }
            // Sparse vector (works for both dense and multi-vector stores)
            (_, VectorData::Sparse(sv)) => self.set_sparse(id, sv, metadata),
            // Error: wrong type for store
            (None, VectorData::Multi(_)) => {
                anyhow::bail!(
                    "Cannot store token embeddings in regular store. \
                    Pass single vector (&[f32] or Vec<f32>), or create multi-vector store with VectorStore::multi_vector()"
                );
            }
            (Some(_), VectorData::Single(_)) => {
                anyhow::bail!(
                    "Cannot store single vector in multi-vector store. \
                    Pass token embeddings (&[&[f32]] or Vec<Vec<f32>>)"
                );
            }
        }
    }

    /// Store vector or token embeddings with text content for hybrid search.
    ///
    /// Indexes the text for BM25 search and stores the vector/tokens.
    /// Works for both regular and multi-vector stores.
    ///
    /// Requires text search to be enabled (`enable_text_search()`).
    pub fn store_with_text<V: VectorInput>(
        &mut self,
        id: &str,
        data: V,
        text: &str,
        metadata: JsonValue,
    ) -> Result<()> {
        let mut text_index_guard = self.text_index.write();
        let Some(text_index) = text_index_guard.as_mut() else {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        };
        text_index.index_document(id, text)?;
        drop(text_index_guard);
        self.store(id, data, metadata)
    }

    /// Internal helper for store() with multi-vector tokens.
    pub(super) fn store_multi_internal(
        &mut self,
        tokens: &[&[f32]],
        id: &str,
        metadata: JsonValue,
    ) -> Result<()> {
        // Validate MUVERA is enabled
        let encoder = self.muvera_encoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Store not configured for multi-vector. Use VectorStore::multi_vector()"
            )
        })?;

        // Validate tokens
        if tokens.is_empty() {
            anyhow::bail!("Cannot store empty token set");
        }

        let token_dim = encoder.token_dimension();
        for (i, token) in tokens.iter().enumerate() {
            if token.len() != token_dim {
                anyhow::bail!(
                    "Token {} has dimension {} but expected {}",
                    i,
                    token.len(),
                    token_dim
                );
            }
        }

        // Apply pooling if configured
        let pooled_tokens: Vec<Vec<f32>> = if let Some(pf) = encoder.config().pool_factor {
            crate::vector::muvera::pool_tokens(tokens, pf)
        } else {
            tokens.iter().map(|t| t.to_vec()).collect()
        };

        // Check post-pooling token count against configured limit
        if let Some(limit) = encoder.config().max_tokens
            && pooled_tokens.len() > limit
        {
            anyhow::bail!(
                "Token count {} exceeds max_tokens {}",
                pooled_tokens.len(),
                limit
            );
        }

        // Create refs from the pooled/original tokens
        let final_refs: Vec<&[f32]> = pooled_tokens.iter().map(std::vec::Vec::as_slice).collect();

        // Encode tokens to FDE (document mode = AVERAGE)
        let fde = encoder.encode_document(&final_refs);

        // Store FDE first (can fail without corrupting multivec_storage)
        let slot = self.set(id, Vector::new(fde), metadata)?;
        self.records
            .update_multi(slot as u32, Some(pooled_tokens.clone()))?;

        // Then add tokens - slot already committed (store pooled tokens for reranking)
        let mut multivec_storage_guard = self.multivec_storage.write();
        let multivec_storage = multivec_storage_guard
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;
        let token_slot = multivec_storage
            .add(&final_refs)
            .map_err(anyhow::Error::msg)?;

        if slot as u32 != token_slot {
            anyhow::bail!(
                "Internal error: slot mismatch RecordStore={slot} MultiVecStorage={token_slot}. \
                 This indicates a bug - please report it."
            );
        }

        Ok(())
    }

    /// Search for similar vectors/documents.
    ///
    /// This unified method works for both regular and multi-vector stores:
    /// - Regular stores: pass `&[f32]` or `Vec<f32>`
    /// - Multi-vector stores: pass `&[&[f32]]` or `Vec<Vec<f32>>`
    ///
    /// For multi-vector stores, reranking is enabled by default for quality.
    /// Use `query_with_options` with `Rerank::Off` for fast approximate search.
    #[allow(dead_code)]
    pub(crate) fn query<Q: super::input::QueryInput>(
        &self,
        query: &Q,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.query_with_options(query, k, &SearchOptions::default())
    }

    /// Search with options.
    ///
    /// Provides control over ef, filter, max_distance, and reranking.
    pub fn query_with_options<Q: super::input::QueryInput>(
        &self,
        query: &Q,
        k: usize,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_data = query.to_query_data();

        match (&self.muvera_encoder, &query_data) {
            // Regular store, single vector query
            (None, QueryData::Single(vec)) => self.search_single_internal(vec, k, options),
            // Multi-vector store, multi-vector query
            (Some(_), QueryData::Multi(tokens)) => self.search_multi_internal(tokens, k, options),
            // Sparse vector query (always available)
            (_, QueryData::Sparse(sv)) => self.sparse_search(sv, k, options.filter.as_ref()),
            // Error: wrong query type for store
            (None, QueryData::Multi(_)) => {
                anyhow::bail!(
                    "Cannot query regular store with token embeddings. \
                    Pass single vector (&[f32] or Vec<f32>)"
                );
            }
            (Some(_), QueryData::Single(_)) => {
                anyhow::bail!(
                    "Cannot query multi-vector store with single vector. \
                    Pass token embeddings (&[&[f32]] or Vec<Vec<f32>>)"
                );
            }
        }
    }

    /// Internal: search with single vector.
    fn search_single_internal(
        &self,
        query: &[f32],
        k: usize,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        let query_vec = Vector::new(query.to_vec());
        self.search_with_options(
            &query_vec,
            k,
            options.filter.as_ref(),
            options.ef,
            options.max_distance,
        )
    }

    /// Threshold below which brute-force MaxSim is used instead of FDE+HNSW.
    /// At 5K docs with ~100 tokens each, brute-force MaxSim takes ~16ms and gives 100% recall.
    const BRUTE_FORCE_MAXSIM_THRESHOLD: u32 = 5000;

    /// Internal: search with multi-vector tokens.
    fn search_multi_internal(
        &self,
        query_tokens: &[&[f32]],
        k: usize,
        options: &SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        if query_tokens.is_empty() {
            anyhow::bail!("Cannot search with empty query tokens");
        }
        if options.filter.is_some() {
            anyhow::bail!("Metadata filters are not yet supported for multi-vector search");
        }
        if options.max_distance.is_some() {
            anyhow::bail!("max_distance is not yet supported for multi-vector search");
        }

        // Validate token dimensions before dispatching to any search path
        let encoder = self
            .muvera_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Store not configured for multi-vector"))?;
        let token_dim = encoder.token_dimension();
        for (i, token) in query_tokens.iter().enumerate() {
            if token.len() != token_dim {
                anyhow::bail!(
                    "Query token {} has dimension {} but expected {}",
                    i,
                    token.len(),
                    token_dim
                );
            }
        }

        // For small collections, skip FDE approximation and use brute-force MaxSim
        // for perfect recall (unless user explicitly disabled reranking)
        if !matches!(options.rerank, Rerank::Off)
            && self.records.len() <= Self::BRUTE_FORCE_MAXSIM_THRESHOLD
            && !self.records.is_empty()
        {
            return self.search_multi_bruteforce(query_tokens, k, options.explain);
        }

        match options.rerank {
            Rerank::Off => self.search_multi_approx(query_tokens, k),
            Rerank::On => self.search_multi_rerank(query_tokens, k, 20, options.explain),
            Rerank::Factor(f) => self.search_multi_rerank(query_tokens, k, f, options.explain),
        }
    }

    /// Brute-force MaxSim search over all live documents.
    /// Used for small collections where perfect recall is cheap.
    fn search_multi_bruteforce(
        &self,
        query_tokens: &[&[f32]],
        k: usize,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let multivec_storage_guard = self.multivec_storage.read();
        let multivec_storage = multivec_storage_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;

        // Collect slot + tokens only (no id/metadata cloning)
        let mut candidate_slots: Vec<u32> = Vec::new();
        let mut candidate_tokens: Vec<Vec<&[f32]>> = Vec::new();

        for (slot, _record) in self.records.iter_live() {
            if let Some(doc_tokens) = multivec_storage.get_tokens(slot) {
                candidate_slots.push(slot);
                candidate_tokens.push(doc_tokens);
            }
        }

        if candidate_slots.is_empty() {
            return Ok(Vec::new());
        }

        let results: Vec<SearchResult> = if explain {
            let mut explained = Vec::with_capacity(candidate_slots.len());
            for (slot, tokens) in candidate_slots
                .into_iter()
                .zip(candidate_tokens.into_iter())
            {
                let explanation = crate::vector::muvera::maxsim_explain(query_tokens, &tokens);
                let score = explanation.score;
                let explanation_json = serde_json::to_value(explanation)?;

                if let Some(record) = self.records.get_by_slot(slot) {
                    let metadata = record.metadata.clone().unwrap_or(JsonValue::Null);
                    explained.push(SearchResult::with_explanation(
                        record.id.clone(),
                        -score,
                        metadata,
                        explanation_json,
                    ));
                }
            }
            explained.sort_by(|a, b| a.distance.total_cmp(&b.distance));
            explained.into_iter().take(k).collect()
        } else {
            // Compute MaxSim scores in parallel
            let doc_tokens_refs: Vec<&Vec<&[f32]>> = candidate_tokens.iter().collect();
            let maxsim_scores =
                super::super::muvera::maxsim_batch_par(query_tokens, &doc_tokens_refs);

            // Sort descending by MaxSim score, take top-k
            let mut scored: Vec<(usize, f32)> = maxsim_scores.into_iter().enumerate().collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));

            // Resolve id/metadata only for final top-k results
            scored
                .into_iter()
                .take(k)
                .filter_map(|(idx, score)| {
                    let slot = candidate_slots[idx];
                    let record = self.records.get_by_slot(slot)?;
                    let metadata = record.metadata.clone().unwrap_or(JsonValue::Null);
                    Some(SearchResult::new(record.id.clone(), -score, metadata))
                })
                .collect()
        };

        Ok(results)
    }

    /// Search multi-vector store using BM25 text candidates + MaxSim reranking.
    ///
    /// Uses BM25 keyword search to find candidate documents, then reranks
    /// them using exact MaxSim scoring on stored token embeddings. This
    /// bypasses the FDE approximation entirely, using text relevance for
    /// candidate generation instead.
    ///
    /// Requires text search to be enabled (`enable_text_search()`).
    ///
    /// * `query_text` - Text query for BM25 candidate generation
    /// * `query_tokens` - Token embeddings for MaxSim reranking
    /// * `k` - Number of results to return
    /// * `num_candidates` - Number of BM25 candidates to fetch before reranking
    /// * `explain` - If true, return token alignment metadata
    pub fn search_multi_with_text(
        &self,
        query_text: &str,
        query_tokens: &[&[f32]],
        k: usize,
        num_candidates: Option<usize>,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let multivec_storage_guard = self.multivec_storage.read();
        let multivec_storage = multivec_storage_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;

        if !self.has_text_search() {
            anyhow::bail!("Text search not enabled. Call enable_text_search() first.");
        }

        if query_tokens.is_empty() {
            anyhow::bail!("Cannot rerank with empty query tokens");
        }

        // Get BM25 candidates (default: 10x k)
        let n_candidates = num_candidates.unwrap_or(k * 10);
        let text_hits = self.search_text(query_text, n_candidates)?;

        if text_hits.is_empty() {
            return Ok(Vec::new());
        }

        // Look up token embeddings for each BM25 candidate
        let mut candidate_data: Vec<(String, JsonValue, Vec<&[f32]>)> = Vec::new();

        for (id, _bm25_score) in &text_hits {
            let slot = match self.records.get_slot(id) {
                Some(s) => s,
                None => continue,
            };
            if let Some(doc_tokens) = multivec_storage.get_tokens(slot) {
                let metadata = self
                    .records
                    .get(id)
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or(JsonValue::Null);
                candidate_data.push((id.clone(), metadata, doc_tokens));
            }
        }

        if candidate_data.is_empty() {
            return Ok(Vec::new());
        }

        // MaxSim rerank
        let results: Vec<SearchResult> = if explain {
            let mut explained = Vec::with_capacity(candidate_data.len());
            for (id, metadata, tokens) in candidate_data {
                let explanation = crate::vector::muvera::maxsim_explain(query_tokens, &tokens);
                let score = explanation.score;
                let explanation_json = serde_json::to_value(explanation)?;
                explained.push(SearchResult::with_explanation(
                    id,
                    -score,
                    metadata,
                    explanation_json,
                ));
            }
            explained.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            explained.into_iter().take(k).collect()
        } else {
            let doc_tokens_refs: Vec<&Vec<&[f32]>> =
                candidate_data.iter().map(|(_, _, tokens)| tokens).collect();

            let maxsim_scores = super::super::muvera::maxsim_batch(query_tokens, &doc_tokens_refs);

            // Sort descending by MaxSim score, take top-k
            let mut scored: Vec<(usize, f32)> = maxsim_scores.into_iter().enumerate().collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));

            scored
                .into_iter()
                .take(k)
                .map(|(idx, score)| {
                    let (ref id, ref metadata, _) = candidate_data[idx];
                    SearchResult::new(id.clone(), -score, metadata.clone())
                })
                .collect()
        };

        Ok(results)
    }

    /// Get stored data by ID.
    ///
    /// Returns `VectorData::Single` for regular stores, `VectorData::Multi` for multi-vector stores.
    pub fn get_data(&self, id: &str) -> Option<(VectorData, JsonValue)> {
        if self.is_multi_vector() {
            self.get_tokens_internal(id)
                .map(|(tokens, meta)| (VectorData::Multi(tokens), meta))
        } else {
            self.get_vector_internal(id)
                .map(|(vec, meta)| (VectorData::Single(vec), meta))
        }
    }

    /// Get as single vector (convenience for regular stores).
    ///
    /// Returns `None` if ID not found or store is multi-vector.
    pub fn get_vector(&self, id: &str) -> Option<(Vec<f32>, JsonValue)> {
        if self.is_multi_vector() {
            None
        } else {
            self.get_vector_internal(id)
        }
    }

    /// Get as token embeddings (convenience for multi-vector stores).
    ///
    /// Returns `None` if ID not found or store is regular.
    pub fn get_tokens(&self, id: &str) -> Option<(Vec<Vec<f32>>, JsonValue)> {
        if self.is_multi_vector() {
            self.get_tokens_internal(id)
        } else {
            None
        }
    }

    /// Internal: get single vector.
    fn get_vector_internal(&self, id: &str) -> Option<(Vec<f32>, JsonValue)> {
        let record = self.records.get(id)?;
        let metadata = record
            .metadata
            .clone()
            .unwrap_or_else(helpers::default_metadata);
        Some((record.vector.to_vec(), metadata))
    }

    /// Internal: get multi-vector tokens.
    fn get_tokens_internal(&self, id: &str) -> Option<(Vec<Vec<f32>>, JsonValue)> {
        let slot = self.records.get_slot(id)?;
        let tokens = self.records.get_multi(slot)?;
        let metadata = self
            .records
            .get(id)?
            .metadata
            .clone()
            .unwrap_or_else(helpers::default_metadata);
        Some((tokens, metadata))
    }

    /// Batch store vectors or token embeddings.
    ///
    /// More efficient than calling `store()` in a loop.
    #[allow(dead_code)]
    pub(crate) fn store_batch<I, B>(&mut self, batch: I) -> Result<()>
    where
        I: IntoIterator<Item = B>,
        B: BatchItem,
    {
        let items: Vec<_> = batch.into_iter().map(BatchItem::into_parts).collect();

        if items.is_empty() {
            return Ok(());
        }

        // Check first item to determine type
        let is_multi = items.first().is_some_and(|(_, data, _)| data.is_multi());

        // Validate all items are same type
        for (i, (_, data, _)) in items.iter().enumerate() {
            if data.is_multi() != is_multi {
                anyhow::bail!(
                    "Batch item {i} has different type than item 0. \
                    All items must be same type (all single vectors or all token embeddings)"
                );
            }
        }

        if is_multi {
            if self.muvera_encoder.is_none() {
                anyhow::bail!(
                    "Cannot store token embeddings in regular store. \
                    Create multi-vector store with VectorStore::multi_vector()"
                );
            }
            self.store_multi_batch_internal(items)
        } else {
            if self.muvera_encoder.is_some() {
                anyhow::bail!(
                    "Cannot store single vectors in multi-vector store. \
                    Pass token embeddings"
                );
            }
            self.store_single_batch_internal(items)
        }
    }

    /// Internal: batch set single vectors.
    fn store_single_batch_internal(
        &mut self,
        items: Vec<(String, VectorData, JsonValue)>,
    ) -> Result<()> {
        let batch: Vec<(String, Vector, JsonValue)> = items
            .into_iter()
            .map(|(id, data, meta)| {
                let VectorData::Single(vec) = data else {
                    unreachable!("pre-filtered to Single")
                };
                (id, Vector::new(vec), meta)
            })
            .collect();

        self.set_batch(batch)?;
        Ok(())
    }

    /// Internal: batch set multi-vectors.
    fn store_multi_batch_internal(
        &mut self,
        items: Vec<(String, VectorData, JsonValue)>,
    ) -> Result<()> {
        let batch: Vec<(&str, Vec<Vec<f32>>, JsonValue)> = items
            .iter()
            .map(|(id, data, meta)| {
                let VectorData::Multi(tokens) = data else {
                    unreachable!("pre-filtered to Multi")
                };
                (id.as_str(), tokens.clone(), meta.clone())
            })
            .collect();

        self.set_multi_batch(batch)
    }

    /// Search multi-vector store using sparse retrieval + MaxSim reranking.
    ///
    /// Uses SPLADE-style sparse vectors for fast candidate generation, then
    /// reranks using exact MaxSim scoring on stored ColBERT token embeddings.
    /// This is the 24x-faster-than-PLAID pipeline.
    ///
    /// Requires both sparse index and multi-vector storage to be enabled.
    ///
    /// # Arguments
    ///
    /// * `sparse_query` - Sparse query vector for candidate retrieval
    /// * `query_tokens` - Token embeddings for MaxSim reranking
    /// * `k` - Number of results to return
    /// * `num_candidates` - Number of sparse candidates to fetch before reranking (default: 10x k)
    /// * `explain` - If true, return token alignment metadata
    pub fn search_multi_with_sparse(
        &self,
        sparse_query: &crate::vector::sparse::SparseVector,
        query_tokens: &[&[f32]],
        k: usize,
        num_candidates: Option<usize>,
        explain: bool,
    ) -> Result<Vec<SearchResult>> {
        let multivec_storage_guard = self.multivec_storage.read();
        let multivec_storage = multivec_storage_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiVecStorage not initialized"))?;
        let sparse_guard = self.sparse_index.read();
        let sparse_index = sparse_guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Sparse index not enabled. Call enable_sparse() first")
        })?;

        if query_tokens.is_empty() {
            anyhow::bail!("Cannot rerank with empty query tokens");
        }

        // Get sparse candidates (default: 10x k)
        let n_candidates = num_candidates.unwrap_or(k * 10);
        let sparse_hits = sparse_index.search(sparse_query, n_candidates);

        if sparse_hits.is_empty() {
            return Ok(Vec::new());
        }

        // Look up token embeddings for each sparse candidate
        let mut candidate_data: Vec<(String, JsonValue, Vec<&[f32]>)> = Vec::new();

        for (slot, _score) in &sparse_hits {
            if !self.records.is_live(*slot) {
                continue;
            }
            if let Some(doc_tokens) = multivec_storage.get_tokens(*slot) {
                let record = self.records.get_by_slot(*slot);
                let id = record
                    .as_ref()
                    .map_or_else(|| format!("__slot_{slot}"), |r| r.id.clone());
                let metadata = record
                    .and_then(|r| r.metadata.clone())
                    .unwrap_or(JsonValue::Null);
                candidate_data.push((id, metadata, doc_tokens));
            }
        }

        if candidate_data.is_empty() {
            return Ok(Vec::new());
        }

        // MaxSim rerank
        let results: Vec<SearchResult> = if explain {
            let mut explained = Vec::with_capacity(candidate_data.len());
            for (id, metadata, tokens) in candidate_data {
                let explanation = crate::vector::muvera::maxsim_explain(query_tokens, &tokens);
                let score = explanation.score;
                let explanation_json = serde_json::to_value(explanation)?;
                explained.push(SearchResult::with_explanation(
                    id,
                    -score,
                    metadata,
                    explanation_json,
                ));
            }
            explained.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            explained.into_iter().take(k).collect()
        } else {
            let doc_tokens_refs: Vec<&Vec<&[f32]>> =
                candidate_data.iter().map(|(_, _, tokens)| tokens).collect();

            let maxsim_scores = super::super::muvera::maxsim_batch(query_tokens, &doc_tokens_refs);

            // Sort descending by MaxSim score, take top-k
            let mut scored: Vec<(usize, f32)> = maxsim_scores.into_iter().enumerate().collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));

            scored
                .into_iter()
                .take(k)
                .map(|(idx, score)| {
                    let (ref id, ref metadata, _) = candidate_data[idx];
                    SearchResult::new(id.clone(), -score, metadata.clone())
                })
                .collect()
        };

        Ok(results)
    }
}
