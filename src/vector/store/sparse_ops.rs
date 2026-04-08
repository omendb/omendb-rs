//! Sparse vector operations for VectorStore.
//!
//! Provides methods for inserting, searching, and hybrid-searching sparse vectors
//! using an inverted index. Sparse vectors are used for learned sparse retrieval
//! models like SPLADE.

use super::helpers;
use super::{MetadataFilter, SearchResult, VectorStore};
use crate::vector::sparse::SparseVector;
use crate::vector::types::Vector;
use anyhow::Result;
use serde_json::Value as JsonValue;

use std::collections::HashMap;

impl VectorStore {
    /// Check if sparse index is enabled.
    #[must_use]
    pub fn has_sparse(&self) -> bool {
        self.sparse_index.read().is_some()
    }

    /// Enable sparse vector indexing.
    ///
    /// Called automatically by `set_sparse()` and `set_hybrid_sparse()`.
    /// Call explicitly if you need `has_sparse()` to return true before inserting,
    /// or before calling `sparse_search()` on an empty index.
    pub fn enable_sparse(&mut self) {
        if self.sparse_index.read().is_none() {
            *self.sparse_index.write() = Some(crate::vector::sparse::SparseIndex::new());
        }
    }

    /// Insert or update a sparse vector with metadata.
    ///
    /// The sparse vector is indexed for dot-product search. If the ID already
    /// exists, the old sparse vector is replaced while preserving the existing
    /// dense vector and HNSW slot.
    ///
    /// Automatically enables sparse indexing if not already enabled.
    ///
    /// Note: This only sets the sparse vector. Dense vectors are independent.
    /// Use `set_hybrid_sparse()` to set both dense and sparse at once.
    pub fn set_sparse(
        &mut self,
        id: &str,
        sparse: SparseVector,
        metadata: JsonValue,
    ) -> Result<()> {
        self.enable_sparse();

        let slot = if let Some(slot) = self.records.get_slot(id) {
            // Existing ID: update in-place to preserve HNSW slot
            self.metadata_index.write().remove(slot);
            self.metadata_index.write().index_json(slot, &metadata);
            self.records.update_metadata(slot, metadata.clone())?;

            // WAL write for metadata update
            if let Some(ref storage) = self.storage
                && let Some(record) = self.records.get_by_slot(slot)
            {
                let mut storage = storage.write();
                storage.log_insert(id, record.vector.as_slice(), &metadata)?;
                storage.sync()?;
            }
            slot
        } else {
            // New ID: create record slot with zero-filled placeholder vector
            let dims = self.dimensions();
            let zero_vec = vec![0.0f32; dims];
            let slot =
                self.records
                    .set(id.to_string(), zero_vec.clone(), Some(metadata.clone()))?;
            self.metadata_index.write().index_json(slot, &metadata);

            if let Some(ref storage) = self.storage {
                let mut storage = storage.write();
                storage.log_insert(id, &zero_vec, &metadata)?;
                storage.sync()?;
            }
            slot
        };

        self.sparse_index
            .write()
            .as_mut()
            .expect("sparse_index enabled by enable_sparse")
            .insert(slot, &sparse);
        Ok(())
    }

    /// Insert or update both dense and sparse vectors together.
    ///
    /// This is the recommended method when documents have both representations.
    pub fn set_hybrid_sparse(
        &mut self,
        id: &str,
        dense: Vector,
        sparse: SparseVector,
        metadata: JsonValue,
    ) -> Result<()> {
        self.enable_sparse();

        // Insert dense vector via normal path
        let slot = u32::try_from(self.set(id, dense, metadata.clone())?)
            .map_err(|_| anyhow::anyhow!("slot index exceeds u32::MAX"))?;

        // Index sparse vector
        self.sparse_index
            .write()
            .as_mut()
            .expect("sparse_index enabled by enable_sparse")
            .insert(slot, &sparse);

        Ok(())
    }

    /// Search sparse vectors by dot product similarity.
    ///
    /// Returns results sorted by dot product score (higher = more similar).
    /// The `distance` field contains `-dot_product` to maintain the
    /// lower-is-better convention.
    pub fn sparse_search(
        &self,
        query: &SparseVector,
        k: usize,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let sparse_guard = self.sparse_index.read();
        let sparse_index = sparse_guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Sparse index not enabled. Call enable_sparse() first")
        })?;

        let results = if let Some(f) = filter {
            // Try bitmap filter first
            if let Some(bitmap) = f.evaluate_bitmap(&self.metadata_index.read()) {
                sparse_index.search_with_bitmap(query, k, &bitmap)
            } else {
                // Fall back to post-filtering: over-fetch then filter
                let candidates = sparse_index.search(query, k * 4);
                candidates
                    .into_iter()
                    .filter(|(slot, _)| {
                        self.records.is_live(*slot) && {
                            let metadata = self
                                .records
                                .get_by_slot(*slot)
                                .and_then(|r| r.metadata)
                                .unwrap_or_else(|| helpers::DEFAULT_METADATA.clone());
                            f.matches(&metadata)
                        }
                    })
                    .take(k)
                    .collect()
            }
        } else {
            sparse_index.search(query, k)
        };

        // Convert to SearchResult with -dot_product as distance
        Ok(results
            .into_iter()
            .filter_map(|(slot, score)| {
                let record = self.records.get_by_slot(slot)?;
                let metadata = record
                    .metadata
                    .clone()
                    .unwrap_or_else(helpers::default_metadata);
                Some(SearchResult::new(record.id.clone(), -score, metadata))
            })
            .collect())
    }

    /// Hybrid dense + sparse search with Reciprocal Rank Fusion (RRF).
    ///
    /// Performs both dense and sparse searches, then fuses results using RRF:
    ///   `score(d) = sum(1 / (k + rank_i(d)))` across result lists.
    ///
    /// # Arguments
    ///
    /// * `dense_query` - Dense query vector for HNSW search
    /// * `sparse_query` - Sparse query vector for inverted index search
    /// * `k` - Number of results to return
    /// * `alpha` - Blending factor: 1.0 = pure dense, 0.0 = pure sparse.
    ///   At 0.5, both sources contribute equally.
    /// * `filter` - Optional metadata filter
    pub fn hybrid_sparse_search(
        &self,
        dense_query: &Vector,
        sparse_query: &SparseVector,
        k: usize,
        alpha: f32,
        filter: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        let alpha = alpha.clamp(0.0, 1.0);
        let rrf_k = 60.0f32; // standard RRF constant

        // Fetch more candidates from each source for better fusion
        let fetch_k = k * 4;

        // Dense search
        let dense_results = if alpha > 0.0 {
            self.search_with_options(dense_query, fetch_k, filter, None, None)?
        } else {
            Vec::new()
        };

        // Sparse search
        let sparse_results = if alpha < 1.0 {
            self.sparse_search(sparse_query, fetch_k, filter)?
        } else {
            Vec::new()
        };

        // RRF fusion
        let mut rrf_scores: HashMap<String, (f32, JsonValue)> = HashMap::new();

        // Add dense contributions (weighted by alpha)
        for (rank, result) in dense_results.iter().enumerate() {
            let rrf_score = alpha / (rrf_k + rank as f32 + 1.0);
            rrf_scores
                .entry(result.id.clone())
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert((rrf_score, result.metadata.clone()));
        }

        // Add sparse contributions (weighted by 1 - alpha)
        for (rank, result) in sparse_results.iter().enumerate() {
            let rrf_score = (1.0 - alpha) / (rrf_k + rank as f32 + 1.0);
            rrf_scores
                .entry(result.id.clone())
                .and_modify(|(score, _)| *score += rrf_score)
                .or_insert((rrf_score, result.metadata.clone()));
        }

        // Sort by RRF score (descending) and return top-k
        let mut fused: Vec<SearchResult> = rrf_scores
            .into_iter()
            .map(|(id, (score, metadata))| SearchResult::new(id, -score, metadata))
            .collect();
        fused.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        fused.truncate(k);

        Ok(fused)
    }
}
