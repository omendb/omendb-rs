//! Full-text search using tantivy.
//!
//! Provides BM25-based text search that integrates with `VectorStore`
//! for hybrid (vector + text) search capabilities.

use anyhow::{anyhow, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

#[cfg(test)]
mod tests;

/// Configuration for text search functionality.
///
/// # Example
/// ```ignore
/// // Default: 50MB buffer (good for most use cases)
/// let config = TextSearchConfig::default();
///
/// // Mobile/constrained: reduce buffer
/// let config = TextSearchConfig { writer_buffer_mb: 15 };
///
/// // Cloud/high-throughput: increase buffer
/// let config = TextSearchConfig { writer_buffer_mb: 200 };
/// ```
#[derive(Debug, Clone)]
pub struct TextSearchConfig {
    /// Writer buffer size in MB (default: 50).
    ///
    /// Larger buffers reduce segment merge frequency but use more memory.
    /// - 15MB: Mobile/constrained environments
    /// - 50MB: Default, good for laptops/servers/desktop apps
    /// - 100-200MB: High-throughput server workloads
    pub writer_buffer_mb: usize,
}

impl Default for TextSearchConfig {
    fn default() -> Self {
        Self {
            writer_buffer_mb: 50,
        }
    }
}

/// Full-text search index backed by tantivy.
///
/// Provides BM25 scoring for text search, designed to work alongside
/// HNSW vector search for hybrid retrieval.
pub struct TextIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    id_field: Field,
    text_field: Field,
}

impl TextIndex {
    /// Create or open a text index at the given path with default config.
    ///
    /// # Example
    /// ```no_run
    /// use omendb::text::TextIndex;
    /// let index = TextIndex::open("./text_index").unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_config(path, &TextSearchConfig::default())
    }

    /// Create or open a text index with custom configuration.
    ///
    /// # Example
    /// ```no_run
    /// use omendb::text::{TextIndex, TextSearchConfig};
    /// let config = TextSearchConfig { writer_buffer_mb: 100 };
    /// let index = TextIndex::open_with_config("./text_index", &config).unwrap();
    /// ```
    pub fn open_with_config<P: AsRef<Path>>(path: P, config: &TextSearchConfig) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let schema = Self::create_schema();
        let id_field = schema.get_field("id").expect("id field exists");
        let text_field = schema.get_field("text").expect("text field exists");

        // Try to open existing index, or create new one
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema.clone())?
        };

        let buffer_bytes = config.writer_buffer_mb * 1_000_000;
        let writer = index.writer(buffer_bytes)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            writer,
            reader,
            id_field,
            text_field,
        })
    }

    /// Create an in-memory text index with default config.
    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_config(&TextSearchConfig::default())
    }

    /// Create an in-memory text index with custom configuration.
    pub fn open_in_memory_with_config(config: &TextSearchConfig) -> Result<Self> {
        let schema = Self::create_schema();
        let id_field = schema.get_field("id").expect("id field exists");
        let text_field = schema.get_field("text").expect("text field exists");

        let index = Index::create_in_ram(schema);

        let buffer_bytes = config.writer_buffer_mb * 1_000_000;
        let writer = index.writer(buffer_bytes)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            writer,
            reader,
            id_field,
            text_field,
        })
    }

    fn create_schema() -> Schema {
        let mut builder = Schema::builder();

        // Document ID - stored for retrieval, indexed as exact match
        builder.add_text_field("id", STRING | STORED);

        // Text content - indexed for full-text search with BM25
        builder.add_text_field("text", TEXT);

        builder.build()
    }

    /// Index a document with the given ID and text content.
    ///
    /// If a document with this ID already exists, it will be updated.
    pub fn index_document(&mut self, id: &str, text: &str) -> Result<()> {
        // Delete existing document with this ID (if any)
        self.delete_document(id)?;

        self.writer.add_document(doc!(
            self.id_field => id,
            self.text_field => text,
        ))?;

        Ok(())
    }

    /// Delete a document by ID.
    pub fn delete_document(&mut self, id: &str) -> Result<()> {
        let term = tantivy::Term::from_field_text(self.id_field, id);
        self.writer.delete_term(term);
        Ok(())
    }

    /// Commit pending changes to the index.
    ///
    /// Changes are not visible to searchers until commit is called.
    /// This also reloads the reader to see the new changes immediately.
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        // Reload reader to see committed changes immediately
        self.reader.reload()?;
        Ok(())
    }

    /// Search for documents matching the query.
    ///
    /// Returns a vector of (id, score) tuples, sorted by score descending.
    ///
    /// # Arguments
    /// * `query_str` - The search query (supports tantivy query syntax)
    /// * `limit` - Maximum number of results to return
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        if query_str.trim().is_empty() {
            return Ok(vec![]);
        }

        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| anyhow!("Invalid query: {e}"))?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let results = top_docs
            .into_iter()
            .filter_map(|(score, doc_addr)| {
                let doc: TantivyDocument = searcher.doc(doc_addr).ok()?;
                let id = doc.get_first(self.id_field)?.as_str()?.to_string();
                Some((id, score))
            })
            .collect();

        Ok(results)
    }

    /// Get the number of documents in the index.
    #[must_use]
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Get a reference to the underlying tantivy index.
    #[must_use]
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get a reference to the index reader.
    #[must_use]
    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }
}

/// Default RRF constant (k=60 per Cormack et al. 2009).
pub const DEFAULT_RRF_K: usize = 60;

/// Result from hybrid search with separate keyword and semantic scores.
///
/// Useful for debugging, custom weighting, or query-adaptive fusion.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Document ID
    pub id: String,
    /// Combined RRF score
    pub score: f32,
    /// BM25 keyword matching score (None if document only matched vector search)
    pub keyword_score: Option<f32>,
    /// Vector similarity score (None if document only matched text search)
    pub semantic_score: Option<f32>,
}

/// Reciprocal Rank Fusion for combining vector and text search results.
///
/// RRF combines rankings from multiple sources without requiring score normalization.
/// Formula: `score(d) = Σ 1 / (k + rank_i(d))` where k is typically 60.
///
/// # Arguments
/// * `vector_results` - Results from vector search as (id, distance)
/// * `text_results` - Results from text search as (id, score)
/// * `limit` - Maximum results to return
/// * `rrf_k` - RRF constant (default: 60)
///
/// # Returns
/// Combined results as (id, score) sorted by score descending.
#[must_use]
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
) -> Vec<(String, f32)> {
    weighted_reciprocal_rank_fusion(vector_results, text_results, limit, rrf_k, 0.5)
}

/// Weighted Reciprocal Rank Fusion for hybrid search with tunable balance.
///
/// Like [`reciprocal_rank_fusion`], but allows weighting vector vs text results.
///
/// # Arguments
/// * `vector_results` - Results from vector search as (id, distance)
/// * `text_results` - Results from text search as (id, score)
/// * `limit` - Maximum results to return
/// * `rrf_k` - RRF constant (default: 60)
/// * `alpha` - Weight for vector results (0.0 = text only, 1.0 = vector only, 0.5 = balanced)
///
/// # Example
/// ```ignore
/// // 70% vector, 30% text
/// let results = weighted_reciprocal_rank_fusion(vec_results, text_results, 10, 60, 0.7);
/// ```
#[must_use]
pub fn weighted_reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    // Clamp alpha to valid range
    let alpha = alpha.clamp(0.0, 1.0);

    let mut scores: HashMap<String, f32> = HashMap::new();

    // Add vector search contributions (lower distance = higher rank)
    // Results are already sorted by distance ascending
    for (rank, (id, _distance)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id.clone()).or_default() += alpha * rrf_score;
    }

    // Add text search contributions (higher BM25 = higher rank)
    // Results are already sorted by score descending
    for (rank, (id, _score)) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id.clone()).or_default() += (1.0 - alpha) * rrf_score;
    }

    // Sort by RRF score descending
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results.truncate(limit);

    results
}

/// Weighted RRF with separate keyword and semantic scores returned.
///
/// Returns [`HybridResult`] with raw BM25 and vector distance scores
/// for transparency and custom post-processing.
///
/// # Arguments
/// * `vector_results` - Results from vector search as (id, distance)
/// * `text_results` - Results from text search as (id, bm25_score)
/// * `limit` - Maximum results to return
/// * `rrf_k` - RRF constant (default: 60)
/// * `alpha` - Weight for vector results (0.0 = text only, 1.0 = vector only, 0.5 = balanced)
#[must_use]
pub fn weighted_reciprocal_rank_fusion_with_subscores(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<HybridResult> {
    use std::collections::HashMap;

    let alpha = alpha.clamp(0.0, 1.0);

    // Track RRF scores and raw scores separately
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    let mut semantic_scores: HashMap<String, f32> = HashMap::new();
    let mut keyword_scores: HashMap<String, f32> = HashMap::new();

    // Vector results: store distance as semantic_score
    for (rank, (id, distance)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *rrf_scores.entry(id.clone()).or_default() += alpha * rrf_score;
        semantic_scores.insert(id.clone(), *distance);
    }

    // Text results: store BM25 as keyword_score
    for (rank, (id, bm25_score)) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *rrf_scores.entry(id.clone()).or_default() += (1.0 - alpha) * rrf_score;
        keyword_scores.insert(id.clone(), *bm25_score);
    }

    // Build results with subscores
    let mut results: Vec<HybridResult> = rrf_scores
        .into_iter()
        .map(|(id, score)| HybridResult {
            keyword_score: keyword_scores.get(&id).copied(),
            semantic_score: semantic_scores.get(&id).copied(),
            id,
            score,
        })
        .collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
    results.truncate(limit);

    results
}
