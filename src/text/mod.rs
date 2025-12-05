//! Full-text search using tantivy.
//!
//! Provides BM25-based text search that integrates with VectorStore
//! for hybrid (vector + text) search capabilities.

use anyhow::{anyhow, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

#[cfg(test)]
mod tests;

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
    /// Create or open a text index at the given path.
    ///
    /// # Arguments
    /// * `path` - Directory for the tantivy index
    ///
    /// # Example
    /// ```no_run
    /// use omendb::text::TextIndex;
    /// let index = TextIndex::open("./text_index").unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
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

        // 50MB writer buffer (smaller than default for embedded use)
        let writer = index.writer(50_000_000)?;

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

    /// Create an in-memory text index (for testing or temporary use).
    pub fn open_in_memory() -> Result<Self> {
        let schema = Self::create_schema();
        let id_field = schema.get_field("id").expect("id field exists");
        let text_field = schema.get_field("text").expect("text field exists");

        let index = Index::create_in_ram(schema);

        let writer = index.writer(50_000_000)?;

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
    /// Returns a vector of (document_id, BM25_score) tuples, sorted by score descending.
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
            .map_err(|e| anyhow!("Invalid query: {}", e))?;

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
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Get a reference to the underlying tantivy index.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get a reference to the index reader.
    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }
}

/// Reciprocal Rank Fusion for combining vector and text search results.
///
/// RRF combines rankings from multiple sources without requiring score normalization.
/// Formula: `score(d) = Σ 1 / (k + rank_i(d))` where k is typically 60.
///
/// # Arguments
/// * `vector_results` - Results from vector search as (id, distance)
/// * `text_results` - Results from text search as (id, BM25_score)
/// * `k` - Maximum results to return
/// * `rrf_k` - RRF constant (default: 60)
///
/// # Returns
/// Combined results as (id, RRF_score) sorted by score descending.
pub fn reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    k: usize,
    rrf_k: usize,
) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, f32> = HashMap::new();

    // Add vector search contributions (lower distance = higher rank)
    // Results are already sorted by distance ascending
    for (rank, (id, _distance)) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id.clone()).or_default() += rrf_score;
    }

    // Add text search contributions (higher BM25 = higher rank)
    // Results are already sorted by score descending
    for (rank, (id, _score)) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id.clone()).or_default() += rrf_score;
    }

    // Sort by RRF score descending
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);

    results
}
