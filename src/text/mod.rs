//! Full-text search using tantivy.
//!
//! Provides BM25-based text search that integrates with `VectorStore`
//! for hybrid (vector + text) search capabilities.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{
    LowerCaser, RawTokenizer, SimpleTokenizer, TextAnalyzer, Token, TokenStream, Tokenizer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, doc};

#[cfg(test)]
mod tests;

/// Result from a text search engine.
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    /// Document ID
    pub id: String,
    /// BM25 matching score
    pub score: f32,
}

impl TextSearchResult {
    pub fn new(id: String, score: f32) -> Self {
        Self { id, score }
    }
}

/// Core trait for full-text search engines.
pub trait TextEngine: Send + Sync {
    /// Index a document with the given ID and text content.
    fn index_document(&mut self, id: &str, text: &str) -> Result<()>;

    /// Delete a document by ID.
    fn delete_document(&mut self, id: &str) -> Result<()>;

    /// Search for documents matching the query.
    fn search(&self, query_str: &str, limit: usize) -> Result<Vec<TextSearchResult>>;

    /// Commit pending changes to the index.
    fn commit(&mut self) -> Result<()>;

    /// Get the number of documents in the index.
    fn num_docs(&self) -> u64;

    /// Flush pending changes (same as commit for now).
    fn flush(&mut self) -> Result<()> {
        self.commit()
    }
}

/// Tokenizer presets for text indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerPreset {
    /// Standard tantivy default (whitespace + lowercase + stemming).
    #[default]
    Default,
    /// Code-aware tokenizer that splits camelCase and HTTPClient-style terms.
    Code,
    /// Raw: no tokenization, exact match only.
    Raw,
}

impl TokenizerPreset {
    #[must_use]
    pub fn schema_tokenizer_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Code => "code",
            Self::Raw => "raw",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "code" => Ok(Self::Code),
            "raw" => Ok(Self::Raw),
            other => Err(anyhow!("Unknown tokenizer preset: {other}")),
        }
    }
}

/// Configuration for text search functionality.
///
/// # Example
/// ```ignore
/// // Default: 50MB buffer (good for most use cases), default tokenizer
/// let config = TextSearchConfig::default();
///
/// // Mobile/constrained: reduce buffer, code-aware tokenizer
/// let config = TextSearchConfig {
///     writer_buffer_mb: 15,
///     tokenizer: TokenizerPreset::Code,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchConfig {
    /// Writer buffer size in MB (default: 50).
    ///
    /// Larger buffers reduce segment merge frequency but use more memory.
    /// - 15MB: Mobile/constrained environments
    /// - 50MB: Default, good for laptops/servers/desktop apps
    /// - 100-200MB: High-throughput server workloads
    pub writer_buffer_mb: usize,

    /// Tokenizer to use for text indexing (default: Default).
    pub tokenizer: TokenizerPreset,
}

impl Default for TextSearchConfig {
    fn default() -> Self {
        Self {
            writer_buffer_mb: 50,
            tokenizer: TokenizerPreset::Default,
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

impl TextEngine for TextIndex {
    fn index_document(&mut self, id: &str, text: &str) -> Result<()> {
        self.index_document(id, text)
    }

    fn delete_document(&mut self, id: &str) -> Result<()> {
        self.delete_document(id)
    }

    fn search(&self, query_str: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        self.search(query_str, limit)
    }

    fn commit(&mut self) -> Result<()> {
        self.commit()
    }

    fn num_docs(&self) -> u64 {
        self.num_docs()
    }
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

        let schema = Self::create_schema(config.tokenizer.schema_tokenizer_name());
        let id_field = schema.get_field("id").expect("id field exists");
        let text_field = schema.get_field("text").expect("text field exists");

        // Try to open existing index, or create new one
        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema.clone())?
        };

        // Register custom tokenizers
        Self::register_tokenizers(&index);

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
        let schema = Self::create_schema(config.tokenizer.schema_tokenizer_name());
        let id_field = schema.get_field("id").expect("id field exists");
        let text_field = schema.get_field("text").expect("text field exists");

        let index = Index::create_in_ram(schema);

        // Register custom tokenizers
        Self::register_tokenizers(&index);

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

    fn create_schema(tokenizer_name: &str) -> Schema {
        let mut builder = Schema::builder();

        // Document ID - stored for retrieval, indexed as exact match
        builder.add_text_field("id", STRING | STORED);

        // Text content - indexed for full-text search with BM25
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer(tokenizer_name)
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default()
            .set_indexing_options(text_indexing)
            .set_stored();
        builder.add_text_field("text", text_options);

        builder.build()
    }

    fn register_tokenizers(index: &Index) {
        // "code" tokenizer: SimpleTokenizer -> CamelCaseFilter -> LowerCaser
        let code_tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(CamelCaseFilter)
            .filter(LowerCaser)
            .build();
        index.tokenizers().register("code", code_tokenizer);

        // "raw" tokenizer: RawTokenizer
        index.tokenizers().register(
            "raw",
            TextAnalyzer::builder(RawTokenizer::default()).build(),
        );
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
    /// Returns a vector of [`TextSearchResult`], sorted by score descending.
    ///
    /// # Arguments
    /// * `query_str` - The search query (supports tantivy query syntax)
    /// * `limit` - Maximum number of results to return
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        if query_str.trim().is_empty() {
            return Ok(vec![]);
        }

        let searcher = self.reader.searcher();

        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let query = query_parser
            .parse_query(query_str)
            .map_err(|e| anyhow!("Invalid query: {e}"))?;

        let top_docs: Vec<(f32, tantivy::DocAddress)> =
            searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        let results = top_docs
            .into_iter()
            .filter_map(|(score, doc_addr)| {
                let doc: TantivyDocument = searcher.doc(doc_addr).ok()?;
                let id = doc.get_first(self.id_field)?.as_str()?.to_string();
                Some(TextSearchResult::new(id, score))
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

/// Token filter that splits camelCase and HTTPClient-style identifiers.
#[derive(Clone)]
pub struct CamelCaseFilter;

impl tantivy::tokenizer::TokenFilter for CamelCaseFilter {
    type Tokenizer<T: Tokenizer> = CamelCaseTokenizer<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CamelCaseTokenizer {
            tokenizer,
            parts: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct CamelCaseTokenizer<T> {
    tokenizer: T,
    parts: Vec<Token>,
}

impl<T: Tokenizer> Tokenizer for CamelCaseTokenizer<T> {
    type TokenStream<'a> = CamelCaseTokenStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.parts.clear();
        CamelCaseTokenStream {
            tail: self.tokenizer.token_stream(text),
            parts: &mut self.parts,
        }
    }
}

pub struct CamelCaseTokenStream<'a, T> {
    parts: &'a mut Vec<Token>,
    tail: T,
}

impl<T: TokenStream> CamelCaseTokenStream<'_, T> {
    fn split_current_token(&mut self) {
        let token = self.tail.token();
        let segments = split_camel_case_segments(&token.text);
        if segments.len() <= 1 {
            return;
        }

        for (start, end) in segments.into_iter().rev() {
            self.parts.push(Token {
                text: token.text[start..end].to_string(),
                offset_from: token.offset_from + start,
                offset_to: token.offset_from + end,
                ..*token
            });
        }
    }
}

impl<T: TokenStream> TokenStream for CamelCaseTokenStream<'_, T> {
    fn advance(&mut self) -> bool {
        self.parts.pop();
        if !self.parts.is_empty() {
            return true;
        }
        if !self.tail.advance() {
            return false;
        }
        self.split_current_token();
        true
    }

    fn token(&self) -> &Token {
        self.parts.last().unwrap_or_else(|| self.tail.token())
    }

    fn token_mut(&mut self) -> &mut Token {
        self.parts
            .last_mut()
            .unwrap_or_else(|| self.tail.token_mut())
    }
}

fn split_camel_case_segments(text: &str) -> Vec<(usize, usize)> {
    if text.len() <= 1 {
        return vec![(0, text.len())];
    }

    let mut chars = text.char_indices().peekable();
    let mut segments = Vec::new();
    let mut start = 0usize;

    let mut prev = match chars.next() {
        Some((_, ch)) => ch,
        None => return vec![],
    };

    while let Some((idx, curr)) = chars.next() {
        let next = chars.peek().map(|(_, ch)| *ch);
        let boundary = (prev.is_lowercase() || prev.is_ascii_digit()) && curr.is_uppercase()
            || (prev.is_uppercase() && curr.is_uppercase() && next.is_some_and(char::is_lowercase));

        if boundary {
            segments.push((start, idx));
            start = idx;
        }

        prev = curr;
    }

    segments.push((start, text.len()));
    segments.retain(|(from, to)| from < to);
    segments
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

/// Weighted Reciprocal Rank Fusion for combining vector and text search results.
///
/// Allows biasing results towards either vector or keyword search.
///
/// # Arguments
/// * `vector_results` - Results from vector search as (id, distance)
/// * `text_results` - Results from text search as (id, score)
/// * `limit` - Maximum results to return
/// * `rrf_k` - RRF constant (default: 60)
/// * `alpha` - Weight for vector results (0.0 = text only, 1.0 = vector only, 0.5 = balanced)
#[must_use]
pub fn weighted_reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let alpha = alpha.clamp(0.0, 1.0);

    let capacity = vector_results.len() + text_results.len();
    let mut scores: HashMap<String, f32> = HashMap::with_capacity(capacity);

    // Consume owned strings to avoid cloning
    for (rank, (id, _distance)) in vector_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id).or_default() += alpha * rrf_score;
    }

    // Add text search contributions — consume owned strings to avoid cloning
    for (rank, (id, _score)) in text_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id).or_default() += (1.0 - alpha) * rrf_score;
    }

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

    let capacity = vector_results.len() + text_results.len();
    let mut rrf_scores: HashMap<String, f32> = HashMap::with_capacity(capacity);
    let mut semantic_scores: HashMap<String, f32> = HashMap::with_capacity(vector_results.len());
    let mut keyword_scores: HashMap<String, f32> = HashMap::with_capacity(text_results.len());

    // Consume owned strings to avoid cloning
    for (rank, (id, distance)) in vector_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        semantic_scores.insert(id.clone(), distance);
        *rrf_scores.entry(id).or_default() += alpha * rrf_score;
    }

    for (rank, (id, bm25_score)) in text_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        keyword_scores.insert(id.clone(), bm25_score);
        *rrf_scores.entry(id).or_default() += (1.0 - alpha) * rrf_score;
    }

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
