//! Text indexing and keyword search using tantivy.
//!
//! Provides BM25 retrieval and utilities for hybrid search fusion.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::{
    LowerCaser, RawTokenizer, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, TokenStream,
    Tokenizer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, doc};

/// Tokenizer presets for different content types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum TokenizerPreset {
    /// Standard natural language tokenizer
    #[default]
    Default,
    /// Optimized for source code (camelCase, snake_case splitting)
    Code,
    /// Raw (exact match only, no splitting)
    Raw,
}

/// Configuration for the text search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchConfig {
    /// Tokenizer preset to use
    pub tokenizer: TokenizerPreset,
    /// Memory buffer for the writer (in MB)
    pub writer_buffer_mb: usize,
}

impl Default for TextSearchConfig {
    fn default() -> Self {
        Self {
            tokenizer: TokenizerPreset::Default,
            writer_buffer_mb: 100,
        }
    }
}

/// Metadata for a text document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocument {
    /// Document identifier (must match VectorStore ID)
    pub id: String,
    /// Text content to index
    pub text: String,
}

/// Search result from the text engine.
#[derive(Debug, Clone)]
pub struct TextSearchResult {
    /// Document identifier
    pub id: String,
    /// BM25 score
    pub score: f32,
}

/// A text search engine powered by Tantivy.
pub struct TextIndex {
    index: Index,
    reader: IndexReader,
    writer: IndexWriter,
    id_field: Field,
    text_field: Field,
}

impl std::fmt::Debug for TextIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextIndex")
            .field("id_field", &self.id_field)
            .field("text_field", &self.text_field)
            .finish_non_exhaustive()
    }
}

impl TextIndex {
    /// Open a text index in memory with default config.
    pub fn open_in_memory() -> Result<Self> {
        Self::open_in_memory_with_config(&TextSearchConfig::default())
    }

    /// Open a text index in memory with custom config.
    pub fn open_in_memory_with_config(config: &TextSearchConfig) -> Result<Self> {
        let schema = Self::build_schema(config.tokenizer);
        let index = Index::create_in_ram(schema);
        Self::new_from_index(index, config)
    }

    /// Open a text index from a directory.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_config(path, &TextSearchConfig::default())
    }

    /// Open a text index from a directory with custom config.
    pub fn open_with_config(path: &Path, config: &TextSearchConfig) -> Result<Self> {
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        let schema = Self::build_schema(config.tokenizer);
        let index = Index::open_or_create(tantivy::directory::MmapDirectory::open(path)?, schema)?;
        Self::new_from_index(index, config)
    }

    fn build_schema(preset: TokenizerPreset) -> Schema {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("id", STRING | STORED);

        let text_indexing = match preset {
            TokenizerPreset::Code => TextFieldIndexing::default()
                .set_tokenizer("code")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            TokenizerPreset::Raw => TextFieldIndexing::default()
                .set_tokenizer("raw")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            TokenizerPreset::Default => TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        };
        let text_options = TextOptions::default()
            .set_indexing_options(text_indexing)
            .set_stored();

        schema_builder.add_text_field("text", text_options);
        schema_builder.build()
    }

    fn new_from_index(index: Index, config: &TextSearchConfig) -> Result<Self> {
        // Register custom tokenizers IF NOT ALREADY REGISTERED (Index may already have them if opened from disk)
        match config.tokenizer {
            TokenizerPreset::Code => {
                let tokenizer = TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(CamelCaseFilter)
                    .filter(LowerCaser)
                    .build();
                index.tokenizers().register("code", tokenizer);
            }
            TokenizerPreset::Raw => {
                index.tokenizers().register("raw", RawTokenizer::default());
            }
            TokenizerPreset::Default => {}
        }

        let schema = index.schema();
        let id_field = schema.get_field("id").unwrap();
        let text_field = schema.get_field("text").unwrap();

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let writer = index.writer(config.writer_buffer_mb * 1024 * 1024)?;

        Ok(Self {
            index,
            reader,
            writer,
            id_field,
            text_field,
        })
    }

    /// Index a document.
    pub fn index_document(&mut self, id: &str, text: &str) -> Result<()> {
        self.writer
            .delete_term(tantivy::Term::from_field_text(self.id_field, id));
        self.writer.add_document(doc!(
            self.id_field => id,
            self.text_field => text,
        ))?;
        Ok(())
    }

    /// Delete a document by ID.
    pub fn delete_document(&mut self, id: &str) -> Result<()> {
        self.writer
            .delete_term(tantivy::Term::from_field_text(self.id_field, id));
        Ok(())
    }

    /// Commit changes to the index.
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Search for documents matching a query.
    pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        if query_text.trim().is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();

        let mut query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        query_parser.set_conjunction_by_default();

        // Tantivy QueryParser uses the field's tokenizer by default.
        // For 'text' field, we configured it to use "code", "raw", or "default" in build_schema.

        let query = query_parser.parse_query(query_text)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            let id = retrieved_doc
                .get_first(self.id_field)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            results.push(TextSearchResult { id, score });
        }

        Ok(results)
    }

    pub fn num_docs(&self) -> usize {
        self.reader.searcher().num_docs() as usize
    }
}

/// A trait for text engines to allow for different implementations (e.g. Tantivy, Meilisearch).
pub trait TextEngine: Send + Sync + std::fmt::Debug {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<TextSearchResult>>;
    fn index_document(&mut self, id: &str, text: &str) -> Result<()>;
    fn delete_document(&mut self, id: &str) -> Result<()>;
    fn commit(&mut self) -> Result<()>;
}

impl TextEngine for TextIndex {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<TextSearchResult>> {
        self.search(query, limit)
    }
    fn index_document(&mut self, id: &str, text: &str) -> Result<()> {
        self.index_document(id, text)
    }
    fn delete_document(&mut self, id: &str) -> Result<()> {
        self.delete_document(id)
    }
    fn commit(&mut self) -> Result<()> {
        self.commit()
    }
}

/// Custom filter for splitting camelCase and snake_case tokens.
#[derive(Clone)]
struct CamelCaseFilter;

impl TokenFilter for CamelCaseFilter {
    type Tokenizer<T: Tokenizer> = CamelCaseTokenizer<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        CamelCaseTokenizer { tail: tokenizer }
    }
}

#[derive(Clone)]
struct CamelCaseTokenizer<T: Tokenizer> {
    tail: T,
}

impl<T: Tokenizer> Tokenizer for CamelCaseTokenizer<T> {
    type TokenStream<'a> = CamelCaseTokenStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        CamelCaseTokenStream {
            tail: self.tail.token_stream(text),
            stack: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

struct CamelCaseTokenStream<'a, T: TokenStream> {
    tail: T,
    stack: Vec<Token>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<T: TokenStream> TokenStream for CamelCaseTokenStream<'_, T> {
    fn advance(&mut self) -> bool {
        if let Some(token) = self.stack.pop() {
            *self.tail.token_mut() = token;
            return true;
        }
        if !self.tail.advance() {
            return false;
        }

        let original_token = self.tail.token().clone();
        let text = &original_token.text;

        let segments = split_camel_case_segments(text);
        if segments.len() > 1 {
            // Push segments in reverse order so they pop in original order
            for (start, end) in segments.into_iter().rev() {
                let mut token = original_token.clone();
                token.text = text[start..end].to_string();
                self.stack.push(token);
            }
            if let Some(token) = self.stack.pop() {
                *self.tail.token_mut() = token;
                return true;
            }
        }
        true
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

fn split_camel_case_segments(text: &str) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return vec![];
    }

    let mut segments = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();

    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let curr = chars[i];
        let next = chars.get(i + 1);

        let boundary = (prev.is_lowercase() && curr.is_uppercase())
            || (prev.is_uppercase()
                && curr.is_uppercase()
                && next.is_some_and(|n| n.is_lowercase()))
            || (curr == '_')
            || (prev == '_');

        if boundary {
            let end = indices[i];
            if start < end {
                segments.push((start, end));
            }
            start = if curr == '_' {
                indices[i] + 1
            } else {
                indices[i]
            };
        }
    }

    if start < text.len() {
        segments.push((start, text.len()));
    }

    segments.retain(|(f, t)| f < t);
    segments
}

/// Default RRF constant (k=60 per Cormack et al. 2009).
pub const DEFAULT_RRF_K: usize = 60;

/// Result from hybrid search with separate keyword and semantic scores.
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub id: String,
    pub score: f32,
    pub keyword_score: Option<f32>,
    pub semantic_score: Option<f32>,
}

/// Reciprocal Rank Fusion for combining vector and text search results.
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
#[must_use]
pub fn weighted_reciprocal_rank_fusion(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<(String, f32)> {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut scores: HashMap<String, f32> =
        HashMap::with_capacity(vector_results.len() + text_results.len());

    for (rank, (id, _distance)) in vector_results.into_iter().enumerate() {
        let rrf_score = 1.0 / (rrf_k + rank + 1) as f32;
        *scores.entry(id).or_default() += alpha * rrf_score;
    }

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
#[must_use]
pub fn weighted_reciprocal_rank_fusion_with_subscores(
    vector_results: Vec<(String, f32)>,
    text_results: Vec<(String, f32)>,
    limit: usize,
    rrf_k: usize,
    alpha: f32,
) -> Vec<HybridResult> {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut rrf_scores: HashMap<String, f32> =
        HashMap::with_capacity(vector_results.len() + text_results.len());
    let mut semantic_scores: HashMap<String, f32> = HashMap::with_capacity(vector_results.len());
    let mut keyword_scores: HashMap<String, f32> = HashMap::with_capacity(text_results.len());

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
