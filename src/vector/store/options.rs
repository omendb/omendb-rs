//! `VectorStore` builder pattern configuration
//!
//! Follows `std::fs::OpenOptions` pattern for familiar, ergonomic API.

use super::VectorStore;
use crate::omen::Metric;
use crate::text::TextSearchConfig;
use anyhow::Result;
use std::path::Path;

/// Configuration options for opening or creating a vector store.
///
/// Follows the `std::fs::OpenOptions` pattern for familiar, ergonomic API.
///
/// # Examples
///
/// ```rust,no_run
/// use omendb::vector::store::VectorStoreOptions;
///
/// // Simple persistent store
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .open("./vectors")?;
///
/// // With custom HNSW parameters
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .m(32)
///     .ef_construction(400)
///     .ef_search(100)
///     .open("./vectors")?;
///
/// // In-memory store
/// let store = VectorStoreOptions::default()
///     .dimensions(384)
///     .build()?;
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct VectorStoreOptions {
    /// Vector dimensionality (0 = infer from first insert or existing data)
    pub(super) dimensions: usize,

    /// HNSW M parameter: neighbors per node (default: 16)
    pub(super) m: Option<usize>,

    /// HNSW `ef_construction`: build quality (default: 100)
    pub(super) ef_construction: Option<usize>,

    /// HNSW `ef_search`: search quality/speed tradeoff (default: 100)
    pub(super) ef_search: Option<usize>,

    /// SQ8 quantization enabled (true = 4x compression, false = full precision)
    pub(super) quantization: bool,

    /// Distance metric for similarity search (default: L2)
    pub(super) metric: Option<Metric>,

    /// Text search configuration (None = disabled)
    pub(super) text_search_config: Option<TextSearchConfig>,

    /// SQ8 refiner: rescore with fp32 after quantized search (default: true when quantized)
    pub(super) rescore: Option<bool>,
    /// SQ8 refiner: candidate multiplier for rescoring (default: 3.0)
    pub(super) oversample: Option<f32>,

    /// Memory limit in bytes (None = unlimited).
    /// When estimated memory exceeds this, the mutable segment is frozen
    /// so the OS can page it via mmap.
    pub(super) max_memory_bytes: Option<usize>,
}

impl VectorStoreOptions {
    /// Create new options with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set vector dimensionality.
    ///
    /// If not set, dimensions will be inferred from:
    /// 1. Existing data when opening a persistent store
    /// 2. First inserted vector
    #[must_use]
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.dimensions = dim;
        self
    }

    /// Set HNSW M parameter (neighbors per node).
    ///
    /// Higher M = better recall, more memory. Range: 4-64, default: 16.
    #[must_use]
    pub fn m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Set HNSW `ef_construction` (build quality).
    ///
    /// Higher = better graph quality, slower build. Default: 100.
    #[must_use]
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = Some(ef);
        self
    }

    /// Set HNSW `ef_search` (search quality/speed tradeoff).
    ///
    /// Higher = better recall, slower search. Default: 100.
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Enable SQ8 quantization for memory-efficient storage.
    ///
    /// SQ8: 4x compression, similar speed, ~99% recall.
    ///
    /// # Example
    /// ```ignore
    /// let store = VectorStoreOptions::default()
    ///     .dimensions(768)
    ///     .quantization(true)
    ///     .open("./vectors")?;
    /// ```
    #[must_use]
    pub fn quantization(mut self, enabled: bool) -> Self {
        self.quantization = enabled;
        self
    }

    /// Set distance metric for similarity search.
    ///
    /// # Metrics
    /// - `"l2"` or `"euclidean"`: Euclidean distance (default)
    /// - `"cosine"`: Cosine distance (1 - cosine similarity)
    /// - `"dot"` or `"ip"`: Inner product (for MIPS)
    ///
    /// # Errors
    /// Returns error if metric string is not recognized.
    ///
    /// # Example
    /// ```ignore
    /// let store = VectorStoreOptions::default()
    ///     .dimensions(768)
    ///     .metric("cosine")?
    ///     .open("./vectors")?;
    /// ```
    pub fn metric(mut self, m: &str) -> Result<Self, String> {
        self.metric = Some(Metric::parse(m)?);
        Ok(self)
    }

    /// Set distance metric directly (no parsing).
    #[must_use]
    pub fn metric_fn(mut self, m: Metric) -> Self {
        self.metric = Some(m);
        self
    }

    /// Enable tantivy-based full-text search with default configuration.
    ///
    /// When enabled, you can use `set_with_text()` to index text alongside vectors,
    /// and `search_hybrid()` to search both with RRF fusion.
    ///
    /// Uses 50MB writer buffer by default. For custom memory settings,
    /// use `text_search_config()` instead.
    #[must_use]
    pub fn text_search(mut self, enabled: bool) -> Self {
        self.text_search_config = if enabled {
            Some(TextSearchConfig::default())
        } else {
            None
        };
        self
    }

    /// Enable text search with custom configuration.
    ///
    /// # Example
    /// ```ignore
    /// // Mobile: lower memory
    /// let store = VectorStoreOptions::default()
    ///     .text_search_config(TextSearchConfig { writer_buffer_mb: 15 })
    ///     .open("./db")?;
    ///
    /// // Cloud: higher throughput
    /// let store = VectorStoreOptions::default()
    ///     .text_search_config(TextSearchConfig { writer_buffer_mb: 200 })
    ///     .open("./db")?;
    /// ```
    #[must_use]
    pub fn text_search_config(mut self, config: TextSearchConfig) -> Self {
        self.text_search_config = Some(config);
        self
    }

    /// Enable or disable SQ8 rescoring.
    ///
    /// When enabled (default for quantized stores), search fetches extra candidates
    /// using the quantized index, then rescores them with original fp32 vectors.
    /// Improves recall from ~97% to ~99%+ with minimal speed impact.
    #[must_use]
    pub fn rescore(mut self, enabled: bool) -> Self {
        self.rescore = Some(enabled);
        self
    }

    /// Set candidate oversample multiplier for rescoring.
    ///
    /// During rescoring, `k * oversample` candidates are fetched from the
    /// quantized index, then the top k are selected after fp32 recomputation.
    /// Default: 3.0 (fetch 3x candidates).
    #[must_use]
    pub fn oversample(mut self, factor: f32) -> Self {
        self.oversample = Some(factor);
        self
    }

    /// Set memory limit in bytes.
    ///
    /// When estimated memory usage exceeds this limit, the mutable segment is
    /// frozen (triggering mmap for the frozen data). This bounds heap usage
    /// while allowing the OS to manage paging of older segments.
    ///
    /// Set to `None` (default) for unlimited memory.
    #[must_use]
    pub fn max_memory_bytes(mut self, limit: Option<usize>) -> Self {
        self.max_memory_bytes = limit;
        self
    }

    /// Open or create a persistent vector store at the given path.
    ///
    /// Creates the directory if it doesn't exist.
    /// Loads existing data if the store already exists.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<VectorStore> {
        VectorStore::open_with_options(path, self)
    }

    /// Build an in-memory vector store (no persistence).
    pub fn build(&self) -> Result<VectorStore> {
        VectorStore::build_with_options(self)
    }
}
