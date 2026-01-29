//! Multi-vector configuration.

/// Configuration for multi-vector (ColBERT-style) document storage.
///
/// Multi-vector stores encode documents as sets of token embeddings, enabling
/// late-interaction retrieval patterns like ColBERT's MaxSim scoring.
///
/// # Example
///
/// ```rust
/// use omendb::MultiVectorConfig;
///
/// // Use defaults (good for most cases)
/// let config = MultiVectorConfig::default();
///
/// // Customize for higher quality
/// let config = MultiVectorConfig {
///     repetitions: 10,
///     ..Default::default()
/// };
///
/// // Disable projection for backwards compatibility
/// let config = MultiVectorConfig {
///     d_proj: None,
///     ..Default::default()
/// };
/// ```
///
/// # Parameters
///
/// - `repetitions`: Number of independent hash functions. Higher = better quality,
///   larger index. Default: 8, range: 3-40.
///
/// - `partition_bits`: Log2 of bucket count per repetition. Default: 4 (16 buckets).
///   Higher values give finer granularity but larger encodings. The MUVERA paper
///   uses 5-6 bits (32-64 buckets) for best quality.
///
/// - `d_proj`: Dimension to project tokens to before encoding. Default: Some(16).
///   Set to None to use full token dimension (backwards compatible but 8x larger).
///   Weaviate and Qdrant both use d_proj=16.
///
/// # Encoded Dimension
///
/// The encoded vector size is: `repetitions × 2^partition_bits × proj_dim`
/// where proj_dim = d_proj (if set) or token_dim (if None).
///
/// | Preset  | Config         | Encoded Size (128D tokens) |
/// |---------|----------------|---------------------------|
/// | fast    | (5, 3, 16)     | 640                       |
/// | default | (8, 4, 16)     | 2,048                     |
/// | quality | (10, 4, 32)    | 5,120                     |
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Number of independent hash repetitions. Higher = better quality, larger index.
    /// Default: 8, range: 3-40.
    pub repetitions: u8,

    /// Log2 of partition count (2^partition_bits buckets per repetition).
    /// Default: 4 (16 partitions), range: 3-6.
    pub partition_bits: u8,

    /// Dimension to project tokens to before FDE encoding.
    /// Default: Some(16). Set to None to use full token dimension.
    /// Must be <= token_dim when set.
    pub d_proj: Option<u8>,

    /// Random seed for reproducible encoding. Default: 42.
    pub seed: u64,

    /// Token pooling factor. Reduces tokens by this factor before encoding.
    /// - None: No pooling (default)
    /// - Some(2): 50% reduction, 100.6% quality (recommended)
    /// - Some(3): 66% reduction, 99% quality
    /// - Some(4): 75% reduction, 97% quality
    pub pool_factor: Option<u8>,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            repetitions: 8,
            partition_bits: 4,
            d_proj: Some(16),
            seed: 42,
            pool_factor: None,
        }
    }
}

impl MultiVectorConfig {
    /// Fast configuration - smaller encoding, faster search.
    ///
    /// Good for prototyping. Use reranking to maintain quality.
    /// Encoded size: 5 × 8 × 16 = 640
    #[must_use]
    pub fn fast() -> Self {
        Self {
            repetitions: 5,
            partition_bits: 3,
            d_proj: Some(16),
            seed: 42,
            pool_factor: None,
        }
    }

    /// Compact configuration - 50% token storage reduction with 100.6% quality.
    ///
    /// Uses hierarchical clustering to pool similar tokens before encoding.
    /// Recommended for production when storage matters.
    #[must_use]
    pub fn compact() -> Self {
        Self {
            pool_factor: Some(2),
            ..Default::default()
        }
    }

    /// Quality configuration - larger encoding, better approximation.
    ///
    /// Use for production when recall matters.
    /// Encoded size: 10 × 16 × 32 = 5,120
    #[must_use]
    pub fn quality() -> Self {
        Self {
            repetitions: 10,
            partition_bits: 4,
            d_proj: Some(32),
            seed: 42,
            pool_factor: None,
        }
    }

    /// Calculate the encoded vector dimension for a given token dimension.
    ///
    /// Returns `repetitions × partitions × proj_dim` where proj_dim is
    /// d_proj (if set) or token_dim (if None).
    #[must_use]
    pub fn encoded_dimension(&self, token_dim: usize) -> usize {
        let proj_dim = self.d_proj.map_or(token_dim, |d| d as usize);
        self.repetitions as usize * self.partitions() * proj_dim
    }

    /// Get the projection dimension for a given token dimension.
    ///
    /// Returns d_proj if set, otherwise token_dim.
    #[must_use]
    pub fn proj_dim(&self, token_dim: usize) -> usize {
        self.d_proj.map_or(token_dim, |d| d as usize)
    }

    /// Number of partitions per repetition (2^partition_bits).
    #[must_use]
    pub fn partitions(&self) -> usize {
        1 << self.partition_bits
    }
}

// Keep old name as alias for internal migration
pub type MuveraConfig = MultiVectorConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MultiVectorConfig::default();
        assert_eq!(config.repetitions, 8);
        assert_eq!(config.partition_bits, 4);
        assert_eq!(config.d_proj, Some(16));
        assert_eq!(config.pool_factor, None);
        assert_eq!(config.partitions(), 16);
    }

    #[test]
    fn test_encoded_dimension_with_dproj() {
        let config = MultiVectorConfig::default();
        // 8 reps * 16 partitions * 16 (d_proj) = 2,048
        assert_eq!(config.encoded_dimension(128), 2048);

        // Quality config: 10 reps * 16 partitions * 32 (d_proj) = 5,120
        let config = MultiVectorConfig::quality();
        assert_eq!(config.encoded_dimension(128), 5120);
    }

    #[test]
    fn test_encoded_dimension_no_dproj() {
        let config = MultiVectorConfig {
            d_proj: None,
            ..Default::default()
        };
        // 8 reps * 16 partitions * 128 (token_dim) = 16,384
        assert_eq!(config.encoded_dimension(128), 16384);
    }

    #[test]
    fn test_proj_dim() {
        let config = MultiVectorConfig::default();
        assert_eq!(config.proj_dim(128), 16);

        let config = MultiVectorConfig {
            d_proj: None,
            ..Default::default()
        };
        assert_eq!(config.proj_dim(128), 128);

        let config = MultiVectorConfig {
            d_proj: Some(32),
            ..Default::default()
        };
        assert_eq!(config.proj_dim(128), 32);
    }

    #[test]
    fn test_struct_init() {
        let config = MultiVectorConfig {
            repetitions: 20,
            partition_bits: 5,
            d_proj: Some(24),
            seed: 123,
            pool_factor: Some(2),
        };
        assert_eq!(config.repetitions, 20);
        assert_eq!(config.partitions(), 32);
        assert_eq!(config.d_proj, Some(24));
        assert_eq!(config.seed, 123);
        assert_eq!(config.pool_factor, Some(2));
    }

    #[test]
    fn test_fast_preset() {
        let config = MultiVectorConfig::fast();
        assert_eq!(config.repetitions, 5);
        assert_eq!(config.partition_bits, 3);
        assert_eq!(config.d_proj, Some(16));
        assert_eq!(config.pool_factor, None);
        assert_eq!(config.partitions(), 8);
        // 5 * 8 * 16 = 640
        assert_eq!(config.encoded_dimension(128), 640);
    }

    #[test]
    fn test_quality_preset() {
        let config = MultiVectorConfig::quality();
        assert_eq!(config.repetitions, 10);
        assert_eq!(config.partition_bits, 4);
        assert_eq!(config.d_proj, Some(32));
        assert_eq!(config.pool_factor, None);
        assert_eq!(config.partitions(), 16);
        // 10 * 16 * 32 = 5,120
        assert_eq!(config.encoded_dimension(128), 5120);
    }

    #[test]
    fn test_compact_preset() {
        let config = MultiVectorConfig::compact();
        assert_eq!(config.repetitions, 8);
        assert_eq!(config.partition_bits, 4);
        assert_eq!(config.d_proj, Some(16));
        assert_eq!(config.pool_factor, Some(2));
        // Same as default encoded dimension
        assert_eq!(config.encoded_dimension(128), 2048);
    }
}
