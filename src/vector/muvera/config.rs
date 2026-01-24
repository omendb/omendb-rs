//! MUVERA configuration.

/// Configuration for MUVERA encoding.
///
/// Controls the FDE (Fixed Dimensional Encoding) generation parameters.
///
/// # Parameters
///
/// - `k_sim`: Number of SimHash hyperplanes. Creates 2^k_sim buckets. Higher values
///   give finer-grained partitioning but larger FDEs. Range: 3-8, default: 3.
///
/// - `r_reps`: Number of independent repetitions concatenated in the FDE. Primary
///   quality lever - more repetitions = better approximation. Range: 5-40, default: 5.
///
/// - `seed`: Random seed for reproducible hyperplane generation. Default: 42.
///
/// # FDE Dimension
///
/// FDE dimension = r_reps × 2^k_sim × token_dim
///
/// | Config (k_sim, r_reps) | FDE Dim (128D tokens) |
/// |------------------------|-----------------------|
/// | (3, 5)                 | 5,120                 |
/// | (3, 10)                | 10,240                |
/// | (4, 5)                 | 10,240                |
/// | (4, 10)                | 20,480                |
#[derive(Debug, Clone)]
pub struct MuveraConfig {
    /// Number of SimHash hyperplanes (creates 2^k_sim buckets).
    pub k_sim: u8,
    /// Number of independent repetitions.
    pub r_reps: u8,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for MuveraConfig {
    fn default() -> Self {
        Self {
            k_sim: 3,
            r_reps: 5,
            seed: 42,
        }
    }
}

impl MuveraConfig {
    /// Create a new configuration with custom parameters.
    #[must_use]
    pub fn new(k_sim: u8, r_reps: u8, seed: u64) -> Self {
        Self {
            k_sim,
            r_reps,
            seed,
        }
    }

    /// Calculate the FDE dimension for a given token dimension.
    ///
    /// Formula: r_reps × 2^k_sim × token_dim
    #[must_use]
    pub fn fde_dimension(&self, token_dim: usize) -> usize {
        self.r_reps as usize * self.num_partitions() * token_dim
    }

    /// Number of partitions (buckets) per repetition.
    ///
    /// Equal to 2^k_sim.
    #[must_use]
    pub fn num_partitions(&self) -> usize {
        1 << self.k_sim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MuveraConfig::default();
        assert_eq!(config.k_sim, 3);
        assert_eq!(config.r_reps, 5);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_fde_dimension_default() {
        let config = MuveraConfig::default();
        // k_sim=3 -> 8 partitions, r_reps=5, token_dim=128
        // 5 * 8 * 128 = 5,120
        assert_eq!(config.fde_dimension(128), 5120);
    }

    #[test]
    fn test_fde_dimension_custom() {
        // k_sim=4 -> 16 partitions, r_reps=10
        // 10 * 16 * 128 = 20,480
        let config = MuveraConfig::new(4, 10, 42);
        assert_eq!(config.fde_dimension(128), 20480);
    }

    #[test]
    fn test_num_partitions() {
        assert_eq!(MuveraConfig::new(3, 5, 42).num_partitions(), 8);
        assert_eq!(MuveraConfig::new(4, 5, 42).num_partitions(), 16);
        assert_eq!(MuveraConfig::new(5, 5, 42).num_partitions(), 32);
        assert_eq!(MuveraConfig::new(6, 5, 42).num_partitions(), 64);
    }
}
