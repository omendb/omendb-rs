//! Architecture-aware prefetch strategy
//!
//! Platform-specific prefetch configuration:
//! - Apple Silicon: Disabled (Data Memory-Dependent Prefetcher handles it)
//! - x86/ARM servers: Enabled (8-50% improvement in HNSW search)

/// Compile-time prefetch configuration based on target architecture
pub struct PrefetchConfig;

impl PrefetchConfig {
    /// Whether prefetching is enabled for this platform
    ///
    /// Disabled on Apple Silicon (M1/M2/M3) where the Data Memory-Dependent
    /// Prefetcher (DMP) handles prefetching automatically and explicit prefetch
    /// instructions can actually hurt performance.
    ///
    /// Enabled on x86_64 and ARM servers where explicit prefetching helps.
    #[inline(always)]
    #[must_use]
    pub const fn enabled() -> bool {
        // Apple Silicon detection: aarch64 + macOS
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            false // Apple M1/M2/M3 - DMP handles prefetching
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            true // x86, ARM servers, etc.
        }
    }

    /// Prefetch stride (how many vectors ahead to prefetch)
    ///
    /// Tuned for typical L2 cache sizes and memory latency.
    #[inline(always)]
    #[must_use]
    pub const fn stride() -> usize {
        4
    }

    /// Cache line size in bytes (for alignment)
    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub const fn cache_line_size() -> usize {
        64
    }
}
