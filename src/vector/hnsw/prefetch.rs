//! Architecture-aware prefetch strategy
//!
//! Different CPUs have vastly different prefetch characteristics:
//! - Apple Silicon (M1-M4): Aggressive DMP makes software prefetch harmful
//! - Intel/AMD x86: Moderate HW prefetcher, software prefetch helps 20-50%
//! - ARM servers (Graviton, Altra): Software prefetch essential (8-17% gains)

/// Prefetch configuration determined at compile time based on target architecture
pub struct PrefetchConfig;

impl PrefetchConfig {
    /// Whether software prefetching should be enabled for this platform
    ///
    /// Apple Silicon's DMP (Data Memory-dependent Prefetcher) already predicts
    /// access patterns from memory contents, making software prefetch redundant
    /// and adding ~2-5% overhead.
    #[inline(always)]
    pub const fn enabled() -> bool {
        // Apple Silicon: DMP handles prefetching optimally
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            false
        }

        // Other ARM (Graviton, Altra): Software prefetch essential
        // Ampere Altra intentionally disables most HW prefetchers
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        {
            true
        }

        // x86_64: Moderate benefit from software prefetch
        #[cfg(target_arch = "x86_64")]
        {
            true
        }

        // Default fallback
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Optimal stride distance for this platform
    ///
    /// - x86: stride=4 works well (VSAG research)
    /// - ARM servers: stride=2-4 depending on memory bandwidth
    /// - Apple: N/A (prefetch disabled)
    #[inline(always)]
    pub const fn stride() -> usize {
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        {
            4 // Graviton/Altra benefit from aggressive prefetch
        }

        #[cfg(target_arch = "x86_64")]
        {
            4 // VSAG-validated optimal stride
        }

        #[cfg(any(
            all(target_arch = "aarch64", target_vendor = "apple"),
            not(any(target_arch = "aarch64", target_arch = "x86_64"))
        ))]
        {
            0 // Disabled
        }
    }

    /// Cache line size for this platform
    ///
    /// Apple Silicon uses 128-byte cache lines, everyone else uses 64 bytes
    #[inline(always)]
    #[allow(dead_code)] // Used in tests, useful utility
    pub const fn cache_line_size() -> usize {
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            128
        }

        #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
        {
            64
        }
    }
}

/// Conditionally execute prefetch based on platform
#[inline(always)]
#[allow(dead_code)] // Useful utility for future use
pub fn prefetch_if_enabled<F: FnOnce()>(f: F) {
    if PrefetchConfig::enabled() {
        f();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_is_const() {
        // These should all be compile-time constants
        const ENABLED: bool = PrefetchConfig::enabled();
        const STRIDE: usize = PrefetchConfig::stride();
        const CACHE_LINE: usize = PrefetchConfig::cache_line_size();

        // Sanity checks - use const blocks for constant assertions
        const { assert!(!PrefetchConfig::enabled() || PrefetchConfig::stride() > 0) };
        const {
            assert!(
                PrefetchConfig::cache_line_size() == 64 || PrefetchConfig::cache_line_size() == 128
            );
        };
        // Silence unused variable warnings
        let _ = (ENABLED, STRIDE, CACHE_LINE);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    fn test_apple_silicon_disabled() {
        assert!(!PrefetchConfig::enabled());
        assert_eq!(PrefetchConfig::cache_line_size(), 128);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_enabled() {
        assert!(PrefetchConfig::enabled());
        assert_eq!(PrefetchConfig::stride(), 4);
        assert_eq!(PrefetchConfig::cache_line_size(), 64);
    }
}
