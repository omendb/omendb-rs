// Bloom filter implementation
//
// Uses bit-packed Vec<u64> storage (standard approach, 8x more efficient than naive Vec<bool>)
// Learned bloom filter available but has accuracy issues (48-51% FPR vs 1% target)

mod bitpacked;
mod blocked;
mod learned;
// mod simd; // Disabled: 18% regression on negative lookups (hot path) - see SIMD_OPPORTUNITIES.md

#[cfg(test)]
mod traditional; // Naive Vec<bool> implementation for benchmarking only

// Export blocked (cache-line optimized) as default BloomFilter (3.4x faster)
pub use blocked::BlockedBloomFilter as BloomFilter;
// Keep old implementations available with explicit names
pub use bitpacked::BloomFilter as BitPackedBloomFilter;
pub use blocked::BlockedBloomFilter;
pub use learned::LearnedBloomFilter;
// pub use simd::SimdBloomFilter; // Disabled - see SIMD_OPPORTUNITIES.md
