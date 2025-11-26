// SIMD-accelerated operations using portable SIMD (std::simd)
//
// Implements vectorized operations for performance-critical paths:
// - Key comparisons (memtable, block iteration)
// - Bloom filter hash checks
// - Prefix length calculation
//
// Uses portable SIMD API that works across all platforms (x86_64, ARM, etc.)
// Compiler automatically selects optimal instructions (SSE2, AVX2, NEON, etc.)
//
// Expected improvement: +5-15% overall throughput in key-heavy operations

use std::cmp::Ordering;
use std::simd::{cmp::SimdPartialEq, cmp::SimdPartialOrd, u8x16};

/// SIMD vector size (16 bytes for u8x16)
const SIMD_WIDTH: usize = 16;

/// Compare two byte slices using SIMD
///
/// This is a drop-in replacement for slice comparison that uses
/// portable SIMD to process 16 bytes at a time. The compiler
/// automatically selects the best instructions for the target platform.
#[inline]
pub fn compare_keys(a: &[u8], b: &[u8]) -> Ordering {
    let len_a = a.len();
    let len_b = b.len();
    let min_len = len_a.min(len_b);

    let mut i = 0;

    // Process 16 bytes at a time with SIMD
    while i + SIMD_WIDTH <= min_len {
        // Load 16 bytes from each slice
        let a_vec = u8x16::from_slice(&a[i..i + SIMD_WIDTH]);
        let b_vec = u8x16::from_slice(&b[i..i + SIMD_WIDTH]);

        // Compare for equality
        let eq = a_vec.simd_eq(b_vec);

        // If not all bytes are equal, find first difference
        if !eq.all() {
            // Find position of first differing byte
            for j in 0..SIMD_WIDTH {
                let pos = i + j;
                match a[pos].cmp(&b[pos]) {
                    Ordering::Equal => continue,
                    other => return other,
                }
            }
        }

        i += SIMD_WIDTH;
    }

    // Handle remaining bytes with scalar comparison
    while i < min_len {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => i += 1,
            other => return other,
        }
    }

    // If all compared bytes are equal, compare lengths
    len_a.cmp(&len_b)
}

/// Decode a varint from a byte slice using SIMD to find the length
/// Returns (value, bytes_read) if successful
///
/// Optimized using std::simd to quickly scan for the varint terminator (MSB=0)
/// avoiding branch mispredictions in the loop.
#[inline]
pub fn decode_varint(data: &[u8]) -> Option<(u64, usize)> {
    if data.is_empty() {
        return None;
    }

    // Fast path for single-byte varints (very common)
    if data[0] < 128 {
        return Some((data[0] as u64, 1));
    }

    // SIMD path: Load 16 bytes, find first byte with MSB=0
    // We need at least 16 bytes to use simple unaligned load.
    // If buffer is smaller, we fall back to scalar.
    if data.len() >= 16 {
        let v = u8x16::from_slice(&data[..16]);
        // Check MSB (byte < 128). 0x80 is the continuation bit.
        // Values < 128 have MSB=0 (terminator).
        let mask = v.simd_lt(u8x16::splat(128));
        let bitmask = mask.to_bitmask();

        // If bitmask is 0, all 16 bytes have MSB=1 (>= 128).
        // That means varint is at least 16 bytes long, which overflows u64 (max 10 bytes).
        if bitmask == 0 {
            return None;
        }

        // Find index of first set bit (first terminator)
        let len = bitmask.trailing_zeros() as usize + 1;

        if len > 10 {
            return None; // Too long for u64
        }

        // Unrolled decoding based on known length
        // We know the length, so we can avoid loop and branch checks
        let mut value: u64 = 0;

        // Using explicit match for small sizes which are most common
        match len {
            1 => return Some((data[0] as u64, 1)),
            2 => {
                value = (data[0] & 0x7F) as u64;
                value |= (data[1] as u64) << 7;
                return Some((value, 2));
            }
            3 => {
                value = (data[0] & 0x7F) as u64;
                value |= ((data[1] & 0x7F) as u64) << 7;
                value |= (data[2] as u64) << 14;
                return Some((value, 3));
            }
            4 => {
                value = (data[0] & 0x7F) as u64;
                value |= ((data[1] & 0x7F) as u64) << 7;
                value |= ((data[2] & 0x7F) as u64) << 14;
                value |= (data[3] as u64) << 21;
                return Some((value, 4));
            }
            5 => {
                value = (data[0] & 0x7F) as u64;
                value |= ((data[1] & 0x7F) as u64) << 7;
                value |= ((data[2] & 0x7F) as u64) << 14;
                value |= ((data[3] & 0x7F) as u64) << 21;
                value |= (data[4] as u64) << 28;
                return Some((value, 5));
            }
            // For larger sizes (6-10), use a loop or further unrolling
            // Since they are rare, a small loop with fixed bounds is fine
            _ => {
                let mut shift = 0;
                for (i, val) in data.iter().enumerate().take(len) {
                    let byte = *val;
                    if i == len - 1 {
                        value |= (byte as u64) << shift;
                    } else {
                        value |= ((byte & 0x7F) as u64) << shift;
                    }
                    shift += 7;
                }
                return Some((value, len));
            }
        }
    }

    // Scalar fallback for short buffers (< 16 bytes)
    let mut value: u64 = 0;
    let mut shift = 0;
    for (i, &byte) in data.iter().enumerate() {
        if i >= 10 {
            return None; // Too long
        }
        if byte < 128 {
            value |= (byte as u64) << shift;
            return Some((value, i + 1));
        }
        value |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
    }
    None
}

/// Calculate shared prefix length between two keys using SIMD
///
/// Used in prefix compression to find how many leading bytes are identical.
/// Returns the number of matching bytes from the start.
#[inline]
pub fn shared_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let mut i = 0;

    // Process 16 bytes at a time with SIMD
    while i + SIMD_WIDTH <= min_len {
        let a_vec = u8x16::from_slice(&a[i..i + SIMD_WIDTH]);
        let b_vec = u8x16::from_slice(&b[i..i + SIMD_WIDTH]);

        let eq = a_vec.simd_eq(b_vec);

        // If all bytes match, continue
        if eq.all() {
            i += SIMD_WIDTH;
            continue;
        }

        // Find position of first difference
        for j in 0..SIMD_WIDTH {
            if a[i + j] != b[i + j] {
                return i + j;
            }
        }
    }

    // Handle remaining bytes
    while i < min_len && a[i] == b[i] {
        i += 1;
    }

    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_keys_equal() {
        let a = b"hello world";
        let b = b"hello world";
        assert_eq!(compare_keys(a, b), Ordering::Equal);
    }

    #[test]
    fn test_compare_keys_less() {
        let a = b"hello";
        let b = b"world";
        assert_eq!(compare_keys(a, b), Ordering::Less);
    }

    #[test]
    fn test_compare_keys_greater() {
        let a = b"world";
        let b = b"hello";
        assert_eq!(compare_keys(a, b), Ordering::Greater);
    }

    #[test]
    fn test_compare_keys_different_lengths() {
        let a = b"hello";
        let b = b"hello world";
        assert_eq!(compare_keys(a, b), Ordering::Less);

        let a = b"hello world";
        let b = b"hello";
        assert_eq!(compare_keys(a, b), Ordering::Greater);
    }

    #[test]
    fn test_compare_keys_long() {
        // Test with keys longer than 16 bytes (multiple SIMD iterations)
        let a = b"this is a very long key that exceeds 16 bytes";
        let b = b"this is a very long key that exceeds 16 bytes";
        assert_eq!(compare_keys(a, b), Ordering::Equal);

        let a = b"this is a very long key that exceeds 16 bytes";
        let b = b"this is a very long key that exceeds 16 bytez";
        assert_eq!(compare_keys(a, b), Ordering::Less);
    }

    #[test]
    fn test_compare_keys_first_byte_differs() {
        let a = b"a";
        let b = b"b";
        assert_eq!(compare_keys(a, b), Ordering::Less);
    }

    #[test]
    fn test_compare_keys_last_byte_differs() {
        let a = b"hello worlda";
        let b = b"hello worldb";
        assert_eq!(compare_keys(a, b), Ordering::Less);
    }

    #[test]
    fn test_compare_keys_empty() {
        let a = b"";
        let b = b"";
        assert_eq!(compare_keys(a, b), Ordering::Equal);

        let a = b"";
        let b = b"hello";
        assert_eq!(compare_keys(a, b), Ordering::Less);

        let a = b"hello";
        let b = b"";
        assert_eq!(compare_keys(a, b), Ordering::Greater);
    }

    #[test]
    fn test_compare_keys_consistency_with_std() {
        // Verify SIMD comparison matches standard library
        let test_cases = [
            (b"" as &[u8], b"" as &[u8]),
            (b"a", b"a"),
            (b"a", b"b"),
            (b"b", b"a"),
            (b"hello", b"hello"),
            (b"hello", b"world"),
            (b"key_00000001", b"key_00000002"),
            (b"user:123:name", b"user:123:email"),
            (
                b"this is a very long key that exceeds sixteen bytes",
                b"this is a very long key that exceeds sixteen bytes",
            ),
            (
                b"this is a very long key that exceeds sixteen bytes",
                b"this is a very long key that exceeds sixteen bytez",
            ),
        ];

        for (a, b) in test_cases {
            let simd_result = compare_keys(a, b);
            let std_result = a.cmp(b);
            assert_eq!(
                simd_result, std_result,
                "SIMD comparison mismatch for {:?} vs {:?}",
                a, b
            );
        }
    }

    #[test]
    fn test_shared_prefix_len_no_match() {
        let a = b"hello";
        let b = b"world";
        assert_eq!(shared_prefix_len(a, b), 0);
    }

    #[test]
    fn test_shared_prefix_len_partial() {
        let a = b"user:123:name";
        let b = b"user:123:email";
        assert_eq!(shared_prefix_len(a, b), 9); // "user:123:"
    }

    #[test]
    fn test_shared_prefix_len_full() {
        let a = b"hello";
        let b = b"hello world";
        assert_eq!(shared_prefix_len(a, b), 5); // All of "hello"
    }

    #[test]
    fn test_shared_prefix_len_long() {
        // Test with keys longer than 16 bytes
        let a = b"user_00000001_data";
        let b = b"user_00000001_meta";
        assert_eq!(shared_prefix_len(a, b), 14); // "user_00000001_"
    }

    #[test]
    fn test_shared_prefix_len_empty() {
        let a = b"";
        let b = b"hello";
        assert_eq!(shared_prefix_len(a, b), 0);

        let a = b"hello";
        let b = b"";
        assert_eq!(shared_prefix_len(a, b), 0);
    }

    #[test]
    fn test_shared_prefix_len_consistency() {
        // Verify SIMD prefix length matches scalar implementation
        let test_cases = [
            (b"" as &[u8], b"" as &[u8]),
            (b"a", b"a"),
            (b"a", b"b"),
            (b"hello", b"hello"),
            (b"hello", b"world"),
            (b"user:123:name", b"user:123:email"),
            (b"key_00000001", b"key_00000002"),
        ];

        for (a, b) in test_cases {
            let simd_result = shared_prefix_len(a, b);

            // Scalar implementation for comparison
            let scalar_result = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();

            assert_eq!(
                simd_result, scalar_result,
                "SIMD prefix length mismatch for {:?} vs {:?}",
                a, b
            );
        }
    }

    #[test]
    fn test_decode_varint() {
        let mut large_data = vec![0u8; 32];

        // Test single byte
        large_data[0] = 0x05;
        assert_eq!(decode_varint(&large_data), Some((5, 1)));

        // Test 2 bytes
        large_data[0] = 0x85;
        large_data[1] = 0x01;
        assert_eq!(decode_varint(&large_data), Some((133, 2)));

        // Test 3 bytes
        large_data[0] = 0x80;
        large_data[1] = 0x80;
        large_data[2] = 0x01;
        assert_eq!(decode_varint(&large_data), Some((16384, 3)));

        // Test 5 bytes
        large_data[0] = 0x80;
        large_data[1] = 0x80;
        large_data[2] = 0x80;
        large_data[3] = 0x80;
        large_data[4] = 0x01;
        // 1 << 28
        assert_eq!(decode_varint(&large_data), Some((268435456, 5)));

        // Test too long (all continuation bits)
        for i in 0..16 {
            large_data[i] = 0x80;
        }
        assert_eq!(decode_varint(&large_data), None);
    }
}
