// Benchmark: SIMD linear vs binary search for different range sizes
// Validates that SIMD linear is optimal for ALEX's typical ranges (32-128 elements)

#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::simd::prelude::*;

/// SIMD linear search (current implementation)
fn simd_linear_search(keys: &[Option<i64>], key: i64) -> Option<usize> {
    const LANES: usize = 4;
    let len = keys.len();

    if len == 0 {
        return None;
    }

    // Fast path: Check first element (common after model prediction)
    if keys[0] == Some(key) {
        return Some(0);
    }

    let key_vec = i64x4::splat(key);
    let mut i = 0;

    // SIMD path: Process 4 values at once
    while i + LANES <= len {
        let mut values = [i64::MAX; LANES];
        for j in 0..LANES {
            values[j] = keys[i + j].unwrap_or(i64::MAX);
        }

        let vec = i64x4::from_array(values);
        let mask = vec.simd_eq(key_vec);

        if mask.any() {
            // Find which lane matched
            for j in 0..LANES {
                if keys[i + j] == Some(key) {
                    return Some(i + j);
                }
            }
        }

        i += LANES;
    }

    // Scalar fallback for remaining elements
    for j in i..len {
        if keys[j] == Some(key) {
            return Some(j);
        }
    }

    None
}

/// Standard binary search (baseline)
fn binary_search(keys: &[Option<i64>], key: i64) -> Option<usize> {
    let mut left = 0;
    let mut right = keys.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match keys[mid] {
            Some(k) if k == key => return Some(mid),
            Some(k) if k < key => left = mid + 1,
            _ => right = mid,
        }
    }

    None
}

/// Scalar linear search (baseline)
fn linear_search(keys: &[Option<i64>], key: i64) -> Option<usize> {
    keys.iter().position(|&k| k == Some(key))
}

fn benchmark_search_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_strategies");

    // Test different range sizes typical for ALEX
    for size in [16, 32, 64, 128, 256, 512] {
        // Create sorted array with gaps (realistic ALEX node)
        let keys: Vec<Option<i64>> = (0..size)
            .map(|i| if i % 4 == 0 { Some(i as i64) } else { None })
            .collect();

        // Search for middle element
        let target = (size / 2) as i64;

        group.bench_with_input(BenchmarkId::new("simd_linear", size), &size, |b, _| {
            b.iter(|| black_box(simd_linear_search(&keys, black_box(target))));
        });

        group.bench_with_input(BenchmarkId::new("binary_search", size), &size, |b, _| {
            b.iter(|| black_box(binary_search(&keys, black_box(target))));
        });

        group.bench_with_input(BenchmarkId::new("scalar_linear", size), &size, |b, _| {
            b.iter(|| black_box(linear_search(&keys, black_box(target))));
        });
    }

    group.finish();
}

fn benchmark_alex_typical_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("alex_typical");

    // Simulate ALEX's typical case: search after exponential search narrows to 64 elements
    let size = 64;
    let keys: Vec<Option<i64>> = (0..size)
        .map(|i| {
            if i % 3 == 0 {
                Some(i as i64 * 10)
            } else {
                None
            }
        })
        .collect();

    // Test different positions (first, middle, last, not found)
    let test_cases = [
        ("first", 0),
        ("early", 30),
        ("middle", 320),
        ("late", 600),
        ("not_found", 999),
    ];

    for (name, target) in test_cases {
        group.bench_with_input(BenchmarkId::new("simd_linear", name), &target, |b, &t| {
            b.iter(|| black_box(simd_linear_search(&keys, black_box(t))));
        });

        group.bench_with_input(BenchmarkId::new("binary_search", name), &target, |b, &t| {
            b.iter(|| black_box(binary_search(&keys, black_box(t))));
        });
    }

    group.finish();
}

fn benchmark_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");

    // Worst case: target is last element or not found
    for size in [32, 64, 128, 256] {
        let keys: Vec<Option<i64>> = (0..size).map(|i| Some(i as i64)).collect();

        let target = size as i64 - 1; // Last element

        group.bench_with_input(BenchmarkId::new("simd_linear", size), &size, |b, _| {
            b.iter(|| black_box(simd_linear_search(&keys, black_box(target))));
        });

        group.bench_with_input(BenchmarkId::new("binary_search", size), &size, |b, _| {
            b.iter(|| black_box(binary_search(&keys, black_box(target))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_search_strategies,
    benchmark_alex_typical_case,
    benchmark_worst_case
);
criterion_main!(benches);
