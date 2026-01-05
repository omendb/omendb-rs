//! Benchmarks for SQ8 (Scalar Quantization 8-bit) distance computation
//!
//! Compares:
//! - SQ8 with integer SIMD (current implementation)
//! - Full precision L2 (baseline with SIMD)
//! - Full precision L2 with decomposition
//!
//! Run: cargo bench --bench sq8_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb_core::compression::ScalarParams;
use omendb_core::distance::{dot_product, l2_distance_squared};
use rand::Rng;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..255.0)).collect())
        .collect()
}

/// Benchmark full-precision L2 distance with SIMD (baseline)
fn bench_fp32_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/fp32_l2_simd");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for v in &vectors {
                    black_box(l2_distance_squared(&query, v));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark full-precision L2 with decomposition (what FP32 HNSW uses)
fn bench_fp32_l2_decomposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/fp32_l2_decomposed");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Pre-compute norms (what HNSW does during insert)
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum())
            .collect();
        let query_norm: f32 = query.iter().map(|x| x * x).sum();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for (v, &vec_norm) in vectors.iter().zip(norms.iter()) {
                    // L2 decomposition: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
                    let dot = dot_product(&query, v);
                    let dist = query_norm + vec_norm - 2.0 * dot;
                    black_box(dist);
                }
            })
        });
    }

    group.finish();
}

/// Benchmark SQ8 with integer SIMD (current implementation)
fn bench_sq8_int_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/int_simd");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Train quantization
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        // Quantize vectors with precomputed metadata
        let quantized: Vec<_> = vectors.iter().map(|v| params.quantize(v)).collect();

        // Prepare query once (precomputes norm, sum, quantized values)
        let query_prep = params.prepare_query(&query);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for q in &quantized {
                    black_box(params.distance_l2_squared(&query_prep, q));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark query preparation overhead
fn bench_sq8_query_prep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/query_prep");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                black_box(params.prepare_query(&query));
            })
        });
    }

    group.finish();
}

/// Benchmark vector quantization overhead
fn bench_sq8_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/quantize");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for v in &vectors {
                    black_box(params.quantize(v));
                }
            })
        });
    }

    group.finish();
}

/// End-to-end comparison: SQ8 vs FP32 for different candidate counts
fn bench_sq8_vs_fp32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/vs_fp32");
    let dim = 768;

    let vectors = generate_random_vectors(1000, dim);
    let query = generate_random_vectors(1, dim).pop().unwrap();

    // Setup SQ8
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = ScalarParams::train(&refs).unwrap();
    let quantized: Vec<_> = vectors.iter().map(|v| params.quantize(v)).collect();
    let query_prep = params.prepare_query(&query);

    // Test different candidate counts
    for n_candidates in [10, 50, 100, 500, 1000] {
        let fp_candidates = &vectors[..n_candidates];
        let sq_candidates = &quantized[..n_candidates];

        group.bench_with_input(
            BenchmarkId::new("fp32", n_candidates),
            &n_candidates,
            |b, _| {
                b.iter(|| {
                    for v in fp_candidates {
                        black_box(l2_distance_squared(&query, v));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sq8", n_candidates),
            &n_candidates,
            |b, _| {
                b.iter(|| {
                    for q in sq_candidates {
                        black_box(params.distance_l2_squared(&query_prep, q));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fp32_l2,
    bench_fp32_l2_decomposed,
    bench_sq8_int_simd,
    bench_sq8_query_prep,
    bench_sq8_quantize,
    bench_sq8_vs_fp32
);
criterion_main!(benches);
