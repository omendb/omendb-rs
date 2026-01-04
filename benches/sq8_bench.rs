//! Benchmarks for SQ8 (Scalar Quantization 8-bit) distance computation
//!
//! Compares:
//! - Direct asymmetric L2 (current implementation)
//! - ADC lookup table approach
//! - Full precision L2 (baseline with SIMD)
//!
//! Run: cargo bench --bench sq8_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb::compression::{ScalarParams, UniformScalarParams};
use omendb::distance::{dot_product, l2_distance_squared};
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

/// Benchmark SQ8 asymmetric L2 (current implementation)
fn bench_sq8_asymmetric_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/asymmetric_l2");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Train quantization
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        // Quantize vectors
        let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for q in &quantized {
                    black_box(params.asymmetric_l2_squared(&query, q));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark SQ8 ADC lookup table
fn bench_sq8_adc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/adc_table");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Train quantization
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        // Quantize vectors
        let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

        // Build ADC table once per query
        let adc_table = params.build_adc_table(&query);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for q in &quantized {
                    black_box(adc_table.distance_squared(q));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark ADC table build time
fn bench_sq8_adc_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/adc_build");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Train quantization
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = ScalarParams::train(&refs).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                black_box(params.build_adc_table(&query));
            })
        });
    }

    group.finish();
}

/// Benchmark Uniform SQ8 with integer SIMD (new fast implementation)
fn bench_uniform_sq8(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/uniform_int_simd");

    for dim in [128, 768] {
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Train uniform quantization
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = UniformScalarParams::train(&refs).unwrap();

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

/// End-to-end comparison: ADC build + N lookups vs N asymmetric distances
fn bench_sq8_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("sq8_comparison/crossover");
    let dim = 768;

    let vectors = generate_random_vectors(1000, dim);
    let query = generate_random_vectors(1, dim).pop().unwrap();

    // Train quantization
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let params = ScalarParams::train(&refs).unwrap();

    // Quantize vectors
    let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

    // Test different candidate counts to find crossover point
    for n_candidates in [10, 50, 100, 500, 1000] {
        let candidates = &quantized[..n_candidates];

        group.bench_with_input(
            BenchmarkId::new("asymmetric", n_candidates),
            &n_candidates,
            |b, _| {
                b.iter(|| {
                    for q in candidates {
                        black_box(params.asymmetric_l2_squared(&query, q));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adc_with_build", n_candidates),
            &n_candidates,
            |b, _| {
                b.iter(|| {
                    let adc_table = params.build_adc_table(&query);
                    for q in candidates {
                        black_box(adc_table.distance_squared(q));
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
    bench_sq8_asymmetric_l2,
    bench_uniform_sq8,
    bench_sq8_adc,
    bench_sq8_adc_build,
    bench_sq8_crossover
);
criterion_main!(benches);
