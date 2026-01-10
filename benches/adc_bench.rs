//! Benchmarks for ADC (Asymmetric Distance Computation) lookup tables
//!
//! Measures the performance improvement of ADC over traditional asymmetric distance.
//! Expected: 5-10x speedup for 4-bit quantization.
//!
//! Run: cargo bench --bench adc_bench

#![allow(deprecated)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use omendb::compression::{RaBitQ, RaBitQParams};
use rand::Rng;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Benchmark asymmetric distance (baseline)
fn bench_asymmetric_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("asymmetric_distance");

    for dim in [128, 384, 768, 1536] {
        let quantizer = RaBitQ::default_4bit();
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Quantize all vectors
        let quantized: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for qv in &quantized {
                    black_box(quantizer.distance_asymmetric_l2(&query, qv));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark ADC lookup table distance (optimized)
fn bench_adc_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("adc_distance");

    for dim in [128, 384, 768, 1536] {
        let quantizer = RaBitQ::default_4bit();
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Quantize all vectors
        let quantized: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();

        // Build ADC table once (use with_scale since quantizer not trained)
        let adc = quantizer.build_adc_table_with_scale(&query, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for qv in &quantized {
                    black_box(adc.distance(&qv.data));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark ADC with SIMD
fn bench_adc_distance_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("adc_distance_simd");

    for dim in [128, 384, 768, 1536] {
        let quantizer = RaBitQ::default_4bit();
        let vectors = generate_random_vectors(1000, dim);
        let query = generate_random_vectors(1, dim).pop().unwrap();

        // Quantize all vectors
        let quantized: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();

        // Build ADC table once (use with_scale since quantizer not trained)
        let adc = quantizer.build_adc_table_with_scale(&query, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                for qv in &quantized {
                    black_box(adc.distance_squared_simd(&qv.data));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark ADC table construction overhead
fn bench_adc_table_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("adc_table_build");

    for dim in [128, 384, 768, 1536] {
        let quantizer = RaBitQ::default_4bit();
        let query = generate_random_vectors(1, dim).pop().unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| {
                black_box(quantizer.build_adc_table_with_scale(&query, 1.0));
            })
        });
    }

    group.finish();
}

/// Compare 2-bit, 4-bit, and 8-bit ADC performance
fn bench_adc_bit_widths(c: &mut Criterion) {
    let mut group = c.benchmark_group("adc_bit_widths");
    let dim = 768;
    let vectors = generate_random_vectors(1000, dim);
    let query = generate_random_vectors(1, dim).pop().unwrap();

    for (name, params) in [
        ("2bit", RaBitQParams::bits2()),
        ("4bit", RaBitQParams::bits4()),
        ("8bit", RaBitQParams::bits8()),
    ] {
        let quantizer = RaBitQ::new(params);
        let quantized: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();
        let adc = quantizer.build_adc_table_with_scale(&query, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(name), &name, |b, _| {
            b.iter(|| {
                for qv in &quantized {
                    black_box(adc.distance(&qv.data));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark convenience method (builds table per call)
fn bench_distance_with_adc(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_with_adc_convenience");

    let dim = 768;
    let quantizer = RaBitQ::default_4bit();
    let vectors = generate_random_vectors(100, dim); // Fewer vectors since this is slower
    let query = generate_random_vectors(1, dim).pop().unwrap();

    let quantized: Vec<_> = vectors.iter().map(|v| quantizer.quantize(v)).collect();

    group.bench_function("convenience_method", |b| {
        b.iter(|| {
            for qv in &quantized {
                black_box(quantizer.distance_with_adc(&query, qv));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_asymmetric_distance,
    bench_adc_distance,
    bench_adc_distance_simd,
    bench_adc_table_build,
    bench_adc_bit_widths,
    bench_distance_with_adc
);
criterion_main!(benches);
