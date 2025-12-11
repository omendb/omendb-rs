//! Benchmark ADC vs SIMD for SQ8 on x86
//!
//! Run: cargo run --release --example bench_adc_x86

use omendb_core::compression::scalar::ScalarParams;
use std::time::Instant;

fn main() {
    let dimensions = 768;
    let num_vectors = 10_000;
    let num_queries = 1_000;

    println!("ADC vs SIMD Benchmark (x86)");
    println!("===========================");
    println!("Dimensions: {dimensions}");
    println!("Vectors: {num_vectors}");
    println!("Queries: {num_queries}");
    println!();

    // Generate random vectors
    let mut rng_seed = 42u64;
    let mut random = || -> f32 {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    // Create quantization params by training on sample data
    let training_data: Vec<Vec<f32>> = (0..256)
        .map(|_| (0..dimensions).map(|_| random()).collect())
        .collect();

    let params = ScalarParams::train(
        training_data
            .iter()
            .map(|v| v.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
        dimensions,
    );

    // Generate and quantize target vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| random()).collect())
        .collect();

    let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| params.quantize(v)).collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dimensions).map(|_| random()).collect())
        .collect();

    // Benchmark ADC (Asymmetric Distance Computation with lookup tables)
    println!("Benchmarking ADC (precomputed lookup tables)...");
    let start = Instant::now();
    let mut adc_sum = 0.0f32;
    for query in &queries {
        let adc_table = params.build_adc_table(query);
        for q in &quantized {
            adc_sum += adc_table.distance_squared(q);
        }
    }
    let adc_time = start.elapsed();
    let adc_ops = (num_queries * num_vectors) as f64;
    let adc_ops_per_sec = adc_ops / adc_time.as_secs_f64();
    println!(
        "  ADC: {:.2}ms total, {:.2}M ops/sec (checksum: {:.2})",
        adc_time.as_secs_f64() * 1000.0,
        adc_ops_per_sec / 1_000_000.0,
        adc_sum
    );

    // Benchmark asymmetric SIMD (on-the-fly dequantization)
    println!("Benchmarking asymmetric SIMD (on-the-fly)...");
    let start = Instant::now();
    let mut simd_sum = 0.0f32;
    for query in &queries {
        for q in &quantized {
            simd_sum += params.asymmetric_l2_squared(query, q);
        }
    }
    let simd_time = start.elapsed();
    let simd_ops_per_sec = adc_ops / simd_time.as_secs_f64();
    println!(
        "  SIMD: {:.2}ms total, {:.2}M ops/sec (checksum: {:.2})",
        simd_time.as_secs_f64() * 1000.0,
        simd_ops_per_sec / 1_000_000.0,
        simd_sum
    );

    // Summary
    println!();
    println!("Summary");
    println!("-------");
    let ratio = simd_time.as_secs_f64() / adc_time.as_secs_f64();
    if ratio > 1.0 {
        println!("ADC is {:.2}x faster than SIMD", ratio);
    } else {
        println!("SIMD is {:.2}x faster than ADC", 1.0 / ratio);
    }

    // Cache info
    println!();
    println!("Cache Analysis");
    println!("--------------");
    let adc_table_size = dimensions * 256 * 4; // 256 bins per dimension, 4 bytes each
    println!("ADC table size: {}KB", adc_table_size / 1024);
    println!("Expected: ADC faster when table fits in L2/L3 cache");
}
