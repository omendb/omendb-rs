//! Benchmark ADC vs SIMD for SQ8 on x86
//!
//! Run: cargo run --release --example bench_adc_x86

use omendb_core::compression::scalar::ScalarParams;
use std::time::Instant;

fn benchmark_dimension(dimensions: usize, num_vectors: usize, num_queries: usize) -> (f64, f64) {
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
    )
    .unwrap();

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
    let start = Instant::now();
    let mut adc_sum = 0.0f32;
    for query in &queries {
        let adc_table = params.build_adc_table(query);
        for q in &quantized {
            adc_sum += adc_table.distance_squared(q);
        }
    }
    let adc_time = start.elapsed();

    // Benchmark asymmetric SIMD (on-the-fly dequantization)
    let start = Instant::now();
    let mut simd_sum = 0.0f32;
    for query in &queries {
        for q in &quantized {
            simd_sum += params.asymmetric_l2_squared(query, q);
        }
    }
    let simd_time = start.elapsed();

    // Verify results approximately match (floating point tolerance)
    let diff = (adc_sum - simd_sum).abs();
    let max_val = adc_sum.abs().max(simd_sum.abs());
    let relative_error = diff / max_val;
    assert!(
        relative_error < 0.0001,
        "Results mismatch: ADC={adc_sum}, SIMD={simd_sum}, rel_err={relative_error}"
    );

    let ops = (num_queries * num_vectors) as f64;
    (
        ops / adc_time.as_secs_f64() / 1_000_000.0,
        ops / simd_time.as_secs_f64() / 1_000_000.0,
    )
}

fn main() {
    let num_vectors = 10_000;
    let num_queries = 1_000;

    println!("ADC vs SIMD Benchmark (x86)");
    println!("===========================");
    println!("Platform: {}", std::env::consts::ARCH);
    println!("Vectors: {num_vectors}");
    println!("Queries: {num_queries}");
    println!();

    println!("| Dimension | ADC (M ops/s) | SIMD (M ops/s) | Winner | Speedup |");
    println!("|-----------|---------------|----------------|--------|---------|");

    for &dimensions in &[128, 384, 768, 1536] {
        let adc_table_kb = dimensions * 256 * 4 / 1024;

        let (adc_mops, simd_mops) = benchmark_dimension(dimensions, num_vectors, num_queries);

        let (winner, speedup) = if adc_mops > simd_mops {
            ("ADC", adc_mops / simd_mops)
        } else {
            ("SIMD", simd_mops / adc_mops)
        };

        println!(
            "| {dimensions:4}D ({adc_table_kb:4}KB) | {adc_mops:13.2} | {simd_mops:14.2} | {winner:6} | {speedup:7.2}x |"
        );
    }

    println!();
    println!("Cache sizes:");
    println!("  M3 Max: 128KB L2 per core");
    println!("  i9-13900KF: 36MB L3 shared");
}
