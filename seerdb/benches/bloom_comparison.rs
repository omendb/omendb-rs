// Benchmark comparing traditional vs learned bloom filters
// Measures: space savings, false positive rate, query time

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::bloom::{BloomFilter, LearnedBloomFilter};
use std::collections::HashSet;

fn generate_keys(n: usize, start: usize) -> Vec<String> {
    (start..start + n)
        .map(|i| format!("key_{:010}", i))
        .collect()
}

fn benchmark_space_savings(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_space_savings");

    for size in [1000, 10000, 100000].iter() {
        // Traditional bloom filter
        let mut bf = BloomFilter::new(*size, 0.01);
        let keys = generate_keys(*size, 0);
        for key in &keys {
            bf.insert(key);
        }
        let trad_size = bf.size_bytes();

        // Learned bloom filter
        let mut lbf = LearnedBloomFilter::new(*size, 0.01, 0.7);
        let negative_keys = generate_keys(*size, 1_000_000);
        lbf.train(&keys, &negative_keys);
        let learned_size = lbf.size_bytes();

        let reduction = ((trad_size - learned_size) as f64 / trad_size as f64) * 100.0;

        println!("\n=== Dataset size: {} ===", size);
        println!("Traditional:  {} bytes", trad_size);
        println!("Learned:      {} bytes", learned_size);
        println!("Reduction:    {:.1}%", reduction);
    }

    group.finish();
}

fn benchmark_false_positive_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_false_positive_rate");

    let size = 10000;
    let keys = generate_keys(size, 0);
    let test_keys = generate_keys(size, 1_000_000); // Keys NOT in set

    // Traditional bloom filter
    let mut bf = BloomFilter::new(size, 0.01);
    for key in &keys {
        bf.insert(key);
    }

    let mut trad_fp = 0;
    for key in &test_keys {
        if bf.contains(key) {
            trad_fp += 1;
        }
    }
    let trad_fpr = trad_fp as f64 / test_keys.len() as f64;

    // Learned bloom filter
    let mut lbf = LearnedBloomFilter::new(size, 0.01, 0.7);
    let negative_training = generate_keys(size, 2_000_000);
    lbf.train(&keys, &negative_training);

    let mut learned_fp = 0;
    for key in &test_keys {
        if lbf.contains(key) {
            learned_fp += 1;
        }
    }
    let learned_fpr = learned_fp as f64 / test_keys.len() as f64;

    println!("\n=== False Positive Rate (target: 1%) ===");
    println!(
        "Traditional: {:.2}% ({}/{})",
        trad_fpr * 100.0,
        trad_fp,
        test_keys.len()
    );
    println!(
        "Learned:     {:.2}% ({}/{})",
        learned_fpr * 100.0,
        learned_fp,
        test_keys.len()
    );

    group.finish();
}

fn benchmark_query_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_query_time");

    let size = 10000;
    let keys = generate_keys(size, 0);
    let negative_keys = generate_keys(size, 1_000_000);

    // Traditional bloom filter
    let mut bf = BloomFilter::new(size, 0.01);
    for key in &keys {
        bf.insert(key);
    }

    // Learned bloom filter
    let mut lbf = LearnedBloomFilter::new(size, 0.01, 0.7);
    lbf.train(&keys, &negative_keys);

    // Benchmark positive lookups (keys in set)
    group.bench_with_input(
        BenchmarkId::new("traditional_positive", size),
        &size,
        |b, _| {
            b.iter(|| {
                for key in &keys {
                    black_box(bf.contains(black_box(key)));
                }
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("learned_positive", size), &size, |b, _| {
        b.iter(|| {
            for key in &keys {
                black_box(lbf.contains(black_box(key)));
            }
        });
    });

    // Benchmark negative lookups (keys NOT in set)
    let test_keys = generate_keys(size, 2_000_000);

    group.bench_with_input(
        BenchmarkId::new("traditional_negative", size),
        &size,
        |b, _| {
            b.iter(|| {
                for key in &test_keys {
                    black_box(bf.contains(black_box(key)));
                }
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("learned_negative", size), &size, |b, _| {
        b.iter(|| {
            for key in &test_keys {
                black_box(lbf.contains(black_box(key)));
            }
        });
    });

    group.finish();
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_scaling");

    println!("\n=== Space Savings at Different Scales ===");
    for size in [100, 1000, 10_000, 100_000].iter() {
        let keys = generate_keys(*size, 0);
        let negative_keys = generate_keys(*size, 1_000_000);

        // Traditional
        let mut bf = BloomFilter::new(*size, 0.01);
        for key in &keys {
            bf.insert(key);
        }
        let trad_size = bf.size_bytes();

        // Learned
        let mut lbf = LearnedBloomFilter::new(*size, 0.01, 0.7);
        lbf.train(&keys, &negative_keys);
        let learned_size = lbf.size_bytes();

        let reduction = ((trad_size - learned_size) as f64 / trad_size as f64) * 100.0;

        println!(
            "{:>10} keys | Trad: {:>8} bytes | Learned: {:>8} bytes | Reduction: {:>5.1}%",
            size, trad_size, learned_size, reduction
        );
    }

    group.finish();
}

fn benchmark_threshold_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_threshold_impact");

    let size = 10000;
    let keys = generate_keys(size, 0);
    let negative_keys = generate_keys(size, 1_000_000);
    let test_keys = generate_keys(size, 2_000_000);

    println!("\n=== Threshold Impact (confidence 0.0-1.0) ===");
    for threshold in [0.3, 0.5, 0.7, 0.9].iter() {
        let mut lbf = LearnedBloomFilter::new(size, 0.01, *threshold);
        lbf.train(&keys, &negative_keys);

        // Check false positive rate
        let mut fp_count = 0;
        for key in &test_keys {
            if lbf.contains(key) {
                fp_count += 1;
            }
        }
        let fpr = fp_count as f64 / test_keys.len() as f64;
        let size_bytes = lbf.size_bytes();

        println!(
            "Threshold {:.1} | Size: {:>8} bytes | FPR: {:.2}%",
            threshold,
            size_bytes,
            fpr * 100.0
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_space_savings,
    benchmark_false_positive_rate,
    benchmark_query_time,
    benchmark_scaling,
    benchmark_threshold_impact
);
criterion_main!(benches);
