// Benchmark comparing standard vs blocked bloom filters
// Measures: cache locality impact, query time, false positive rate
//
// Expected results based on research:
// - Blocked: ~3x faster due to cache-line locality
// - Standard: More random memory accesses (k cache misses per query)
// - Trade-off: Blocked has slightly higher FPR due to reduced entropy

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use seerdb::bloom::{BlockedBloomFilter, BloomFilter};

fn generate_keys(n: usize, start: usize) -> Vec<String> {
    (start..start + n)
        .map(|i| format!("key_{:010}", i))
        .collect()
}

fn benchmark_insert_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_insert");

    for size in [1000, 10000, 100000].iter() {
        let keys = generate_keys(*size, 0);

        group.bench_with_input(BenchmarkId::new("standard", size), &keys, |b, keys| {
            b.iter(|| {
                let mut bf = BloomFilter::new(*size, 0.01);
                for key in keys {
                    bf.insert(black_box(key));
                }
                black_box(bf);
            });
        });

        group.bench_with_input(BenchmarkId::new("blocked", size), &keys, |b, keys| {
            b.iter(|| {
                let mut bbf = BlockedBloomFilter::new(*size, 0.01);
                for key in keys {
                    bbf.insert(black_box(key));
                }
                black_box(bbf);
            });
        });
    }

    group.finish();
}

fn benchmark_positive_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_positive_lookups");

    for size in [1000, 10000, 100000].iter() {
        let keys = generate_keys(*size, 0);

        // Standard bloom filter
        let mut bf = BloomFilter::new(*size, 0.01);
        for key in &keys {
            bf.insert(key);
        }

        // Blocked bloom filter
        let mut bbf = BlockedBloomFilter::new(*size, 0.01);
        for key in &keys {
            bbf.insert(key);
        }

        group.bench_with_input(BenchmarkId::new("standard", size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(bf.contains(black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("blocked", size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(bbf.contains(black_box(key)));
                }
            });
        });
    }

    group.finish();
}

fn benchmark_negative_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_negative_lookups");

    for size in [1000, 10000, 100000].iter() {
        let keys = generate_keys(*size, 0);
        let test_keys = generate_keys(*size, 1_000_000); // Keys NOT in set

        // Standard bloom filter
        let mut bf = BloomFilter::new(*size, 0.01);
        for key in &keys {
            bf.insert(key);
        }

        // Blocked bloom filter
        let mut bbf = BlockedBloomFilter::new(*size, 0.01);
        for key in &keys {
            bbf.insert(key);
        }

        group.bench_with_input(BenchmarkId::new("standard", size), &test_keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(bf.contains(black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("blocked", size), &test_keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(bbf.contains(black_box(key)));
                }
            });
        });
    }

    group.finish();
}

fn benchmark_cache_misses(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_cache_behavior");

    // Large filter to force cache misses
    let size = 1_000_000;
    let keys = generate_keys(10000, 0);
    let test_keys = generate_keys(10000, 1_000_000);

    // Standard bloom filter
    let mut bf = BloomFilter::new(size, 0.01);
    for key in &keys {
        bf.insert(key);
    }

    // Blocked bloom filter
    let mut bbf = BlockedBloomFilter::new(size, 0.01);
    for key in &keys {
        bbf.insert(key);
    }

    println!("\n=== Cache Behavior (1M capacity, testing 10K queries) ===");
    println!(
        "Standard filter size: {} bytes ({:.1} MB)",
        bf.size_bytes(),
        bf.size_bytes() as f64 / 1_048_576.0
    );
    println!(
        "Blocked filter size:  {} bytes ({:.1} MB)",
        bbf.size_bytes(),
        bbf.size_bytes() as f64 / 1_048_576.0
    );

    group.bench_function("standard_cold_cache", |b| {
        b.iter(|| {
            for key in &test_keys {
                black_box(bf.contains(black_box(key)));
            }
        });
    });

    group.bench_function("blocked_cold_cache", |b| {
        b.iter(|| {
            for key in &test_keys {
                black_box(bbf.contains(black_box(key)));
            }
        });
    });

    group.finish();
}

fn benchmark_false_positive_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_false_positive_rate");

    let size = 10000;
    let keys = generate_keys(size, 0);
    let test_keys = generate_keys(100000, 1_000_000); // Large test set

    // Standard bloom filter
    let mut bf = BloomFilter::new(size, 0.01);
    for key in &keys {
        bf.insert(key);
    }

    // Blocked bloom filter
    let mut bbf = BlockedBloomFilter::new(size, 0.01);
    for key in &keys {
        bbf.insert(key);
    }

    let mut standard_fp = 0;
    for key in &test_keys {
        if bf.contains(key) {
            standard_fp += 1;
        }
    }

    let mut blocked_fp = 0;
    for key in &test_keys {
        if bbf.contains(key) {
            blocked_fp += 1;
        }
    }

    let standard_fpr = standard_fp as f64 / test_keys.len() as f64;
    let blocked_fpr = blocked_fp as f64 / test_keys.len() as f64;

    println!("\n=== False Positive Rate Comparison (target: 1%) ===");
    println!(
        "Standard: {:.3}% ({}/{})",
        standard_fpr * 100.0,
        standard_fp,
        test_keys.len()
    );
    println!(
        "Blocked:  {:.3}% ({}/{})",
        blocked_fpr * 100.0,
        blocked_fp,
        test_keys.len()
    );
    println!(
        "FPR Ratio: {:.2}x (blocked/standard)",
        blocked_fpr / standard_fpr
    );

    group.finish();
}

fn benchmark_space_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_space");

    println!("\n=== Space Efficiency Comparison ===");
    for size in [100, 1000, 10_000, 100_000].iter() {
        let keys = generate_keys(*size, 0);

        // Standard
        let mut bf = BloomFilter::new(*size, 0.01);
        for key in &keys {
            bf.insert(key);
        }

        // Blocked
        let mut bbf = BlockedBloomFilter::new(*size, 0.01);
        for key in &keys {
            bbf.insert(key);
        }

        let standard_bytes = bf.size_bytes();
        let blocked_bytes = bbf.size_bytes();
        let overhead = ((blocked_bytes - standard_bytes) as f64 / standard_bytes as f64) * 100.0;

        println!(
            "{:>7} keys | Standard: {:>8} bytes ({:.2}/key) | Blocked: {:>8} bytes ({:.2}/key) | Overhead: {:>+5.1}%",
            size,
            standard_bytes,
            standard_bytes as f64 / *size as f64,
            blocked_bytes,
            blocked_bytes as f64 / *size as f64,
            overhead
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_insert_performance,
    benchmark_positive_lookups,
    benchmark_negative_lookups,
    benchmark_cache_misses,
    benchmark_false_positive_rate,
    benchmark_space_efficiency,
);
criterion_main!(benches);
