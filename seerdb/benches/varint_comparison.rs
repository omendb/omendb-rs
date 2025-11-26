// Varint decoding comparison: varint-rs vs varint-simd
// Tests whether SIMD varint provides meaningful speedup for block parsing

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::io::Cursor;

// Generate test data: varint-encoded values
fn generate_varint_data(count: usize) -> Vec<u8> {
    use varint_rs::VarintWriter;
    let mut buf = Vec::new();

    // Mix of small (1-byte), medium (2-byte), and large (3-byte) varints
    // Matches typical SSTable block entry metadata
    for i in 0..count {
        match i % 3 {
            0 => buf.write_u64_varint(127).unwrap(), // 1 byte: prefix_len
            1 => buf.write_u64_varint(16383).unwrap(), // 2 bytes: suffix_len
            _ => buf.write_u64_varint(2097151).unwrap(), // 3 bytes: value_len
        }
    }

    buf
}

// Benchmark: varint-rs (current)
fn bench_varint_rs(c: &mut Criterion) {
    use varint_rs::VarintReader;

    let mut group = c.benchmark_group("varint_rs");

    for count in [10, 50, 100, 500].iter() {
        let data = generate_varint_data(*count);

        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&data[..]);
                let mut sum = 0u64;

                for _ in 0..*count {
                    if let Ok(val) = cursor.read_u64_varint() {
                        sum = sum.wrapping_add(val);
                    }
                }

                black_box(sum);
            });
        });
    }

    group.finish();
}

// Benchmark: varint-simd (SIMD-accelerated)
#[cfg(feature = "varint-simd")]
fn bench_varint_simd(c: &mut Criterion) {
    use varint_simd::{decode, VarIntTarget};

    let mut group = c.benchmark_group("varint_simd");

    for count in [10, 50, 100, 500].iter() {
        let data = generate_varint_data(*count);

        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, _| {
            b.iter(|| {
                let mut offset = 0usize;
                let mut sum = 0u64;

                for _ in 0..*count {
                    if let Some((val, bytes_read)) = decode::<u64>(&data[offset..]) {
                        sum = sum.wrapping_add(val);
                        offset += bytes_read;
                    }
                }

                black_box(sum);
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "varint-simd"))]
fn bench_varint_simd(_c: &mut Criterion) {
    println!("varint-simd benchmarks skipped - enable 'varint-simd' feature to run");
}

// Benchmark: Full block parsing simulation (50 entries typical)
fn bench_full_block_simulation(c: &mut Criterion) {
    use varint_rs::VarintReader;

    let num_entries = 50;
    let varint_data = generate_varint_data(num_entries * 3); // prefix_len, suffix_len, value_len

    c.bench_function("full_block_50_entries_varint_rs", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(&varint_data[..]);
            let mut sum = 0u64;

            for _ in 0..num_entries {
                // Decode 3 varints per entry
                if let Ok(prefix_len) = cursor.read_u64_varint() {
                    sum = sum.wrapping_add(prefix_len);
                }
                if let Ok(suffix_len) = cursor.read_u64_varint() {
                    sum = sum.wrapping_add(suffix_len);
                }
                if let Ok(value_len) = cursor.read_u64_varint() {
                    sum = sum.wrapping_add(value_len);
                }
            }

            black_box(sum);
        });
    });

    // SIMD version
    #[cfg(feature = "varint-simd")]
    {
        use varint_simd::{decode, VarIntTarget};

        c.bench_function("full_block_50_entries_varint_simd", |b| {
            b.iter(|| {
                let mut offset = 0usize;
                let mut sum = 0u64;

                for _ in 0..num_entries {
                    // Decode 3 varints per entry
                    if let Some((val, bytes_read)) = decode::<u64>(&varint_data[offset..]) {
                        sum = sum.wrapping_add(val);
                        offset += bytes_read;
                    }
                    if let Some((val, bytes_read)) = decode::<u64>(&varint_data[offset..]) {
                        sum = sum.wrapping_add(val);
                        offset += bytes_read;
                    }
                    if let Some((val, bytes_read)) = decode::<u64>(&varint_data[offset..]) {
                        sum = sum.wrapping_add(val);
                        offset += bytes_read;
                    }
                }

                black_box(sum);
            });
        });
    }
}

criterion_group!(
    benches,
    bench_varint_rs,
    bench_varint_simd,
    bench_full_block_simulation
);
criterion_main!(benches);
