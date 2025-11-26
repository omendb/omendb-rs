// Block parsing micro-benchmark
// Tests varint decoding and prefix reconstruction hot paths

use bytes::{Bytes, BytesMut};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::io::Cursor;

// Trait for varint decoding abstraction
trait VarintDecoder {
    fn decode_u64(cursor: &mut Cursor<&[u8]>) -> std::io::Result<u64>;
}

// Current implementation: varint-rs
struct VarintRs;
impl VarintDecoder for VarintRs {
    fn decode_u64(cursor: &mut Cursor<&[u8]>) -> std::io::Result<u64> {
        use varint_rs::VarintReader;
        cursor.read_u64_varint()
    }
}

// Generate test data: varint-encoded values
fn generate_varint_data(count: usize) -> Vec<u8> {
    use varint_rs::VarintWriter;
    let mut buf = Vec::new();

    // Mix of small (1-byte), medium (2-byte), and large (4-byte) varints
    for i in 0..count {
        match i % 3 {
            0 => buf.write_u64_varint(127).unwrap(),     // 1 byte
            1 => buf.write_u64_varint(16383).unwrap(),   // 2 bytes
            _ => buf.write_u64_varint(2097151).unwrap(), // 3 bytes
        }
    }

    buf
}

// Benchmark: Varint decoding (current: varint-rs)
fn bench_varint_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("varint_decoding");

    for count in [10, 50, 100, 500].iter() {
        let data = generate_varint_data(*count);

        group.bench_with_input(BenchmarkId::new("varint_rs", count), count, |b, _| {
            b.iter(|| {
                let mut cursor = Cursor::new(&data[..]);
                let mut sum = 0u64;

                for _ in 0..*count {
                    if let Ok(val) = VarintRs::decode_u64(&mut cursor) {
                        sum = sum.wrapping_add(val);
                    }
                }

                black_box(sum);
            });
        });
    }

    group.finish();
}

// Benchmark: Prefix reconstruction (key hot path)
fn bench_prefix_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_reconstruction");

    // Simulate block entry parsing: reconstruct keys from prefix + suffix
    let last_key = Bytes::from(b"user:12345:profile:name".to_vec());
    let suffix = Bytes::from(b":email".to_vec());

    for prefix_len in [0usize, 8, 16, 20].iter() {
        // Current implementation: BytesMut + extend_from_slice
        group.bench_with_input(
            BenchmarkId::new("bytesmut_extend", prefix_len),
            prefix_len,
            |b, &prefix_len| {
                b.iter(|| {
                    if prefix_len == 0 {
                        // Restart point: suffix is full key
                        black_box(suffix.clone());
                    } else {
                        // Reconstruct: prefix + suffix
                        let mut key_data = BytesMut::with_capacity(prefix_len + suffix.len());
                        key_data.extend_from_slice(&last_key[..prefix_len]);
                        key_data.extend_from_slice(&suffix);
                        black_box(key_data.freeze());
                    }
                });
            },
        );

        // Optimized: Vec + unsafe copy
        group.bench_with_input(
            BenchmarkId::new("vec_unsafe_copy", prefix_len),
            prefix_len,
            |b, &prefix_len| {
                b.iter(|| {
                    if prefix_len == 0 {
                        black_box(suffix.clone());
                    } else {
                        let suffix_len = suffix.len();
                        let mut key_data = Vec::with_capacity(prefix_len + suffix_len);

                        unsafe {
                            // Single copy: prefix + suffix
                            std::ptr::copy_nonoverlapping(
                                last_key[..prefix_len].as_ptr(),
                                key_data.as_mut_ptr(),
                                prefix_len,
                            );
                            std::ptr::copy_nonoverlapping(
                                suffix.as_ptr(),
                                key_data.as_mut_ptr().add(prefix_len),
                                suffix_len,
                            );
                            key_data.set_len(prefix_len + suffix_len);
                        }

                        black_box(Bytes::from(key_data));
                    }
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Full block parsing simulation
fn bench_block_parsing_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_parsing_full");

    // Simulate parsing 50 entries (typical 4KB block)
    let num_entries = 50;
    let varint_data = generate_varint_data(num_entries * 3); // prefix_len, suffix_len, value_len

    let last_key = Bytes::from(b"key:12345678:field".to_vec());
    let suffix = Bytes::from(b":value".to_vec());

    // Current implementation
    group.bench_function("current_impl", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(&varint_data[..]);
            let mut entries = Vec::with_capacity(num_entries);

            for _ in 0..num_entries {
                // Decode varints
                let prefix_len =
                    VarintRs::decode_u64(&mut cursor).unwrap() as usize % last_key.len();
                let suffix_len = VarintRs::decode_u64(&mut cursor).unwrap() as usize;
                let value_len = VarintRs::decode_u64(&mut cursor).unwrap() as usize;

                // Reconstruct key
                let key = if prefix_len == 0 {
                    suffix.clone()
                } else {
                    let mut key_data = BytesMut::with_capacity(prefix_len + suffix_len);
                    key_data.extend_from_slice(&last_key[..prefix_len]);
                    key_data.extend_from_slice(&suffix);
                    key_data.freeze()
                };

                entries.push((key, value_len));
            }

            black_box(entries);
        });
    });

    // Optimized: unsafe memcpy
    group.bench_function("optimized_memcpy", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(&varint_data[..]);
            let mut entries = Vec::with_capacity(num_entries);

            for _ in 0..num_entries {
                // Decode varints (same)
                let prefix_len =
                    VarintRs::decode_u64(&mut cursor).unwrap() as usize % last_key.len();
                let suffix_len = VarintRs::decode_u64(&mut cursor).unwrap() as usize;
                let value_len = VarintRs::decode_u64(&mut cursor).unwrap() as usize;

                // Reconstruct key (optimized)
                let key = if prefix_len == 0 {
                    suffix.clone()
                } else {
                    let mut key_data = Vec::with_capacity(prefix_len + suffix_len);
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            last_key[..prefix_len].as_ptr(),
                            key_data.as_mut_ptr(),
                            prefix_len,
                        );
                        std::ptr::copy_nonoverlapping(
                            suffix.as_ptr(),
                            key_data.as_mut_ptr().add(prefix_len),
                            suffix_len,
                        );
                        key_data.set_len(prefix_len + suffix_len);
                    }
                    Bytes::from(key_data)
                };

                entries.push((key, value_len));
            }

            black_box(entries);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_varint_decoding,
    bench_prefix_reconstruction,
    bench_block_parsing_simulation
);
criterion_main!(benches);
