use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use seerdb::sstable::block::{Block, BlockBuilder};

fn bench_block_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_parsing_cost");

    // Prepare random data (moderately compressible)
    let mut rng = StdRng::seed_from_u64(42);
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for i in 0..50 {
        keys.push(format!("key{:04}", i).into_bytes());
        // Random values
        let mut val = vec![0u8; 64];
        rng.fill(&mut val[..]);
        values.push(val);
    }

    // 1. Compressed Block
    let mut builder = BlockBuilder::new();
    builder.set_compression(true);
    for (k, v) in keys.iter().zip(values.iter()) {
        if !builder.add(k, v) {
            break;
        }
    }
    let compressed_bytes = builder.finish();

    // 2. Uncompressed Block
    let mut builder = BlockBuilder::new();
    builder.set_compression(false);
    for (k, v) in keys.iter().zip(values.iter()) {
        if !builder.add(k, v) {
            break;
        }
    }
    let uncompressed_bytes = builder.finish();

    println!("Compressed size: {}", compressed_bytes.len());
    println!("Uncompressed size: {}", uncompressed_bytes.len());

    // Benchmark Block::new() only (Parsing + Decompression)
    group.bench_function("compressed_new", |b| {
        b.iter(|| Block::from_bytes(black_box(compressed_bytes.clone())).unwrap())
    });

    group.bench_function("uncompressed_new", |b| {
        b.iter(|| Block::from_bytes(black_box(uncompressed_bytes.clone())).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_block_parsing);
criterion_main!(benches);
