// Storage abstraction overhead benchmark
// Verifies that file handle reuse has zero overhead compared to direct file I/O

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

// Generate test data: 1MB file with 256 blocks of 4KB each
fn create_test_file(path: &PathBuf) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;

    // Write 256 blocks of 4KB (1MB total)
    let block = vec![0xABu8; 4096];
    for _ in 0..256 {
        file.write_all(&block)?;
    }
    file.sync_all()?;
    Ok(())
}

// Benchmark: Direct file I/O (baseline)
fn bench_direct_file_io(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.dat");
    create_test_file(&file_path).unwrap();

    c.bench_function("direct_file_read_4kb", |b| {
        b.iter(|| {
            let mut file = File::open(&file_path).unwrap();
            let mut buf = vec![0u8; 4096];

            // Read 10 random blocks
            for offset in [
                0u64, 16384, 32768, 65536, 131072, 262144, 524288, 786432, 917504, 1044480,
            ] {
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });
}

// Benchmark: File handle reuse (optimized SSTable approach)
fn bench_file_handle_reuse(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.dat");
    create_test_file(&file_path).unwrap();

    // Open file once and reuse handle (matches optimized SSTable)
    let file = File::open(&file_path).unwrap();
    let file_handle = Arc::new(Mutex::new(file));

    c.bench_function("file_handle_reuse_4kb", |b| {
        b.iter(|| {
            // Read 10 random blocks using shared handle
            for offset in [
                0u64, 16384, 32768, 65536, 131072, 262144, 524288, 786432, 917504, 1044480,
            ] {
                let mut f = file_handle.lock().unwrap();
                f.seek(SeekFrom::Start(offset)).unwrap();
                let mut buf = vec![0u8; 4096];
                f.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });
}

// Benchmark: Sequential read throughput comparison
fn bench_sequential_throughput(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.dat");
    create_test_file(&file_path).unwrap();

    let mut group = c.benchmark_group("sequential_read_throughput");

    // Direct file I/O
    group.bench_function("direct_file_sequential", |b| {
        b.iter(|| {
            let mut file = File::open(&file_path).unwrap();
            let mut buf = vec![0u8; 4096];

            // Read all 256 blocks sequentially
            for offset in (0..1048576).step_by(4096) {
                file.seek(SeekFrom::Start(offset as u64)).unwrap();
                file.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });

    // File handle reuse (optimized SSTable approach)
    let file = File::open(&file_path).unwrap();
    let file_handle = Arc::new(Mutex::new(file));
    group.bench_function("file_handle_reuse_sequential", |b| {
        b.iter(|| {
            // Read all 256 blocks sequentially using shared handle
            for offset in (0..1048576).step_by(4096) {
                let mut f = file_handle.lock().unwrap();
                f.seek(SeekFrom::Start(offset as u64)).unwrap();
                let mut buf = vec![0u8; 4096];
                f.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });

    group.finish();
}

// Benchmark: Random read patterns (simulates SSTable block cache misses)
fn bench_random_reads(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.dat");
    create_test_file(&file_path).unwrap();

    let mut group = c.benchmark_group("random_read_patterns");

    // Generate random offsets (aligned to 4KB blocks)
    let random_offsets: Vec<u64> = vec![
        0, 16384, 8192, 32768, 4096, 49152, 24576, 65536, 12288, 40960, 20480, 57344, 28672, 73728,
        36864, 81920,
    ];

    // Direct file I/O
    group.bench_function("direct_file_random", |b| {
        b.iter(|| {
            let mut file = File::open(&file_path).unwrap();
            let mut buf = vec![0u8; 4096];

            for &offset in &random_offsets {
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });

    // File handle reuse (optimized SSTable approach)
    let file = File::open(&file_path).unwrap();
    let file_handle = Arc::new(Mutex::new(file));
    group.bench_function("file_handle_reuse_random", |b| {
        b.iter(|| {
            for &offset in &random_offsets {
                let mut f = file_handle.lock().unwrap();
                f.seek(SeekFrom::Start(offset)).unwrap();
                let mut buf = vec![0u8; 4096];
                f.read_exact(&mut buf).unwrap();
                black_box(&buf);
            }
        });
    });

    group.finish();
}

// Benchmark: Small reads (index blocks, metadata)
fn bench_small_reads(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.dat");
    create_test_file(&file_path).unwrap();

    let mut group = c.benchmark_group("small_read_sizes");

    // Test various sizes: 64B, 256B, 1KB, 4KB
    for size in [64usize, 256, 1024, 4096].iter() {
        // Direct file I/O
        group.bench_with_input(BenchmarkId::new("direct_file", size), size, |b, &size| {
            b.iter(|| {
                let mut file = File::open(&file_path).unwrap();
                let mut buf = vec![0u8; size];

                for offset in [0u64, 8192, 16384, 32768, 65536] {
                    file.seek(SeekFrom::Start(offset)).unwrap();
                    file.read_exact(&mut buf).unwrap();
                    black_box(&buf);
                }
            });
        });

        // File handle reuse (optimized SSTable approach)
        let file = File::open(&file_path).unwrap();
        let file_handle = Arc::new(Mutex::new(file));
        group.bench_with_input(
            BenchmarkId::new("file_handle_reuse", size),
            size,
            |b, &size| {
                b.iter(|| {
                    for offset in [0u64, 8192, 16384, 32768, 65536] {
                        let mut f = file_handle.lock().unwrap();
                        f.seek(SeekFrom::Start(offset)).unwrap();
                        let mut buf = vec![0u8; size];
                        f.read_exact(&mut buf).unwrap();
                        black_box(&buf);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_file_io,
    bench_file_handle_reuse,
    bench_sequential_throughput,
    bench_random_reads,
    bench_small_reads
);
criterion_main!(benches);
