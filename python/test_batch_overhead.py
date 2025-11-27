#!/usr/bin/env python3
"""Test to isolate Python batch insert overhead"""

import omendb
import tempfile
import os
import time
import numpy as np


def test_batch_sizes():
    """Test different batch sizes to find overhead pattern"""
    print("="*60)
    print("BATCH SIZE OVERHEAD TEST")
    print("="*60)

    dimensions = 128
    total_vectors = 10_000

    # Test different batch sizes
    batch_sizes = [100, 500, 1000, 2500, 5000, 10000]

    for batch_size in batch_sizes:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = omendb.open(db_path, dimensions=dimensions)

            # Generate vectors upfront
            all_vectors = [{
                "id": f"vec{i}",
                "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
                "metadata": {"idx": i}
            } for i in range(total_vectors)]

            # Time the set
            start = time.time()

            for i in range(0, total_vectors, batch_size):
                batch = all_vectors[i:i+batch_size]
                db.set(batch)

            elapsed = time.time() - start
            throughput = total_vectors / elapsed

            print(f"  Batch {batch_size:5d}: {throughput:7,.0f} vec/s ({elapsed:.2f}s)")


def test_rust_direct_comparison():
    """See if Rust core matches standalone benchmark"""
    print("\n" + "="*60)
    print("RUST CORE COMPARISON")
    print("="*60)

    dimensions = 128
    n_vectors = 10_000

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = omendb.open(db_path, dimensions=dimensions)

        # Single batch (should be fastest)
        vectors = [{
            "id": f"vec{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {"idx": i}
        } for i in range(n_vectors)]

        start = time.time()
        db.set(vectors)
        elapsed = time.time() - start

        throughput = n_vectors / elapsed

        print("\nPython bindings (single 10K batch):")
        print(f"  Throughput: {throughput:,.0f} vec/s")
        print(f"  Time: {elapsed:.2f}s")

        print("\nRust benchmark (parallel batch_insert):")
        print("  Throughput: 36,915 vec/s (from benchmark_parallel_hnsw)")
        print("  Time: 0.27s")

        print(f"\nGap: {36915 / throughput:.1f}x slower through Python bindings")


def test_minimal_overhead():
    """Measure just Python→Rust conversion overhead"""
    print("\n" + "="*60)
    print("MINIMAL OVERHEAD TEST")
    print("="*60)

    dimensions = 128
    n_vectors = 10_000

    # Time just creating the Python list
    start = time.time()
    vectors = [{
        "id": f"vec{i}",
        "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
        "metadata": {"idx": i}
    } for i in range(n_vectors)]
    python_time = time.time() - start

    print(f"\nPython list creation: {python_time:.3f}s ({n_vectors/python_time:,.0f} vec/s)")

    # Time the set (which includes Rust HNSW build)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = omendb.open(db_path, dimensions=dimensions)

        start = time.time()
        db.set(vectors)
        total_time = time.time() - start

    rust_time = total_time - python_time

    print(f"Total set time: {total_time:.3f}s ({n_vectors/total_time:,.0f} vec/s)")
    print(f"Rust portion: {rust_time:.3f}s ({n_vectors/rust_time:,.0f} vec/s)")
    print(f"Python overhead: {python_time/total_time*100:.1f}%")


if __name__ == "__main__":
    test_batch_sizes()
    test_rust_direct_comparison()
    test_minimal_overhead()
