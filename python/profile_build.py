#!/usr/bin/env python3
"""Profile build performance bottlenecks"""

import omendb
import tempfile
import os
import time
import numpy as np


def profile_batch_sizes():
    """Test different batch sizes to find optimal"""
    print("="*60)
    print("BATCH SIZE PROFILING")
    print("="*60)

    dimensions = 128
    total_vectors = 10_000

    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]

    for batch_size in batch_sizes:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = omendb.open(db_path, dimensions=dimensions)

            # Generate all vectors upfront
            all_vectors = [{
                "id": f"vec{i}",
                "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
                "metadata": {"idx": i}
            } for i in range(total_vectors)]

            # Insert in batches
            start = time.time()
            for i in range(0, total_vectors, batch_size):
                batch = all_vectors[i:i+batch_size]
                db.set(batch)
            elapsed = time.time() - start

            throughput = total_vectors / elapsed
            print(f"  Batch {batch_size:5d}: {throughput:7,.0f} vec/s ({elapsed:.2f}s)")


def profile_metadata_overhead():
    """Test if JSON metadata is the bottleneck"""
    print("\n" + "="*60)
    print("METADATA OVERHEAD TEST")
    print("="*60)

    dimensions = 128
    n_vectors = 10_000

    # Test 1: No metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "no_meta.db")
        db = omendb.open(db_path, dimensions=dimensions)

        vectors = [{
            "id": f"vec{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {}
        } for i in range(n_vectors)]

        start = time.time()
        db.set(vectors)
        no_meta_time = time.time() - start

    # Test 2: Small metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "small_meta.db")
        db = omendb.open(db_path, dimensions=dimensions)

        vectors = [{
            "id": f"vec{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {"idx": i}
        } for i in range(n_vectors)]

        start = time.time()
        db.set(vectors)
        small_meta_time = time.time() - start

    # Test 3: Large metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "large_meta.db")
        db = omendb.open(db_path, dimensions=dimensions)

        vectors = [{
            "id": f"vec{i}",
            "embedding": np.random.randn(dimensions).astype(np.float32).tolist(),
            "metadata": {
                "idx": i,
                "category": f"cat_{i % 10}",
                "text": f"This is document {i}" * 10,  # Larger metadata
                "tags": [f"tag{j}" for j in range(5)]
            }
        } for i in range(n_vectors)]

        start = time.time()
        db.set(vectors)
        large_meta_time = time.time() - start

    print(f"\n  No metadata:    {n_vectors/no_meta_time:,.0f} vec/s ({no_meta_time:.2f}s)")
    print(f"  Small metadata: {n_vectors/small_meta_time:,.0f} vec/s ({small_meta_time:.2f}s)")
    print(f"  Large metadata: {n_vectors/large_meta_time:,.0f} vec/s ({large_meta_time:.2f}s)")
    print(f"\n  Metadata overhead: {((large_meta_time - no_meta_time) / large_meta_time * 100):.1f}%")


def profile_python_overhead():
    """Measure Python list creation overhead"""
    print("\n" + "="*60)
    print("PYTHON OVERHEAD TEST")
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

    print(f"\n  Python list creation: {python_time:.2f}s")
    print(f"  Python overhead: {n_vectors/python_time:,.0f} vec/s")

    # Time the actual insert
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = omendb.open(db_path, dimensions=dimensions)

        start = time.time()
        db.set(vectors)
        insert_time = time.time() - start

    print(f"  Actual insert: {insert_time:.2f}s")
    print(f"  Insert throughput: {n_vectors/insert_time:,.0f} vec/s")
    print(f"\n  Python % of total: {python_time / (python_time + insert_time) * 100:.1f}%")


def compare_to_rust_core():
    """Compare to Rust core benchmarks"""
    print("\n" + "="*60)
    print("RUST CORE COMPARISON")
    print("="*60)

    print("\nRust core (from git history):")
    print("  Build: 7,094 QPS @ 10K vectors")
    print("  Query: 7,094 QPS")
    print("\nPython bindings (current):")
    print("  Build: ~2,265 vec/s")
    print("  Query: 3,224 QPS")
    print("\nOverhead:")
    print("  Build: 3.1x slower")
    print("  Query: 2.2x slower")
    print("\nExpected after optimizations:")
    print("  Build target: 15K-20K vec/s (beat ChromaDB's 24K)")
    print("  Query target: 5K-6K QPS (match Rust core)")


def main():
    print("\n" + "="*60)
    print("BUILD PERFORMANCE PROFILING")
    print("="*60)

    profile_batch_sizes()
    profile_metadata_overhead()
    profile_python_overhead()
    compare_to_rust_core()

    print("\n" + "="*60)
    print("OPTIMIZATION PRIORITIES")
    print("="*60)
    print("""
1. Cache index_to_id in VectorDatabase (avoid rebuild on every search)
2. Reduce JSON metadata serialization overhead
3. Optimize PyO3 conversions (use Cow, avoid clones)
4. Pre-allocate Python result vectors
5. Consider unsafe direct buffer access for embeddings
6. Profile Rust HNSW build to ensure it's not the bottleneck

Goal: Beat ChromaDB (24K vec/s build, 3K QPS query) by 2x
Target: 40K+ vec/s build, 6K+ QPS query
    """)


if __name__ == "__main__":
    main()
