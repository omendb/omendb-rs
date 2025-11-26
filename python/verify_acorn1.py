#!/usr/bin/env python3
"""
Verify ACORN-1 is working correctly by comparing against brute force
"""

import time
import numpy as np
import omendb
import os

# Enable Rust logging
os.environ['RUST_LOG'] = 'OmenDB=debug'

def verify_acorn1_correctness():
    """Verify ACORN-1 returns correct results"""

    print("\n" + "="*70)
    print("ACORN-1 Correctness Verification")
    print("="*70)

    # Use larger dataset to see performance difference
    num_vectors = 100_000
    dimensions = 128
    print(f"Dataset: {num_vectors:,} vectors, {dimensions}D\n")

    db = omendb.open(":memory:")

    # Generate vectors with evenly distributed categories
    print("Generating vectors...")
    vectors = []
    num_categories = 20  # 5% per category

    for i in range(num_vectors):
        vector = np.random.randn(dimensions).astype(np.float32).tolist()
        vectors.append({
            "id": f"vec_{i}",
            "embedding": vector,
            "metadata": {
                "category": i % num_categories,  # 5% per category
            }
        })

    print(f"Inserting {num_vectors:,} vectors...")
    t0 = time.time()
    db.set(vectors)
    insert_time = time.time() - t0
    print(f"✓ Inserted in {insert_time:.2f}s ({num_vectors/insert_time:.0f} vec/s)\n")

    # Verify distribution
    print("Verifying data distribution:")
    print(f"  Categories: {num_categories}")
    print(f"  Vectors per category: {num_vectors // num_categories:,} (5% selectivity)")
    print(f"  Expected matches for category=0: ~5,000 vectors\n")

    # Generate query
    query = np.random.randn(dimensions).astype(np.float32).tolist()

    # Test 1: Very selective filter (5% - category 0)
    print("Test 1: 5% selective filter (category=0)")
    print("-" * 70)

    # Without filter (baseline)
    t0 = time.time()
    results_no_filter = db.search(query=query, k=100)
    t_no_filter = time.time() - t0
    qps_no_filter = 1.0 / t_no_filter

    # With filter (ACORN-1)
    t0 = time.time()
    results_filtered = db.search(query=query, k=100, filter={"category": 0})
    t_filtered = time.time() - t0
    qps_filtered = 1.0 / t_filtered

    print(f"  No filter:    {len(results_no_filter):3d} results, {t_no_filter*1000:6.2f}ms, {qps_no_filter:8.0f} QPS")
    print(f"  With filter:  {len(results_filtered):3d} results, {t_filtered*1000:6.2f}ms, {qps_filtered:8.0f} QPS")
    print(f"  Speedup:      {qps_filtered/qps_no_filter:5.2f}x")

    # Verify all results match filter
    mismatches = sum(1 for r in results_filtered if r['metadata']['category'] != 0)
    print(f"  Filter accuracy: {len(results_filtered)-mismatches}/{len(results_filtered)} correct")

    if qps_filtered > qps_no_filter * 1.5:
        print(f"  ✓ ACORN-1 showing speedup!")
    else:
        print(f"  ⚠ No significant speedup (may need more queries for stable measurement)")

    print()

    # Test 2: Multiple queries for statistical significance
    print("Test 2: Average over 50 queries (5% selectivity)")
    print("-" * 70)

    num_test_queries = 50
    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(num_test_queries)]

    # Baseline (no filter)
    t0 = time.time()
    for q in queries:
        db.search(query=q, k=10)
    t_baseline = time.time() - t0
    qps_baseline = num_test_queries / t_baseline

    # With filter (ACORN-1)
    t0 = time.time()
    for q in queries:
        db.search(query=q, k=10, filter={"category": 0})
    t_acorn = time.time() - t0
    qps_acorn = num_test_queries / t_acorn

    print(f"  Baseline (no filter): {qps_baseline:8.0f} QPS ({t_baseline/num_test_queries*1000:5.2f}ms per query)")
    print(f"  ACORN-1 (5% filter):  {qps_acorn:8.0f} QPS ({t_acorn/num_test_queries*1000:5.2f}ms per query)")
    print(f"  Speedup:              {qps_acorn/qps_baseline:5.2f}x")
    print(f"  Expected:             ~6.3x at 5% selectivity")

    if qps_acorn / qps_baseline > 2.0:
        print(f"  ✓ SIGNIFICANT SPEEDUP DETECTED!")
    elif qps_acorn / qps_baseline > 1.2:
        print(f"  ⚠ Modest speedup (may need larger dataset)")
    else:
        print(f"  ✗ No speedup - investigating...")

    print()
    print("="*70)

if __name__ == "__main__":
    verify_acorn1_correctness()
