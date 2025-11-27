#!/usr/bin/env python3
"""
Simple ACORN-1 test with random data (no sampling bias)
"""

import time
import numpy as np
import omendb

def test_acorn1_simple():
    """Test ACORN-1 with random category assignment"""

    print("\n" + "="*70)
    print("ACORN-1 Simple Test")
    print("="*70)

    # Use smaller dataset for quick testing
    num_vectors = 10_000
    dimensions = 128
    print(f"Dataset: {num_vectors:,} vectors, {dimensions}D\n")

    db = omendb.open(":memory:")

    # Generate vectors with RANDOM category assignment (no pattern bias)
    print("Generating vectors...")
    vectors = []
    rng = np.random.default_rng(42)
    categories = rng.integers(0, 20, size=num_vectors)  # Random categories 0-19

    for i in range(num_vectors):
        vector = np.random.randn(dimensions).astype(np.float32).tolist()
        vectors.append({
            "id": f"vec_{i}",
            "embedding": vector,
            "metadata": {
                "category": int(categories[i]),  # Random assignment
            }
        })

    # Count actual distribution
    category_0_count = np.sum(categories == 0)
    actual_selectivity = category_0_count / num_vectors

    print(f"Inserting {num_vectors:,} vectors...")
    t0 = time.time()
    db.set(vectors)
    insert_time = time.time() - t0
    print(f"✓ Inserted in {insert_time:.2f}s ({num_vectors/insert_time:.0f} vec/s)\n")

    print("Actual distribution:")
    print(f"  Category 0: {category_0_count:,} vectors ({actual_selectivity*100:.1f}% selectivity)")
    print()

    # Generate queries
    num_queries = 50
    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(num_queries)]

    # Baseline (no filter)
    print("Baseline (no filter):")
    t0 = time.time()
    for q in queries:
        db.search(query=q, k=10)
    t_baseline = time.time() - t0
    qps_baseline = num_queries / t_baseline
    print(f"  {qps_baseline:8.0f} QPS ({t_baseline/num_queries*1000:5.2f}ms per query)\n")

    # With filter (ACORN-1 or post-filtering depending on selectivity)
    print(f"With filter (category=0, {actual_selectivity*100:.1f}% selectivity):")
    t0 = time.time()
    for q in queries:
        db.search(query=q, k=10, filter={"category": 0})
    t_filtered = time.time() - t0
    qps_filtered = num_queries / t_filtered
    speedup = qps_filtered / qps_baseline
    print(f"  {qps_filtered:8.0f} QPS ({t_filtered/num_queries*1000:5.2f}ms per query)")
    print(f"  Speedup: {speedup:.2f}x\n")

    # Interpretation
    if actual_selectivity < 0.1:
        expected_min_speedup = 2.0
        algorithm = "ACORN-1 with 2-hop"
    elif actual_selectivity < 0.6:
        expected_min_speedup = 1.5
        algorithm = "ACORN-1"
    else:
        expected_min_speedup = 1.0
        algorithm = "Post-filtering"

    print(f"Expected algorithm: {algorithm}")
    print(f"Expected minimum speedup: {expected_min_speedup:.1f}x")

    if speedup >= expected_min_speedup:
        print(f"✓ PASS: Achieved {speedup:.2f}x speedup")
    else:
        print(f"⚠ Modest speedup (expected >{expected_min_speedup:.1f}x)")

    print("="*70)

if __name__ == "__main__":
    test_acorn1_simple()
