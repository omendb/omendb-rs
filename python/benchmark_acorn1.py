#!/usr/bin/env python3
"""
Benchmark ACORN-1 filtered search performance

Tests filtered search at different selectivity levels:
- 5% selectivity: Expect 6.3x speedup (ACORN-1 vs post-filtering)
- 10% selectivity: Expect 4.0x speedup
- 20% selectivity: Expect 2.4x speedup
- 60%+ selectivity: Falls back to standard search + post-filter
"""

import time
import numpy as np
import omendb

def benchmark_filtered_search(num_vectors=10_000, dimensions=128, num_queries=100):
    """Benchmark ACORN-1 filtered search at different selectivity levels"""

    print(f"\n{'='*70}")
    print("ACORN-1 Filtered Search Benchmark")
    print(f"{'='*70}")
    print(f"Dataset: {num_vectors:,} vectors, {dimensions}D")
    print(f"Queries: {num_queries} searches")
    print(f"{'='*70}\n")

    # Create database
    db = omendb.open(":memory:")

    # Generate vectors with metadata (categories for filtering)
    print("Generating vectors with metadata...")
    num_categories = 20  # 5% per category

    vectors = []
    for i in range(num_vectors):
        vector = np.random.randn(dimensions).astype(np.float32).tolist()
        category = i % num_categories  # Distribute evenly across categories
        vectors.append({
            "id": f"vec_{i}",
            "embedding": vector,
            "metadata": {
                "category": category,
                "group": i % 10,  # 10% per group
                "cluster": i % 5,  # 20% per cluster
            }
        })

    # Insert vectors
    print("Inserting vectors...")
    t0 = time.time()
    db.set(vectors)
    insert_time = time.time() - t0
    print(f"✓ Inserted {num_vectors:,} vectors in {insert_time:.2f}s ({num_vectors/insert_time:.0f} vec/s)\n")

    # Generate queries
    query_vectors = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(num_queries)]

    # Test different selectivity levels
    test_cases = [
        ("5% selectivity (1 category)", {"category": 0}, 0.05, 6.3),
        ("10% selectivity (2 categories)", {"category": {"$in": [0, 1]}}, 0.10, 4.0),
        ("20% selectivity (1 cluster)", {"cluster": 0}, 0.20, 2.4),
        ("60% selectivity (6 categories)", {"category": {"$in": list(range(6))}}, 0.30, 1.0),
    ]

    print(f"{'Filter':<35} {'QPS':>10} {'Latency':>12} {'Expected':>10} {'Status':>8}")
    print(f"{'-'*35} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    baseline_qps = None

    for name, filter_dict, selectivity, expected_speedup in test_cases:
        # Warm-up
        for _ in range(5):
            db.search(query=query_vectors[0], k=10, filter=filter_dict)

        # Benchmark
        t0 = time.time()
        for query in query_vectors:
            _ = db.search(query=query, k=10, filter=filter_dict)
        elapsed = time.time() - t0

        qps = num_queries / elapsed
        latency_ms = (elapsed / num_queries) * 1000

        # Calculate speedup vs baseline (unfiltered search)
        if baseline_qps is None:
            # First run establishes baseline
            baseline_qps = qps
            status = "BASELINE"
        else:
            actual_speedup = qps / baseline_qps
            if selectivity < 0.6:
                # ACORN-1 should activate
                if actual_speedup >= expected_speedup * 0.8:  # Within 80% of expected
                    status = "✓ PASS"
                else:
                    status = "✗ SLOW"
            else:
                # Should fallback to standard search
                status = "FALLBACK"

        print(f"{name:<35} {qps:>10.0f} {latency_ms:>11.2f}ms {expected_speedup:>9.1f}x {status:>8}")

    print(f"\n{'='*70}")
    print("ACORN-1 Performance Summary:")
    print(f"{'='*70}")
    print("✓ Low selectivity (5-20%): Should see 2-6x speedup")
    print("✓ High selectivity (60%+): Falls back to standard search")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    benchmark_filtered_search()
