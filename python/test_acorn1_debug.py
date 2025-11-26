#!/usr/bin/env python3
"""
Debug ACORN-1 to verify it's activating correctly
"""

import time
import numpy as np
import omendb

# Enable Rust logging to see ACORN-1 debug messages
import os
os.environ['RUST_LOG'] = 'OmenDB=debug'

def test_acorn1_activation():
    """Test that ACORN-1 activates at different selectivity levels"""

    print("\n" + "="*70)
    print("ACORN-1 Activation Test")
    print("="*70)

    # Create database with 50K vectors for better performance differentiation
    num_vectors = 50_000
    dimensions = 128
    print(f"Dataset: {num_vectors:,} vectors, {dimensions}D\n")

    db = omendb.open(":memory:")

    # Generate vectors with metadata
    print("Generating vectors...")
    vectors = []
    num_categories = 100  # 1% per category

    for i in range(num_vectors):
        vector = np.random.randn(dimensions).astype(np.float32).tolist()
        vectors.append({
            "id": f"vec_{i}",
            "embedding": vector,
            "metadata": {
                "category": i % num_categories,  # 1% per category
                "group": i % 20,  # 5% per group
                "cluster": i % 10,  # 10% per cluster
            }
        })

    print(f"Inserting {num_vectors:,} vectors...")
    t0 = time.time()
    db.set(vectors)
    print(f"✓ Inserted in {time.time()-t0:.2f}s\n")

    # Test query
    query = np.random.randn(dimensions).astype(np.float32).tolist()

    print("Testing different filters:\n")

    # 1. Very selective filter (1% - should use ACORN-1)
    print("1. Very selective filter (1% selectivity):")
    print("   Filter: category == 0")
    print("   Expected: ACORN-1 with 2-hop exploration")
    t0 = time.time()
    results1 = db.search(query=query, k=10, filter={"category": 0})
    t1 = time.time()
    print(f"   Results: {len(results1)} found in {(t1-t0)*1000:.2f}ms\n")

    # 2. Moderately selective filter (10% - should use ACORN-1)
    print("2. Moderately selective filter (10% selectivity):")
    print("   Filter: cluster == 0")
    print("   Expected: ACORN-1 without 2-hop")
    t0 = time.time()
    results2 = db.search(query=query, k=10, filter={"cluster": 0})
    t1 = time.time()
    print(f"   Results: {len(results2)} found in {(t1-t0)*1000:.2f}ms\n")

    # 3. Broad filter (70% - should fallback to standard search)
    print("3. Broad filter (70% selectivity):")
    print("   Filter: category in [0..69]")
    print("   Expected: Standard search + post-filter")
    t0 = time.time()
    results3 = db.search(query=query, k=10, filter={"category": {"$in": list(range(70))}})
    t1 = time.time()
    print(f"   Results: {len(results3)} found in {(t1-t0)*1000:.2f}ms\n")

    # 4. No filter baseline
    print("4. No filter (baseline):")
    t0 = time.time()
    results4 = db.search(query=query, k=10)
    t1 = time.time()
    print(f"   Results: {len(results4)} found in {(t1-t0)*1000:.2f}ms\n")

    print("="*70)
    print("Check Rust logs above for:")
    print("- 'Estimated filter selectivity: X.XX'")
    print("- 'Using ACORN-1 filtered search'")
    print("- 'Using 2-hop exploration'")
    print("- 'Filter too broad, using standard search'")
    print("="*70)

if __name__ == "__main__":
    test_acorn1_activation()
