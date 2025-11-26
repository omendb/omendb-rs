#!/usr/bin/env python3
"""Benchmark search/query performance (QPS and latency)"""

import omendb
import tempfile
import os
import numpy as np
import time

n_vectors = 10_000
n_queries = 1_000
dimensions = 128
k = 10

print(f"\n{'='*60}")
print(f"Search Performance Benchmark")
print(f"{'='*60}")
print(f"Dataset: {n_vectors:,} vectors ({dimensions}D)")
print(f"Queries: {n_queries:,} searches (k={k})")
print(f"{'='*60}\n")

# Build index
print("Building index...")
vectors = [{
    'id': f'vec{i}',
    'embedding': np.random.randn(dimensions).astype(np.float32).tolist(),
    'metadata': {'idx': i}
} for i in range(n_vectors)]

with tempfile.TemporaryDirectory() as tmpdir:
    db = omendb.open(os.path.join(tmpdir, 'test.db'), dimensions=dimensions)

    t0 = time.time()
    db.set(vectors)
    build_time = time.time() - t0
    print(f"  Built {n_vectors:,} vectors in {build_time:.3f}s ({n_vectors/build_time:,.0f} vec/s)\n")

    # Generate random queries
    print("Generating queries...")
    queries = [np.random.randn(dimensions).astype(np.float32).tolist() for _ in range(n_queries)]
    print(f"  Generated {n_queries:,} random queries\n")

    # Warmup
    print("Warmup (10 queries)...")
    for i in range(10):
        db.search(queries[i], k=k)
    print("  Done\n")

    # Benchmark individual query latency
    print("Benchmarking individual query latency...")
    latencies = []
    for query in queries[:100]:  # Sample 100 queries for latency measurement
        t0 = time.time()
        results = db.search(query, k=k)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    p99 = latencies[int(len(latencies)*0.99)]
    avg = sum(latencies) / len(latencies)

    print(f"  Latency (100 queries):")
    print(f"    p50: {p50:.2f}ms")
    print(f"    p95: {p95:.2f}ms")
    print(f"    p99: {p99:.2f}ms")
    print(f"    avg: {avg:.2f}ms\n")

    # Benchmark throughput (QPS)
    print(f"Benchmarking throughput ({n_queries:,} queries)...")
    t0 = time.time()
    for query in queries:
        results = db.search(query, k=k)
    total_time = time.time() - t0
    qps = n_queries / total_time

    print(f"  Total time: {total_time:.3f}s")
    print(f"  QPS: {qps:,.0f} queries/sec")
    print(f"  Avg latency: {(total_time/n_queries)*1000:.2f}ms\n")

    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset size: {n_vectors:,} vectors")
    print(f"QPS:          {qps:,.0f} queries/sec")
    print(f"Latency p50:  {p50:.2f}ms")
    print(f"Latency p95:  {p95:.2f}ms")
    print(f"Latency p99:  {p99:.2f}ms")
    print(f"{'='*60}\n")

    # Compare to target
    print("Comparison to ChromaDB (Nov 2025 benchmark):")
    print(f"  ChromaDB:  3,282 QPS, 0.30ms p50")
    print(f"  OmenDB:      {qps:,.0f} QPS, {p50:.2f}ms p50")
    if qps < 3282:
        speedup_needed = 3282 / qps
        print(f"  Need {speedup_needed:.1f}x faster to match ChromaDB")
    else:
        print(f"  ✅ Faster than ChromaDB!")
    print()
