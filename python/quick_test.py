#!/usr/bin/env python3
"""Quick comparison test for development iteration (<10 seconds)

Run this after making changes to quickly check if performance improved.
"""

import omendb
import chromadb
import hnswlib
import numpy as np
import time
import tempfile
import os

# Small dataset for quick testing
N_VECTORS = 1_000  # Small for speed
N_QUERIES = 100
DIM = 128
K = 10

def p95(values):
    return np.percentile(values, 95)

def benchmark_omendb():
    """Benchmark OmenDB"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(os.path.join(tmpdir, 'test.db'), dimensions=DIM)

        # Build
        vectors = [{
            'id': f'vec{i}',
            'embedding': np.random.randn(DIM).astype(np.float32).tolist(),
            'metadata': {'idx': i}
        } for i in range(N_VECTORS)]

        t0 = time.time()
        db.set(vectors)
        build_time = time.time() - t0

        # Search
        queries = [np.random.randn(DIM).astype(np.float32).tolist() for _ in range(N_QUERIES)]

        # Warmup
        for i in range(5):
            db.search(queries[i], k=K)

        latencies = []
        for query in queries:
            t0 = time.time()
            db.search(query, k=K)
            latencies.append(time.time() - t0)

        return {
            'build_vps': N_VECTORS / build_time,
            'qps': N_QUERIES / sum(latencies),
            'p50_ms': np.median(latencies) * 1000,
            'p95_ms': p95(latencies) * 1000,
        }

def benchmark_chromadb():
    """Benchmark ChromaDB (baseline)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        collection = client.create_collection("test")

        # Build
        ids = [f'vec{i}' for i in range(N_VECTORS)]
        embeddings = [np.random.randn(DIM).astype(np.float32).tolist() for _ in range(N_VECTORS)]
        metadatas = [{'idx': i} for i in range(N_VECTORS)]

        t0 = time.time()
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        build_time = time.time() - t0

        # Search
        queries = [np.random.randn(DIM).astype(np.float32).tolist() for _ in range(N_QUERIES)]

        # Warmup
        for i in range(5):
            collection.query(query_embeddings=[queries[i]], n_results=K)

        latencies = []
        for query in queries:
            t0 = time.time()
            collection.query(query_embeddings=[query], n_results=K)
            latencies.append(time.time() - t0)

        return {
            'build_vps': N_VECTORS / build_time,
            'qps': N_QUERIES / sum(latencies),
            'p50_ms': np.median(latencies) * 1000,
            'p95_ms': p95(latencies) * 1000,
        }

def benchmark_hnswlib():
    """Benchmark hnswlib (C++ baseline)"""
    index = hnswlib.Index(space='l2', dim=DIM)
    index.init_index(max_elements=N_VECTORS, ef_construction=200, M=16)

    # Build
    embeddings = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    ids = np.arange(N_VECTORS)

    t0 = time.time()
    index.add_items(embeddings, ids)
    build_time = time.time() - t0

    # Search
    index.set_ef(200)
    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)

    # Warmup
    for i in range(5):
        index.knn_query(queries[i:i+1], k=K)

    latencies = []
    for query in queries:
        t0 = time.time()
        index.knn_query(query, k=K)
        latencies.append(time.time() - t0)

    return {
        'build_vps': N_VECTORS / build_time,
        'qps': N_QUERIES / sum(latencies),
        'p50_ms': np.median(latencies) * 1000,
        'p95_ms': p95(latencies) * 1000,
    }

if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"Quick Development Test ({N_VECTORS:,} vectors, {N_QUERIES} queries)")
    print(f"{'='*70}\n")

    print("Running omendb...")
    omendb_results = benchmark_omendb()

    print("Running ChromaDB...")
    chroma_results = benchmark_chromadb()

    print("Running hnswlib (C++)...")
    hnswlib_results = benchmark_hnswlib()

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}\n")

    # Build comparison
    print("Build Performance (vectors/sec):")
    print(f"  OmenDB:     {omendb_results['build_vps']:>10,.0f}")
    print(f"  ChromaDB: {chroma_results['build_vps']:>10,.0f} ({chroma_results['build_vps']/omendb_results['build_vps']:.2f}x)")
    print(f"  hnswlib:  {hnswlib_results['build_vps']:>10,.0f} ({hnswlib_results['build_vps']/omendb_results['build_vps']:.2f}x)")

    # Query comparison
    print("\nQuery Performance (QPS):")
    print(f"  OmenDB:     {omendb_results['qps']:>10,.0f}")
    print(f"  ChromaDB: {chroma_results['qps']:>10,.0f} ({chroma_results['qps']/omendb_results['qps']:.2f}x)")
    print(f"  hnswlib:  {hnswlib_results['qps']:>10,.0f} ({hnswlib_results['qps']/omendb_results['qps']:.2f}x)")

    print("\nQuery Latency p95 (ms):")
    print(f"  OmenDB:     {omendb_results['p95_ms']:>7.2f}")
    print(f"  ChromaDB: {chroma_results['p95_ms']:>7.2f}")
    print(f"  hnswlib:  {hnswlib_results['p95_ms']:>7.2f}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")

    omendb_vs_chroma_qps = omendb_results['qps'] / chroma_results['qps']
    omendb_vs_hnswlib_qps = omendb_results['qps'] / hnswlib_results['qps']

    if omendb_vs_chroma_qps >= 1.0:
        print(f"✅ OmenDB is {omendb_vs_chroma_qps:.2f}x vs ChromaDB (GOOD)")
    else:
        print(f"⚠️  OmenDB is {1/omendb_vs_chroma_qps:.2f}x SLOWER than ChromaDB")

    if omendb_vs_hnswlib_qps >= 0.5:
        print(f"✅ OmenDB is {omendb_vs_hnswlib_qps:.2f}x vs hnswlib (C++)")
    else:
        print(f"❌ OmenDB is {1/omendb_vs_hnswlib_qps:.2f}x SLOWER than hnswlib (needs work)")

    # Goal: match hnswlib (pure C++ HNSW)
    gap = hnswlib_results['qps'] / omendb_results['qps']
    print(f"\nGap to close: {gap:.2f}x to match C++ HNSW baseline")
    print("Target after SOTA optimizations: 2-3x faster than hnswlib")
