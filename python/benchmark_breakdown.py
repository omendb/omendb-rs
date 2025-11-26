#!/usr/bin/env python3
"""Comprehensive benchmark to identify the 15x Python binding slowdown"""

import omendb
import tempfile
import os
import numpy as np
import time

n = 10_000
dimensions = 128

print(f"\n{'='*60}")
print(f"Benchmarking {n:,} vectors ({dimensions}D)")
print(f"{'='*60}\n")

# Step 1: Create vectors
print("Step 1: Creating vectors...")
t0 = time.time()
vectors = [{
    'id': f'vec{i}',
    'embedding': np.random.randn(dimensions).astype(np.float32).tolist(),
    'metadata': {'idx': i}
} for i in range(n)]
create_time = time.time() - t0
print(f"  Time: {create_time:.3f}s\n")

# Step 2: Set
with tempfile.TemporaryDirectory() as tmpdir:
    db = omendb.open(os.path.join(tmpdir, 'test.db'), dimensions=dimensions)

    print("Step 2: Upserting (Python-measured)...")
    t0 = time.time()
    db.set(vectors)
    python_set_time = time.time() - t0
    print(f"  Time: {python_set_time:.3f}s")
    print(f"  Throughput: {n/python_set_time:,.0f} vec/s\n")

print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Vector creation:     {create_time:.3f}s")
print(f"Python set():     {python_set_time:.3f}s")
print(f"  (Rust internal timing shown above in [PROFILING] section)")
print(f"\nTotal Python time:   {create_time + python_set_time:.3f}s")
print(f"Throughput:          {n/python_set_time:,.0f} vec/s")
print(f"{'='*60}\n")
