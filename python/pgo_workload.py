"""PGO profiling workload"""
import numpy as np
import tempfile
import omendb

N, D, K = 10000, 128, 10
print(f"PGO workload: {N} vectors, {D}D")

vectors = np.random.rand(N, D).astype(np.float32)
queries = np.random.rand(100, D).astype(np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    db = omendb.open(f"{tmpdir}/db", dimensions=D, m=16, ef_construction=100)
    batch = [{"id": f"d{i}", "vector": vectors[i].tolist()} for i in range(N)]
    db.set(batch)
    for q in queries:
        db.search(q.tolist(), k=K)
    db.search_batch([q.tolist() for q in queries], k=K)
    db.close()

print("Workload complete")
