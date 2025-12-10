"""Quick benchmark: OmenDB vs ChromaDB"""
import tempfile
import time
import numpy as np
import omendb

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("ChromaDB not installed")

def generate_vectors(n: int, dim: int) -> np.ndarray:
    return np.random.randn(n, dim).astype(np.float32)

def bench_omendb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    dim = vectors.shape[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(f"{tmpdir}/omendb", dimensions=dim)
        
        # Insert
        items = [{"id": f"d{i}", "vector": v.tolist()} for i, v in enumerate(vectors)]
        start = time.perf_counter()
        db.set(items)
        insert_time = time.perf_counter() - start
        
        # Warmup
        for q in queries[:5]:
            db.search(q.tolist(), k=k)
        db.search_batch([q.tolist() for q in queries[:5]], k=k)
        
        # Single search
        start = time.perf_counter()
        for q in queries:
            db.search(q.tolist(), k=k)
        single_time = time.perf_counter() - start
        
        # Batch search
        query_list = [q.tolist() for q in queries]
        start = time.perf_counter()
        db.search_batch(query_list, k=k)
        batch_time = time.perf_counter() - start
        
        return {
            "single_qps": len(queries) / single_time,
            "batch_qps": len(queries) / batch_time,
        }

def bench_chromadb(vectors: np.ndarray, queries: np.ndarray, k: int = 10):
    if not HAS_CHROMA:
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=f"{tmpdir}/chroma")
        collection = client.create_collection("bench")
        
        # Insert in batches (ChromaDB limit: 5461)
        batch_size = 5000
        start = time.perf_counter()
        for i in range(0, len(vectors), batch_size):
            end = min(i + batch_size, len(vectors))
            ids = [f"d{j}" for j in range(i, end)]
            collection.add(ids=ids, embeddings=vectors[i:end].tolist())
        insert_time = time.perf_counter() - start
        
        # Warmup
        for q in queries[:5]:
            collection.query(query_embeddings=[q.tolist()], n_results=k)
        
        # Single search
        start = time.perf_counter()
        for q in queries:
            collection.query(query_embeddings=[q.tolist()], n_results=k)
        single_time = time.perf_counter() - start
        
        # Batch search
        start = time.perf_counter()
        collection.query(query_embeddings=queries.tolist(), n_results=k)
        batch_time = time.perf_counter() - start
        
        return {
            "single_qps": len(queries) / single_time,
            "batch_qps": len(queries) / batch_time,
        }

def main():
    n_vectors = 10_000
    n_queries = 100
    k = 10
    
    print(f"\n=== Benchmark: OmenDB vs ChromaDB ===")
    print(f"Dataset: {n_vectors} vectors, {n_queries} queries, k={k}\n")
    
    print("| Dimension | Metric | OmenDB | ChromaDB | Improvement |")
    print("|-----------|--------|--------|----------|-------------|")
    
    for dim in [128, 768, 1536]:
        np.random.seed(42)
        vectors = generate_vectors(n_vectors, dim)
        queries = generate_vectors(n_queries, dim)
        
        omen = bench_omendb(vectors, queries, k)
        chroma = bench_chromadb(vectors, queries, k)
        
        if chroma:
            for metric in ["single_qps", "batch_qps"]:
                o_val = omen[metric]
                c_val = chroma[metric]
                improvement = ((o_val / c_val) - 1) * 100
                label = "Single" if "single" in metric else "Batch"
                print(f"| {dim}D | {label} | {o_val:,.0f} QPS | {c_val:,.0f} QPS | +{improvement:.0f}% |")

if __name__ == "__main__":
    main()
