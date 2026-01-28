#!/usr/bin/env python3
"""Multi-vector (MUVERA) validation benchmark.

Validates recall and throughput for multi-vector search with exact MaxSim ground truth.

Usage:
    python benchmarks/multivec_validation.py          # Quick (100 docs)
    python benchmarks/multivec_validation.py --docs 1000  # Full validation
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

import omendb


@dataclass
class Config:
    n_docs: int = 100
    n_queries: int = 20
    tokens_per_doc: tuple = (5, 20)  # (min, max) tokens per document
    tokens_per_query: tuple = (3, 10)
    token_dim: int = 128
    k: int = 10
    seed: int = 42
    # MUVERA config: "fast" (default), "balanced", "quality"
    muvera_preset: str = "balanced"


def generate_clustered_tokens(
    n_docs: int,
    tokens_range: tuple,
    dim: int,
    n_clusters: int = 10,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate documents with clustered token embeddings (realistic structure)."""
    rng = np.random.default_rng(seed)

    # Create cluster centers
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    docs = []
    for i in range(n_docs):
        n_tokens = rng.integers(tokens_range[0], tokens_range[1] + 1)
        # Each doc draws from 1-3 clusters (topic coherence)
        doc_clusters = rng.choice(n_clusters, size=min(3, n_clusters), replace=False)

        tokens = []
        for _ in range(n_tokens):
            cluster = rng.choice(doc_clusters)
            # Token = cluster center + small noise
            token = centers[cluster] + rng.standard_normal(dim).astype(np.float32) * 0.3
            token /= np.linalg.norm(token)
            tokens.append(token)

        docs.append(np.array(tokens, dtype=np.float32))

    return docs


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    """Exact MaxSim: sum of max similarities for each query token."""
    if len(query_tokens) == 0 or len(doc_tokens) == 0:
        return 0.0
    # Similarity matrix: [n_query_tokens, n_doc_tokens]
    sims = query_tokens @ doc_tokens.T
    # Max similarity for each query token, then sum
    return float(np.sum(np.max(sims, axis=1)))


def compute_ground_truth(
    queries: list[np.ndarray],
    docs: list[np.ndarray],
    k: int,
) -> list[list[int]]:
    """Compute ground truth top-k for each query via brute force MaxSim."""
    ground_truth = []
    for query in queries:
        scores = [(i, maxsim_score(query, doc)) for i, doc in enumerate(docs)]
        scores.sort(key=lambda x: -x[1])
        ground_truth.append([idx for idx, _ in scores[:k]])
    return ground_truth


def compute_recall(results: list[dict], ground_truth: list[int], k: int) -> float:
    """Compute recall@k."""
    result_ids = {int(r["id"]) for r in results[:k]}
    gt_ids = set(ground_truth[:k])
    return len(result_ids & gt_ids) / k


def run_benchmark(cfg: Config):
    print(f"=== Multi-Vector Validation ({cfg.n_docs} docs, {cfg.n_queries} queries) ===\n")

    # Generate data
    print("Generating synthetic clustered tokens...")
    t0 = time.perf_counter()
    docs = generate_clustered_tokens(cfg.n_docs, cfg.tokens_per_doc, cfg.token_dim, seed=cfg.seed)
    queries = generate_clustered_tokens(
        cfg.n_queries, cfg.tokens_per_query, cfg.token_dim, seed=cfg.seed + 1000
    )
    gen_time = time.perf_counter() - t0
    print(f"  Generated in {gen_time:.2f}s")

    # Compute ground truth
    print("Computing ground truth (brute force MaxSim)...")
    t0 = time.perf_counter()
    ground_truth = compute_ground_truth(queries, docs, cfg.k)
    gt_time = time.perf_counter() - t0
    print(f"  Ground truth in {gt_time:.2f}s")

    # MUVERA presets - tradeoff between speed and FDE approximation quality
    presets = {
        "fast": {"repetitions": 5, "partition_bits": 3},  # 5*8*D = 5,120D
        "balanced": {"repetitions": 8, "partition_bits": 4},  # 8*16*D = 16,384D
        "quality": {"repetitions": 10, "partition_bits": 4},  # 10*16*D = 20,480D
    }
    muvera_config = presets[cfg.muvera_preset]
    fde_dim = muvera_config["repetitions"] * (2 ** muvera_config["partition_bits"]) * cfg.token_dim

    print("Building OmenDB multi-vector index...")
    print(
        f"  MUVERA: {cfg.muvera_preset} (reps={muvera_config['repetitions']}, bits={muvera_config['partition_bits']}, FDE={fde_dim}D)"
    )
    db = omendb.open(":memory:", dimensions=cfg.token_dim, multi_vector=muvera_config)

    t0 = time.perf_counter()
    records = [{"id": str(i), "vectors": doc.tolist()} for i, doc in enumerate(docs)]
    db.set(records)
    build_time = time.perf_counter() - t0
    build_throughput = cfg.n_docs / build_time
    print(f"  Built in {build_time:.3f}s ({build_throughput:.0f} docs/s)")

    # Search WITHOUT reranking (FDE only)
    print("\nSearching (FDE only, no rerank)...")
    t0 = time.perf_counter()
    recalls_fde = []
    for i, query in enumerate(queries):
        results = db.search(query.tolist(), k=cfg.k, rerank=False)
        recalls_fde.append(compute_recall(results, ground_truth[i], cfg.k))
    fde_time = time.perf_counter() - t0
    fde_qps = cfg.n_queries / fde_time
    fde_recall = np.mean(recalls_fde)

    # Search WITH reranking (FDE + MaxSim) - try different rerank factors
    print("Searching (FDE + MaxSim rerank)...")
    rerank_factors = [4, 8, 16, 32, 64]
    rerank_results = {}

    for rf in rerank_factors:
        t0 = time.perf_counter()
        recalls = []
        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=cfg.k, rerank=True, rerank_factor=rf)
            recalls.append(compute_recall(results, ground_truth[i], cfg.k))
        elapsed = time.perf_counter() - t0
        rerank_results[rf] = {
            "recall": np.mean(recalls),
            "qps": cfg.n_queries / elapsed,
        }

    # Use best rerank factor for reporting
    best_rf = max(rerank_results.keys(), key=lambda x: rerank_results[x]["recall"])
    rerank_recall = rerank_results[best_rf]["recall"]
    rerank_qps = rerank_results[best_rf]["qps"]

    # For backwards compat
    recalls_rerank = [rerank_recall]  # dummy
    rerank_time = cfg.n_queries / rerank_qps
    rerank_time = time.perf_counter() - t0
    rerank_qps = cfg.n_queries / rerank_time
    rerank_recall = np.mean(recalls_rerank)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nConfig: {cfg.n_docs} docs, {cfg.token_dim}D tokens, k={cfg.k}")
    print(f"Tokens per doc: {cfg.tokens_per_doc}, per query: {cfg.tokens_per_query}")
    print()
    print(f"| Mode              | Recall@{cfg.k} | QPS    |")
    print("|-------------------|----------|--------|")
    print(f"| FDE only          | {fde_recall:>7.1%} | {fde_qps:>6.0f} |")
    for rf, res in sorted(rerank_results.items()):
        print(f"| Rerank (factor={rf:>2}) | {res['recall']:>7.1%} | {res['qps']:>6.0f} |")
    print()
    print(f"Build: {build_throughput:.0f} docs/s")
    print(f"Ground truth: {gt_time:.2f}s")

    # Pass/fail
    passed = rerank_recall >= 0.80
    print()
    if passed:
        print(f"PASS: Rerank recall {rerank_recall:.1%} >= 80%")
    else:
        print(f"FAIL: Rerank recall {rerank_recall:.1%} < 80%")

    return passed, {
        "n_docs": cfg.n_docs,
        "fde_recall": fde_recall,
        "rerank_recall": rerank_recall,
        "fde_qps": fde_qps,
        "rerank_qps": rerank_qps,
        "build_docs_per_s": build_throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-vector validation benchmark")
    parser.add_argument("--docs", type=int, default=100, help="Number of documents")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Token dimension")
    parser.add_argument("--k", type=int, default=10, help="Top-k for recall")
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="MUVERA quality preset",
    )
    args = parser.parse_args()

    cfg = Config(
        n_docs=args.docs,
        n_queries=args.queries,
        token_dim=args.dim,
        k=args.k,
        muvera_preset=args.preset,
    )

    passed, _ = run_benchmark(cfg)
    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
