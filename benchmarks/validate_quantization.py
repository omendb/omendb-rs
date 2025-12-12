#!/usr/bin/env python3
"""
Quantization Recall & Performance Validation

Validates recall and performance claims for all quantization modes:
- f32 (baseline): Full precision
- sq8: 4x compression, ~99% recall
- rabitq: 8x compression, ~96% recall
- rabitq-2: 16x compression, ~93% recall
- rabitq-8: 4x compression, ~99% recall

Usage:
    python benchmarks/validate_quantization.py           # 10K vectors (quick)
    python benchmarks/validate_quantization.py --scale 50000   # 50K vectors
    python benchmarks/validate_quantization.py --scale 100000  # 100K vectors
"""

import argparse
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import omendb


@dataclass
class ValidationResult:
    mode: str
    n_vectors: int
    dimensions: int
    recall_at_10: float
    recall_at_100: float
    single_qps: float
    batch_qps: float
    build_time_s: float

    def __str__(self):
        return (
            f"{self.mode:12} | {self.recall_at_10:5.1%} | {self.recall_at_100:5.1%} | "
            f"{self.single_qps:>7,.0f} | {self.batch_qps:>8,.0f} | {self.build_time_s:5.1f}s"
        )


def generate_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate normalized random vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def brute_force_knn(query: np.ndarray, vectors: np.ndarray, k: int) -> list[int]:
    """Compute ground truth k-NN using brute force."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k].tolist()


def compute_recall(hnsw_ids: list[str], ground_truth_indices: list[int]) -> float:
    """Compute recall between HNSW results and ground truth."""
    hnsw_indices = {int(id.split("_")[1]) for id in hnsw_ids}
    gt_set = set(ground_truth_indices)
    return len(hnsw_indices & gt_set) / len(gt_set)


def validate_mode(
    mode: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth_10: list[list[int]],
    ground_truth_100: list[list[int]],
    dimensions: int,
) -> ValidationResult:
    """Validate a single quantization mode."""
    n_vectors = len(vectors)
    n_queries = len(queries)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build database
        build_start = time.perf_counter()

        if mode == "f32":
            db = omendb.open(f"{tmpdir}/db", dimensions=dimensions)
        else:
            db = omendb.open(f"{tmpdir}/db", dimensions=dimensions, quantization=mode)

        items = [{"id": f"v_{i}", "vector": v.tolist()} for i, v in enumerate(vectors)]
        db.set(items)

        build_time = time.perf_counter() - build_start

        # Measure recall@10
        recall_10_sum = 0.0
        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=10)
            result_ids = [r["id"] for r in results]
            recall_10_sum += compute_recall(result_ids, ground_truth_10[i])
        recall_at_10 = recall_10_sum / n_queries

        # Measure recall@100
        recall_100_sum = 0.0
        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=100)
            result_ids = [r["id"] for r in results]
            recall_100_sum += compute_recall(result_ids, ground_truth_100[i])
        recall_at_100 = recall_100_sum / n_queries

        # Warmup
        for q in queries[:3]:
            db.search(q.tolist(), k=10)

        # Measure single-query QPS
        iterations = 5
        single_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for q in queries:
                db.search(q.tolist(), k=10)
            single_times.append(time.perf_counter() - start)
        single_time = np.median(single_times)
        single_qps = n_queries / single_time

        # Measure batch QPS
        query_list = [q.tolist() for q in queries]
        batch_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            db.search_batch(query_list, k=10)
            batch_times.append(time.perf_counter() - start)
        batch_time = np.median(batch_times)
        batch_qps = n_queries / batch_time

    return ValidationResult(
        mode=mode,
        n_vectors=n_vectors,
        dimensions=dimensions,
        recall_at_10=recall_at_10,
        recall_at_100=recall_at_100,
        single_qps=single_qps,
        batch_qps=batch_qps,
        build_time_s=build_time,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate quantization recall and performance"
    )
    parser.add_argument(
        "--scale", type=int, default=10000, help="Number of vectors (default: 10000)"
    )
    parser.add_argument(
        "--dim", type=int, default=768, help="Vector dimensions (default: 768)"
    )
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of queries (default: 100)"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="f32,sq8,rabitq,rabitq-2,rabitq-8",
        help="Comma-separated modes to test",
    )
    args = parser.parse_args()

    n_vectors = args.scale
    dimensions = args.dim
    n_queries = args.queries
    modes = [m.strip() for m in args.modes.split(",")]

    print(
        f"\nQuantization Validation: {n_vectors:,} vectors, {dimensions}D, {n_queries} queries"
    )
    print("=" * 80)

    # Generate data
    print("\nGenerating vectors...", end=" ", flush=True)
    vectors = generate_vectors(n_vectors, dimensions, seed=42)
    queries = generate_vectors(n_queries, dimensions, seed=12345)
    print(f"done ({vectors.nbytes / 1e6:.1f} MB)")

    # Compute ground truth
    print("Computing ground truth...", end=" ", flush=True)
    ground_truth_10 = [brute_force_knn(q, vectors, k=10) for q in queries]
    ground_truth_100 = [brute_force_knn(q, vectors, k=100) for q in queries]
    print("done")

    # Run validation for each mode
    print("\n" + "-" * 80)
    print(
        f"{'Mode':12} | {'R@10':>5} | {'R@100':>5} | {'S-QPS':>7} | {'B-QPS':>8} | Build"
    )
    print("-" * 80)

    results = []
    for mode in modes:
        print(f"Testing {mode}...", end="\r", file=sys.stderr, flush=True)
        result = validate_mode(
            mode, vectors, queries, ground_truth_10, ground_truth_100, dimensions
        )
        results.append(result)
        print(result)

    print("-" * 80)

    # Summary with expected vs actual
    print("\nRecall Summary (Expected vs Actual):")
    print("-" * 50)

    expected = {
        "f32": (1.00, "baseline"),
        "sq8": (0.99, "~99%"),
        "rabitq": (0.96, "~96%"),
        "rabitq-2": (0.93, "~93%"),
        "rabitq-8": (0.99, "~99%"),
    }

    baseline_qps = next((r.single_qps for r in results if r.mode == "f32"), None)

    for r in results:
        exp_recall, exp_str = expected.get(r.mode, (None, "?"))
        status = (
            "PASS"
            if exp_recall is None or r.recall_at_10 >= exp_recall - 0.05
            else "WARN"
        )
        qps_ratio = r.single_qps / baseline_qps if baseline_qps else 0
        print(
            f"{r.mode:12}: {r.recall_at_10:5.1%} (expected {exp_str:>5}) [{status}] {qps_ratio:.2f}x QPS"
        )

    print()


if __name__ == "__main__":
    main()
