#!/usr/bin/env python3
"""
Quantization Recall & Performance Validation

Validates recall and performance claims for quantization modes using real SIFT
embeddings (128D). Uses precomputed ground truth from the SIFT dataset.

- f32 (baseline): Full precision
- sq8: 4x compression, ~99% recall

Usage:
    python benchmarks/validate_quantization.py           # SIFT-10K, 128D
    python benchmarks/validate_quantization.py --modes f32  # f32 only
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

SIFT_10K_PATH = Path(__file__).parent / "data" / "sift-10k.npz"


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


def load_sift() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT-10K dataset with precomputed ground truth."""
    if not SIFT_10K_PATH.exists():
        print(f"SIFT-10K not found at {SIFT_10K_PATH}", file=sys.stderr)
        print("Download sift-10k.npz and place it in benchmarks/data/", file=sys.stderr)
        sys.exit(1)
    data = np.load(SIFT_10K_PATH)
    return data["vectors"], data["queries"], data["ground_truth"]


def compute_recall(
    hnsw_ids: list[str], ground_truth_indices: np.ndarray, k: int
) -> float:
    """Compute recall@k between HNSW results and precomputed ground truth."""
    hnsw_indices = {int(id.split("_")[1]) for id in hnsw_ids}
    gt_set = set(ground_truth_indices[:k].tolist())
    return len(hnsw_indices & gt_set) / k


def validate_mode(
    mode: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> ValidationResult:
    n_vectors, dimensions = vectors.shape
    n_queries = len(queries)

    with tempfile.TemporaryDirectory() as tmpdir:
        if mode == "f32":
            db = omendb.open(f"{tmpdir}/db", dimensions=dimensions)
        else:
            db = omendb.open(f"{tmpdir}/db", dimensions=dimensions, quantization=mode)

        items = [{"id": f"v_{i}", "vector": v.tolist()} for i, v in enumerate(vectors)]
        build_start = time.perf_counter()
        db.set(items)
        build_time = time.perf_counter() - build_start

        # Recall@10
        recall_10_sum = 0.0
        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=10)
            result_ids = [r["id"] for r in results]
            recall_10_sum += compute_recall(result_ids, ground_truth[i], k=10)
        recall_at_10 = recall_10_sum / n_queries

        # Recall@100
        recall_100_sum = 0.0
        for i, query in enumerate(queries):
            results = db.search(query.tolist(), k=100)
            result_ids = [r["id"] for r in results]
            recall_100_sum += compute_recall(result_ids, ground_truth[i], k=100)
        recall_at_100 = recall_100_sum / n_queries

        # Warmup
        for q in queries[:10]:
            db.search(q.tolist(), k=10)

        # Single QPS (median of 5)
        single_times = []
        for _ in range(5):
            t = time.perf_counter()
            for q in queries:
                db.search(q.tolist(), k=10)
            single_times.append(time.perf_counter() - t)
        single_qps = n_queries / float(np.median(single_times))

        # Batch QPS (median of 5)
        query_list = [q.tolist() for q in queries]
        batch_times = []
        for _ in range(5):
            t = time.perf_counter()
            db.search_batch(query_list, k=10)
            batch_times.append(time.perf_counter() - t)
        batch_qps = n_queries / float(np.median(batch_times))

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
        "--modes",
        type=str,
        default="f32,sq8",
        help="Comma-separated modes to test (default: f32,sq8)",
    )
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",")]

    print("Loading SIFT-10K...", end=" ", flush=True)
    vectors, queries, ground_truth = load_sift()
    n_vectors, dimensions = vectors.shape
    print(f"done ({n_vectors:,} vectors, {dimensions}D, {len(queries)} queries)")

    print(f"\nQuantization Validation: {n_vectors:,} vectors, {dimensions}D")
    print("=" * 80)
    print(
        f"\n{'Mode':12} | {'R@10':>5} | {'R@100':>5} | {'S-QPS':>7} | {'B-QPS':>8} | Build"
    )
    print("-" * 80)

    results = []
    for mode in modes:
        print(f"Testing {mode}...", end="\r", file=sys.stderr, flush=True)
        result = validate_mode(mode, vectors, queries, ground_truth)
        results.append(result)
        print(result)

    print("-" * 80)

    print("\nRecall Summary (Expected vs Actual):")
    print("-" * 50)

    expected = {
        "f32": (1.00, "baseline"),
        "sq8": (0.99, "~99%"),
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
