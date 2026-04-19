#!/usr/bin/env python3
"""SIFT-10K benchmark: OmenDB vs hnswlib on real embeddings.

Downloads the SIFT-128-euclidean dataset from ann-benchmarks (HDF5 format),
builds indices with identical parameters, and compares recall@10 and QPS
across multiple ef_search settings.

Usage:
    python benchmarks/sift_benchmark.py
    python benchmarks/sift_benchmark.py --output results.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py
import hnswlib
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parent.parent


def rebuild_native_extension(argv: list[str]) -> bool:
    """Rebuild the editable native extension before importing omendb."""
    if (
        "--help" in argv
        or "-h" in argv
        or "--no-rebuild-native" in argv
        or os.environ.get("OMENDB_SKIP_NATIVE_REBUILD") == "1"
    ):
        return False

    print("Rebuilding Python native extension before benchmark...", flush=True)
    env = os.environ.copy()
    env.setdefault("RUSTC_BOOTSTRAP", "1")
    subprocess.run(
        ["uv", "run", "maturin", "develop", "--release"],
        cwd=PYTHON_ROOT,
        env=env,
        check=True,
    )
    return True


NATIVE_EXTENSION_REBUILT = rebuild_native_extension(sys.argv[1:])

import omendb  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main"

DATASETS = {
    "sift-10k": {
        "url": f"{HF_BASE}/siftsmall-128-euclidean.hdf5",
        "filename": "siftsmall-128-euclidean.hdf5",
        "size_mb": 5,
    },
    "sift-1m": {
        "url": f"{HF_BASE}/sift-128-euclidean.hdf5",
        "filename": "sift-128-euclidean.hdf5",
        "size_mb": 525,
    },
}

DATA_DIR = Path(__file__).parent / "data"

# Standard HNSW parameters
M = 16
EF_CONSTRUCTION = 100
K = 10
EF_SEARCH_VALUES = [50, 100, 200]


def download_dataset(dataset_key: str) -> Path:
    """Download dataset if not cached. Returns path to HDF5 file."""
    info = DATASETS[dataset_key]
    path = DATA_DIR / info["filename"]

    if path.exists():
        print(f"Using cached dataset: {path}")
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = info["url"]
    print(f"Downloading {dataset_key} from {url}...")
    print(f"(~{info['size_mb']} MB, one-time download)")

    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")
    return path


def load_dataset(dataset_key: str):
    """Load dataset: base vectors, queries, ground truth."""
    path = download_dataset(dataset_key)

    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])

    print(f"Loaded: {train.shape[0]:,} base, {test.shape[0]} queries, {train.shape[1]}D")
    return train, test, neighbors


def compute_recall(results_ids: list[set[int]], ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k."""
    total = 0.0
    for i, res_ids in enumerate(results_ids):
        true_ids = set(ground_truth[i, :k].tolist())
        total += len(res_ids & true_ids) / k
    return total / len(results_ids)


def get_metadata() -> dict:
    """System and version info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).parent.parent,
        ).strip()
    except Exception:
        commit = "unknown"

    return {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "omendb_version": getattr(omendb, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "native_rebuilt": NATIVE_EXTENSION_REBUILT,
    }


def benchmark_omendb(train, test, neighbors, ef_values):
    """Benchmark OmenDB on SIFT."""
    n, dim = train.shape
    n_queries = len(test)
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build index
        db = omendb.create(f"{tmpdir}/sift_db", {"dense": {"dim": dim}})
        db.ef_search = EF_CONSTRUCTION

        batch = [{"id": f"{i}", "vector": train[i].tolist()} for i in range(n)]

        start = time.time()
        db.set(batch)
        build_time = time.time() - start
        build_vps = n / build_time

        results["build"] = {
            "time_s": build_time,
            "vec_per_s": build_vps,
        }

        print(f"\n  Build: {build_vps:,.0f} vec/s ({build_time:.2f}s)")

        # Search at each ef_search value
        for ef in ef_values:
            db.ef_search = ef

            # Warmup
            for q in test[:10]:
                db.search(q.tolist(), k=K)

            # Timed search
            all_ids = []
            start = time.time()
            for q in test:
                res = db.search(q.tolist(), k=K)
                ids = {int(r["id"]) for r in res}
                all_ids.append(ids)
            search_time = time.time() - start
            qps = n_queries / search_time

            recall = compute_recall(all_ids, neighbors, K)

            results[f"ef_{ef}"] = {
                "ef_search": ef,
                "recall_at_10": recall,
                "qps": qps,
                "latency_ms": (search_time / n_queries) * 1000,
            }

            print(
                f"  ef={ef:<4d}  recall@10={recall:.4f}  QPS={qps:,.0f}  ({search_time / n_queries * 1000:.2f}ms/q)"
            )

    return results


def benchmark_hnswlib(train, test, neighbors, ef_values):
    """Benchmark hnswlib on SIFT with identical parameters."""
    n, dim = train.shape
    n_queries = len(test)
    results = {}

    # Build index
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=EF_CONSTRUCTION, M=M)

    start = time.time()
    index.add_items(train, np.arange(n))
    build_time = time.time() - start
    build_vps = n / build_time

    results["build"] = {
        "time_s": build_time,
        "vec_per_s": build_vps,
    }

    print(f"\n  Build: {build_vps:,.0f} vec/s ({build_time:.2f}s)")

    # Search at each ef_search value
    for ef in ef_values:
        index.set_ef(ef)

        # Warmup
        for q in test[:10]:
            index.knn_query(q, k=K)

        # Timed search (single-threaded, one query at a time)
        all_ids = []
        start = time.time()
        for q in test:
            labels, _ = index.knn_query(q, k=K)
            all_ids.append(set(labels[0].tolist()))
        search_time = time.time() - start
        qps = n_queries / search_time

        recall = compute_recall(all_ids, neighbors, K)

        results[f"ef_{ef}"] = {
            "ef_search": ef,
            "recall_at_10": recall,
            "qps": qps,
            "latency_ms": (search_time / n_queries) * 1000,
        }

        print(
            f"  ef={ef:<4d}  recall@10={recall:.4f}  QPS={qps:,.0f}  ({search_time / n_queries * 1000:.2f}ms/q)"
        )

    return results


def print_comparison(omen_results, hnsw_results, ef_values):
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 72}")
    print(f"SIFT Comparison: OmenDB v{getattr(omendb, '__version__', '?')} vs hnswlib")
    print(f"Config: M={M}, ef_construction={EF_CONSTRUCTION}, k={K}")
    print(f"{'=' * 72}")

    # Build comparison
    ob = omen_results["build"]
    hb = hnsw_results["build"]
    print(f"\n{'Build Throughput':>20s}  {'OmenDB':>12s}  {'hnswlib':>12s}  {'Ratio':>8s}")
    print(
        f"{'':>20s}  {ob['vec_per_s']:>10,.0f}/s  {hb['vec_per_s']:>10,.0f}/s  {ob['vec_per_s'] / hb['vec_per_s']:>7.2f}x"
    )

    # Search comparison
    print(
        f"\n{'ef_search':>10s}  {'Recall@10':>10s}  {'QPS':>10s}  {'Recall@10':>10s}  {'QPS':>10s}  {'Recall Gap':>10s}  {'QPS Ratio':>10s}"
    )
    print(f"{'':>10s}  {'--- OmenDB ---':>22s}  {'--- hnswlib ---':>22s}  {'':>22s}")

    for ef in ef_values:
        ok = f"ef_{ef}"
        o = omen_results[ok]
        h = hnsw_results[ok]
        recall_gap = o["recall_at_10"] - h["recall_at_10"]
        qps_ratio = o["qps"] / h["qps"] if h["qps"] > 0 else float("inf")

        print(
            f"{ef:>10d}  "
            f"{o['recall_at_10']:>10.4f}  {o['qps']:>10,.0f}  "
            f"{h['recall_at_10']:>10.4f}  {h['qps']:>10,.0f}  "
            f"{recall_gap:>+10.4f}  "
            f"{qps_ratio:>9.2f}x"
        )

    print(f"\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(description="SIFT benchmark: OmenDB vs hnswlib")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="sift-10k",
        help="Dataset to benchmark (default: sift-10k)",
    )
    parser.add_argument(
        "--no-rebuild-native",
        action="store_true",
        help="Skip the default maturin rebuild before importing the native extension",
    )
    args = parser.parse_args()

    metadata = get_metadata()
    dataset_key = args.dataset

    print("=" * 72)
    print(f"SIFT Benchmark: OmenDB vs hnswlib ({dataset_key})")
    print("=" * 72)
    print(f"Commit:   {metadata['commit']}")
    print(f"OmenDB:   {metadata['omendb_version']}")
    print(f"Python:   {metadata['python']}")
    print(f"Platform: {metadata['platform']}")
    print(f"CPU:      {metadata['cpu']}")
    print(f"Config:   M={M}, ef_construction={EF_CONSTRUCTION}, k={K}")
    print(f"ef_search values: {EF_SEARCH_VALUES}")

    # Load dataset
    train, test, neighbors = load_dataset(dataset_key)

    # Benchmark OmenDB
    print(f"\n--- OmenDB {metadata['omendb_version']} ---")
    omen_results = benchmark_omendb(train, test, neighbors, EF_SEARCH_VALUES)

    # Benchmark hnswlib
    print("\n--- hnswlib ---")
    hnsw_results = benchmark_hnswlib(train, test, neighbors, EF_SEARCH_VALUES)

    # Print comparison
    print_comparison(omen_results, hnsw_results, EF_SEARCH_VALUES)

    # Save results
    output = {
        "metadata": metadata,
        "config": {
            "dataset": dataset_key,
            "n_base": int(train.shape[0]),
            "n_queries": int(test.shape[0]),
            "dimensions": int(train.shape[1]),
            "M": M,
            "ef_construction": EF_CONSTRUCTION,
            "k": K,
            "ef_search_values": EF_SEARCH_VALUES,
        },
        "omendb": omen_results,
        "hnswlib": hnsw_results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    # Always save to benchmark data dir
    results_path = DATA_DIR / f"{dataset_key}_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
