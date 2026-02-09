#!/usr/bin/env python3
"""
OmenDB Benchmark Runner

Runs benchmarks on SIFT-10K (real embeddings) with QPS and recall measurement.

Usage:
    python benchmarks/run.py                        # SIFT-10K benchmark (~30s)
    python benchmarks/run.py --quick                # Quick run (~10s)
    python benchmarks/run.py --output FILE          # Save results to FILE
    python benchmarks/run.py --history              # Show history
    python benchmarks/run.py --compare              # Compare last 2 runs
    python benchmarks/run.py --notes "text"         # Add notes to run
    python benchmarks/run.py -q sq8                 # Test SQ8 quantization
    python benchmarks/run.py --all-modes            # Test fp32 and SQ8

Save to cloud/ for canonical history:
    python benchmarks/run.py --output ../../cloud/benchmarks/history.jsonl
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure we can import omendb
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import omendb

DEFAULT_HISTORY_FILE = Path(__file__).parent / "history.jsonl"
SIFT_10K_PATH = Path(__file__).parent / "data" / "sift-10k.npz"


@dataclass
class BenchmarkConfig:
    n_vectors: int
    n_queries: int
    dimensions: int
    k: int
    ef: Optional[int] = None
    m: int = 16
    ef_construction: int = 100
    quantization: Optional[str] = None  # None or "sq8"
    dataset: str = "sift-10k"


@dataclass
class BenchmarkResult:
    name: str
    config: dict
    build_vec_per_s: int
    single_qps: float
    batch_qps: float
    single_latency_ms: float
    batch_latency_ms: float
    speedup: float
    recall_at_10: Optional[float] = None


def load_sift_10k() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT-10K dataset (real embeddings with pre-computed ground truth)."""
    if not SIFT_10K_PATH.exists():
        print(
            f"SIFT-10K not found at {SIFT_10K_PATH}. "
            "Run: python benchmarks/create_subsets.py",
            file=sys.stderr,
        )
        sys.exit(1)

    data = np.load(SIFT_10K_PATH)
    return data["vectors"], data["queries"], data["ground_truth"]


def compute_recall(result_ids: list[str], ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k between HNSW results and brute force ground truth."""
    hnsw_indices = {int(rid) for rid in result_ids}
    gt_set = set(ground_truth[:k].tolist())
    return len(hnsw_indices & gt_set) / k


def get_system_info() -> dict:
    """Collect system information."""
    cpu = "Unknown"
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    ram_gb = 0.0
    try:
        if platform.system() == "Darwin":
            ram_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            )
            ram_gb = ram_bytes / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024**2)
                        break
    except Exception:
        pass

    return {
        "cpu": cpu,
        "cores": os.cpu_count() or 0,
        "ram_gb": round(ram_gb, 1),
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
        "host": platform.node().split(".")[0],
    }


def get_git_info() -> dict:
    """Collect git repository information."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            != ""
        )
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def get_version_info() -> dict:
    """Collect version information."""
    rust_version = "unknown"
    try:
        out = subprocess.check_output(["rustc", "--version"], text=True).strip()
        rust_version = out.split()[1] if out else "unknown"
    except Exception:
        pass

    return {
        "rust": rust_version,
        "python": platform.python_version(),
        "omendb": getattr(omendb, "__version__", "unknown"),
    }


def run_benchmark(
    config: BenchmarkConfig,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    quick: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark configuration on provided dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(
            f"{tmpdir}/bench",
            dimensions=config.dimensions,
            m=config.m,
            ef_construction=config.ef_construction,
            quantization=config.quantization,
        )

        # Build index and measure throughput
        items = [{"id": str(i), "vector": v.tolist()} for i, v in enumerate(vectors)]
        build_start = time.perf_counter()
        db.set(items)
        build_time = time.perf_counter() - build_start
        build_vec_per_s = round(len(vectors) / build_time)

        # Measure recall
        recall_sum = 0.0
        for i, q in enumerate(queries):
            results = db.search(q.tolist(), k=config.k)
            result_ids = [r["id"] for r in results]
            recall_sum += compute_recall(result_ids, ground_truth[i], config.k)
        recall_at_10 = recall_sum / len(queries)

        # Warmup
        for q in queries[:10]:
            db.search(q.tolist(), k=config.k)
        db.search_batch([q.tolist() for q in queries[:10]], k=config.k)

        # Single-query benchmark
        iterations = 3 if quick else 10
        single_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            for q in queries:
                db.search(q.tolist(), k=config.k)
            single_times.append(time.perf_counter() - start)

        single_time = np.median(single_times)
        single_qps = config.n_queries / single_time
        single_latency_ms = (single_time / config.n_queries) * 1000

        # Batch benchmark
        batch_times = []
        query_list = [q.tolist() for q in queries]
        for _ in range(iterations):
            start = time.perf_counter()
            db.search_batch(query_list, k=config.k)
            batch_times.append(time.perf_counter() - start)

        batch_time = np.median(batch_times)
        batch_qps = config.n_queries / batch_time
        batch_latency_ms = (batch_time / config.n_queries) * 1000

    # Build name with quantization mode
    quant_suffix = ""
    if config.quantization == "sq8":
        quant_suffix = "_sq8"

    return BenchmarkResult(
        name=f"{config.dataset}{quant_suffix}",
        config=asdict(config),
        build_vec_per_s=build_vec_per_s,
        single_qps=round(single_qps),
        batch_qps=round(batch_qps),
        single_latency_ms=round(single_latency_ms, 3),
        batch_latency_ms=round(batch_latency_ms, 3),
        speedup=round(batch_qps / single_qps, 1),
        recall_at_10=round(recall_at_10, 4),
    )


def run_all_benchmarks(
    quick: bool = False, quantization: Optional[str] = None, all_modes: bool = False
) -> list[BenchmarkResult]:
    """Run benchmarks on SIFT-10K (real embeddings).

    Args:
        quick: Use fewer iterations for faster runs
        quantization: Specific mode to test (None or "sq8")
        all_modes: Run all quantization modes (fp32, SQ8)
    """
    vectors, queries, ground_truth = load_sift_10k()
    n_vectors, dimensions = vectors.shape
    n_queries = queries.shape[0]

    # Determine which quantization modes to run
    if all_modes:
        quant_modes = [None, "sq8"]
    else:
        quant_modes = [quantization]

    results = []
    for qmode in quant_modes:
        config = BenchmarkConfig(
            n_vectors=n_vectors,
            n_queries=n_queries,
            dimensions=dimensions,
            k=10,
            quantization=qmode,
            dataset="sift-10k",
        )
        mode_str = qmode or "fp32"
        print(f"Running SIFT-10K ({mode_str})...", file=sys.stderr)
        result = run_benchmark(config, vectors, queries, ground_truth, quick=quick)
        print(
            f"  Build: {result.build_vec_per_s:,} vec/s | "
            f"Search: {result.single_qps:,} / {result.batch_qps:,} QPS | "
            f"Recall@10: {result.recall_at_10:.1%}",
            file=sys.stderr,
        )
        results.append(result)

    return results


def save_run(
    results: list[BenchmarkResult], history_file: Path, notes: str = ""
) -> dict:
    """Save benchmark run to JSONL file."""
    results_dict = {}
    for r in results:
        entry = {
            "s": r.single_qps,
            "b": r.batch_qps,
            "s_ms": r.single_latency_ms,
            "b_ms": r.batch_latency_ms,
            "build": r.build_vec_per_s,
            "r": r.recall_at_10,
        }
        results_dict[r.name] = entry

    run = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "sys": get_system_info(),
        "git": get_git_info(),
        "ver": get_version_info(),
        "results": results_dict,
    }
    if notes:
        run["notes"] = notes

    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a") as f:
        f.write(json.dumps(run) + "\n")

    return run


def load_history(history_file: Path, limit: int = None) -> list[dict]:
    """Load benchmark history from JSONL file."""
    if not history_file.exists():
        return []

    runs = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    if limit:
        runs = runs[-limit:]
    return runs


def print_summary(run: dict):
    """Print a summary of a benchmark run."""
    dirty = " [dirty]" if run["git"]["dirty"] else ""

    print(f"\n{'=' * 70}")
    print("OmenDB Benchmark Results (SIFT-10K, 128D, M=16, ef_c=100, k=10)")
    print(f"{'=' * 70}")
    print(f"Time:   {run['ts']}")
    print(f"System: {run['sys']['cpu']} ({run['sys']['cores']} cores)")
    print(f"Git:    {run['git']['commit']} ({run['git']['branch']}){dirty}")
    print()

    print("| Mode       | Build    | Single QPS | Batch QPS | Speedup | Recall |")
    print("|------------|----------|------------|-----------|---------|--------|")
    for name, r in run["results"].items():
        speedup = r["b"] / r["s"]
        recall = f"{r['r']:.1%}" if "r" in r else "-"
        build = f"{r['build']:,}" if "build" in r else "-"
        print(
            f"| {name:10} | {build:>8} | {r['s']:>10,} | {r['b']:>9,} | {speedup:>6.1f}x | {recall:>6} |"
        )
    print()


def show_history(history_file: Path, limit: int = 10):
    """Show recent benchmark history."""
    runs = load_history(history_file, limit)
    if not runs:
        print("No benchmark history found.")
        return

    print(f"\n{'=' * 75}")
    print("Recent Benchmarks (SIFT-10K)")
    print(f"{'=' * 75}")
    print(
        f"| {'Date':10} | {'Commit':7} | {'Host':8} | {'Build':>8} | {'QPS':>8} | {'Batch':>8} | {'Recall':>7} |"
    )
    print(
        f"|{'-' * 12}|{'-' * 9}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 9}|"
    )

    for run in runs:
        date = run["ts"][:10]
        commit = run["git"]["commit"]
        host = run["sys"]["host"][:8]

        # Show the fp32 result (or first result)
        r = None
        for key in ["sift-10k", "128D"]:
            if key in run["results"]:
                r = run["results"][key]
                break
        if r is None:
            r = next(iter(run["results"].values()))

        build = f"{r['build']:,}" if "build" in r else "-"
        recall = f"{r['r']:.1%}" if "r" in r else "-"
        print(
            f"| {date} | {commit:7} | {host:8} | {build:>8} | {r['s']:>8,} | {r['b']:>8,} | {recall:>7} |"
        )
    print()


def compare_runs(run1: dict, run2: dict):
    """Compare two benchmark runs."""
    print(f"\nComparing: {run1['git']['commit']} -> {run2['git']['commit']}")
    print(f"  Before: {run1['ts']} ({run1['sys']['host']})")
    print(f"  After:  {run2['ts']} ({run2['sys']['host']})")
    print()
    print("| Mode       | Metric | Before | After  | Change |")
    print("|------------|--------|--------|--------|--------|")

    for key in run2["results"]:
        if key in run1["results"]:
            r1, r2 = run1["results"][key], run2["results"][key]
            for metric, mkey in [("Search", "s"), ("Batch", "b"), ("Build", "build")]:
                if mkey in r1 and mkey in r2:
                    v1, v2 = r1[mkey], r2[mkey]
                    change = ((v2 / v1) - 1) * 100
                    sign = "+" if change >= 0 else ""
                    print(
                        f"| {key:10} | {metric:6} | {v1:>6,} | {v2:>6,} | {sign}{change:>5.1f}% |"
                    )
            if "r" in r1 and "r" in r2:
                print(
                    f"| {key:10} | {'Recall':6} | {r1['r']:.1%} | {r2['r']:.1%} |        |"
                )
    print()


def main():
    parser = argparse.ArgumentParser(description="OmenDB Benchmark Runner (SIFT-10K)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (~10s)")
    parser.add_argument("--output", "-o", type=str, help="Save results to file (JSONL)")
    parser.add_argument("--notes", type=str, default="", help="Notes to include")
    parser.add_argument("--history", action="store_true", help="Show history")
    parser.add_argument("--compare", action="store_true", help="Compare last 2 runs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--quantization",
        "-q",
        choices=["sq8"],
        help="Test specific quantization mode",
    )
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run all quantization modes (fp32, SQ8)",
    )
    args = parser.parse_args()

    history_file = Path(args.output) if args.output else DEFAULT_HISTORY_FILE

    if args.history:
        show_history(history_file)
        return

    if args.compare:
        runs = load_history(history_file, 2)
        if len(runs) < 2:
            print("Need at least 2 runs to compare")
            return
        compare_runs(runs[0], runs[1])
        return

    # Run benchmarks on SIFT-10K
    results = run_all_benchmarks(
        quick=args.quick,
        quantization=args.quantization,
        all_modes=args.all_modes,
    )

    # Build run record
    results_dict = {}
    for r in results:
        entry = {
            "s": r.single_qps,
            "b": r.batch_qps,
            "s_ms": r.single_latency_ms,
            "b_ms": r.batch_latency_ms,
            "build": r.build_vec_per_s,
            "r": r.recall_at_10,
        }
        results_dict[r.name] = entry

    run = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "sys": get_system_info(),
        "git": get_git_info(),
        "ver": get_version_info(),
        "results": results_dict,
    }
    if args.notes:
        run["notes"] = args.notes

    if args.output:
        save_run(results, history_file, notes=args.notes)
        print(f"\nSaved to: {history_file}")

    if args.json:
        print(json.dumps(run, indent=2))
    else:
        print_summary(run)


if __name__ == "__main__":
    main()
