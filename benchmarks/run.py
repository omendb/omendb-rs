#!/usr/bin/env python3
"""
OmenDB Benchmark Runner

Runs benchmarks with QPS and recall measurement.

Usage:
    python benchmarks/run.py                        # Full benchmark (~45s)
    python benchmarks/run.py --quick                # Quick run (~10s)
    python benchmarks/run.py --output FILE          # Save results to FILE
    python benchmarks/run.py --history              # Show history
    python benchmarks/run.py --compare              # Compare last 2 runs
    python benchmarks/run.py --notes "text"         # Add notes to run

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


@dataclass
class BenchmarkConfig:
    n_vectors: int
    n_queries: int
    dimensions: int
    k: int
    ef: Optional[int] = None
    m: int = 16
    ef_construction: int = 200


@dataclass
class BenchmarkResult:
    name: str
    config: dict
    single_qps: float
    batch_qps: float
    single_latency_ms: float
    batch_latency_ms: float
    speedup: float
    recall_at_10: Optional[float] = None


def brute_force_knn(query: np.ndarray, vectors: np.ndarray, k: int) -> list[int]:
    """Compute ground truth k-NN using brute force L2 distance."""
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k].tolist()


def compute_recall(result_ids: list[str], ground_truth: list[int]) -> float:
    """Compute recall between HNSW results and brute force ground truth."""
    hnsw_indices = {int(id[1:]) for id in result_ids}  # d0 -> 0
    return len(hnsw_indices & set(ground_truth)) / len(ground_truth)


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
        "host": platform.node().split(".")[0],  # Short hostname
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


def generate_vectors(n: int, dim: int) -> np.ndarray:
    """Generate random vectors for benchmarking."""
    return np.random.randn(n, dim).astype(np.float32)


def run_benchmark(
    config: BenchmarkConfig, quick: bool = False, measure_recall: bool = False
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    np.random.seed(42)  # Reproducibility

    vectors = generate_vectors(config.n_vectors, config.dimensions)
    queries = generate_vectors(config.n_queries, config.dimensions)

    # Pre-compute ground truth if measuring recall
    ground_truth = None
    if measure_recall:
        ground_truth = [brute_force_knn(q, vectors, config.k) for q in queries]

    with tempfile.TemporaryDirectory() as tmpdir:
        db = omendb.open(f"{tmpdir}/bench", dimensions=config.dimensions)

        # Insert vectors
        items = [{"id": f"d{i}", "vector": v.tolist()} for i, v in enumerate(vectors)]
        db.set(items)

        # Measure recall if requested
        recall_at_10 = None
        if measure_recall and ground_truth:
            recall_sum = 0.0
            for i, q in enumerate(queries):
                results = db.search(q.tolist(), k=config.k)
                result_ids = [r["id"] for r in results]
                recall_sum += compute_recall(result_ids, ground_truth[i])
            recall_at_10 = recall_sum / len(queries)

        # Warmup
        for q in queries[:5]:
            db.search(q.tolist(), k=config.k)
        db.search_batch([q.tolist() for q in queries[:5]], k=config.k)

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

    return BenchmarkResult(
        name=f"{config.dimensions}D",
        config=asdict(config),
        single_qps=round(single_qps),
        batch_qps=round(batch_qps),
        single_latency_ms=round(single_latency_ms, 3),
        batch_latency_ms=round(batch_latency_ms, 3),
        speedup=round(batch_qps / single_qps, 1),
        recall_at_10=round(recall_at_10, 4) if recall_at_10 else None,
    )


def run_all_benchmarks(quick: bool = False) -> list[BenchmarkResult]:
    """Run the standard benchmark suite with recall measurement."""
    configs = [
        BenchmarkConfig(n_vectors=10_000, n_queries=100, dimensions=128, k=10),
        BenchmarkConfig(n_vectors=10_000, n_queries=100, dimensions=768, k=10),
        BenchmarkConfig(n_vectors=10_000, n_queries=100, dimensions=1536, k=10),
    ]

    results = []
    for config in configs:
        print(f"Running {config.dimensions}D...", file=sys.stderr)
        result = run_benchmark(config, quick=quick, measure_recall=True)
        print(
            f"  {result.single_qps:,} / {result.batch_qps:,} QPS, "
            f"recall@10: {result.recall_at_10:.1%}",
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
        entry = {"s": r.single_qps, "b": r.batch_qps}
        if r.recall_at_10 is not None:
            entry["r"] = r.recall_at_10
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
    has_recall = any("r" in r for r in run["results"].values())

    print(f"\n{'=' * 65}")
    print("OmenDB Benchmark Results")
    print(f"{'=' * 65}")
    print(f"Time:   {run['ts']}")
    print(f"System: {run['sys']['cpu']} ({run['sys']['cores']} cores)")
    print(f"Git:    {run['git']['commit']} ({run['git']['branch']}){dirty}")
    print()

    if has_recall:
        print("| Dim   | Single QPS | Batch QPS | Speedup | Recall |")
        print("|-------|------------|-----------|---------|--------|")
        for name, r in run["results"].items():
            speedup = r["b"] / r["s"]
            recall = f"{r['r']:.1%}" if "r" in r else "-"
            print(
                f"| {name:5} | {r['s']:>10,} | {r['b']:>9,} | {speedup:>6.1f}x | {recall:>6} |"
            )
    else:
        print("| Dim   | Single QPS | Batch QPS | Speedup |")
        print("|-------|------------|-----------|---------|")
        for name, r in run["results"].items():
            speedup = r["b"] / r["s"]
            print(f"| {name:5} | {r['s']:>10,} | {r['b']:>9,} | {speedup:>6.1f}x |")
    print()


def show_history(history_file: Path, limit: int = 10):
    """Show recent benchmark history."""
    runs = load_history(history_file, limit)
    if not runs:
        print("No benchmark history found.")
        return

    print(f"\n{'=' * 75}")
    print("Recent Benchmarks (Single / Batch QPS)")
    print(f"{'=' * 75}")
    print(
        f"| {'Date':10} | {'Commit':7} | {'Host':8} | {'128D':>13} | {'768D':>13} | {'1536D':>13} |"
    )
    print(f"|{'-' * 12}|{'-' * 9}|{'-' * 10}|{'-' * 15}|{'-' * 15}|{'-' * 15}|")

    for run in runs:
        date = run["ts"][:10]
        commit = run["git"]["commit"]
        host = run["sys"]["host"][:8]

        dims = []
        for d in ["128D", "768D", "1536D"]:
            if d in run["results"]:
                r = run["results"][d]
                dims.append(f"{r['s']:>5,}/{r['b']:>6,}")
            else:
                dims.append("-")

        print(f"| {date} | {commit:7} | {host:8} | {dims[0]} | {dims[1]} | {dims[2]} |")
    print()


def compare_runs(run1: dict, run2: dict):
    """Compare two benchmark runs."""
    print(f"\nComparing: {run1['git']['commit']} → {run2['git']['commit']}")
    print(f"  Before: {run1['ts']} ({run1['sys']['host']})")
    print(f"  After:  {run2['ts']} ({run2['sys']['host']})")
    print()
    print("| Dim   | Metric | Before | After  | Change |")
    print("|-------|--------|--------|--------|--------|")

    for dim in ["128D", "768D", "1536D"]:
        if dim in run1["results"] and dim in run2["results"]:
            r1, r2 = run1["results"][dim], run2["results"][dim]
            for metric, key in [("Single", "s"), ("Batch", "b")]:
                v1, v2 = r1[key], r2[key]
                change = ((v2 / v1) - 1) * 100
                sign = "+" if change >= 0 else ""
                print(
                    f"| {dim:5} | {metric:6} | {v1:>6,} | {v2:>6,} | {sign}{change:>5.1f}% |"
                )
    print()


def main():
    parser = argparse.ArgumentParser(description="OmenDB Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Quick mode (~10s)")
    parser.add_argument("--output", "-o", type=str, help="Save results to file (JSONL)")
    parser.add_argument("--notes", type=str, default="", help="Notes to include")
    parser.add_argument("--history", action="store_true", help="Show history")
    parser.add_argument("--compare", action="store_true", help="Compare last 2 runs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
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

    # Run benchmarks (always includes recall)
    results = run_all_benchmarks(quick=args.quick)

    # Build run record
    results_dict = {}
    for r in results:
        entry = {"s": r.single_qps, "b": r.batch_qps}
        if r.recall_at_10 is not None:
            entry["r"] = r.recall_at_10
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
