#!/usr/bin/env python3
"""Rust API SIFT publish benchmark.

This wrapper reuses the existing SIFT .npz datasets, exports raw arrays for the
Rust binary, runs the Rust API benchmark, and optionally appends the median
result to the shared benchmark history.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "benchmarks" / "data"
DATASETS = {
    10_000: "sift-10k.npz",
    100_000: "sift-100k.npz",
    1_000_000: "sift-1m.npz",
}


def metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    version = "unknown"
    cargo_toml = ROOT / "Cargo.toml"
    for line in cargo_toml.read_text().splitlines():
        if line.startswith("version = "):
            version = line.split("=", maxsplit=1)[1].strip().strip('"')
            break

    return {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "omendb_version": version,
        "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "api": "rust",
    }


def load_sift(n_vectors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filename = DATASETS.get(n_vectors)
    if filename is None:
        raise SystemExit(f"unsupported vector count: {n_vectors}")
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing dataset: {path}")

    data = np.load(path)
    return data["vectors"], data["queries"], data["ground_truth"]


def run_once(args: argparse.Namespace, tmpdir: Path) -> dict:
    vectors, queries, ground_truth = load_sift(args.vectors)
    queries = queries[: args.queries]
    ground_truth = ground_truth[: args.queries, : args.k]

    vectors_path = tmpdir / "vectors.f32"
    queries_path = tmpdir / "queries.f32"
    ground_truth_path = tmpdir / "ground_truth.i32"

    np.asarray(vectors, dtype="<f4").tofile(vectors_path)
    np.asarray(queries, dtype="<f4").tofile(queries_path)
    np.asarray(ground_truth, dtype="<i4").tofile(ground_truth_path)

    command = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "bench_sift_publish",
        "--",
        "--vectors-file",
        str(vectors_path),
        "--queries-file",
        str(queries_path),
        "--ground-truth-file",
        str(ground_truth_path),
        "--vectors",
        str(args.vectors),
        "--queries",
        str(len(queries)),
        "--dimensions",
        str(vectors.shape[1]),
        "--k",
        str(args.k),
        "--warmup",
        str(args.warmup),
    ]
    env = os.environ.copy()
    env.setdefault("RUSTC_BOOTSTRAP", "1")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def median_result(results: list[dict]) -> dict:
    result = json.loads(json.dumps(results[0]))
    paths = [
        ("build", "vec_per_s"),
        ("search", "qps"),
        ("search", "latency_avg_ms"),
        ("search", "latency_p50_ms"),
        ("search", "latency_p99_ms"),
        ("recall", "recall_at_k"),
        ("filtered", "qps"),
        ("filtered", "latency_ms"),
        ("batch", "qps"),
        ("batch", "latency_ms"),
    ]
    for section, key in paths:
        result[section][key] = median(r[section][key] for r in results)
    return result


def append_history(path: Path, metadata_entry: dict, result: dict) -> None:
    if path.exists():
        history = json.loads(path.read_text())
    else:
        history = []

    history.append({"metadata": metadata_entry, "results": [result]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history[-100:], indent=2) + "\n")
    print(f"History updated: {path} ({len(history[-100:])} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--append", type=Path)
    args = parser.parse_args()

    if args.publish:
        args.runs = max(args.runs, 5)

    print(f"Running Rust SIFT benchmark: {args.runs} run(s)")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i in range(args.runs):
            print(f"  Run {i + 1}/{args.runs}...", flush=True)
            result = run_once(args, tmpdir)
            results.append(result)

    result = median_result(results) if len(results) > 1 else results[0]
    meta = metadata()
    if args.publish:
        meta["runs"] = args.runs
        meta["methodology"] = "median"

    print(json.dumps({"metadata": meta, "results": [result]}, indent=2))
    if args.append:
        append_history(args.append, meta, result)


if __name__ == "__main__":
    main()
