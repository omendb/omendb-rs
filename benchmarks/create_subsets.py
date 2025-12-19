#!/usr/bin/env python3
"""
Create 10K subset files from ann-benchmarks datasets for CI validation.

This is a one-time script to generate the subset files that are stored in the repo.
Run this when you need to regenerate the subsets.

Output files (~12-15MB total, stored in repo):
  benchmarks/data/sift-10k.npz
  benchmarks/data/glove-10k.npz

Each NPZ contains:
  - vectors: float32[10000, D] - base vectors
  - queries: float32[1000, D] - query vectors
  - ground_truth: int32[1000, 100] - brute-force KNN indices
  - metadata: int32[10000] - synthetic category labels (id % 10)
  - filtered_ground_truth: int32[1000, 100] - GT for category < 5 filter
"""

import urllib.request
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "output": "sift-10k.npz",
        "metric": "l2",
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "output": "glove-10k.npz",
        "metric": "cosine",
    },
}

SUBSET_SIZE = 10_000
NUM_QUERIES = 1000
GT_K = 100


def download_dataset(name: str) -> Path:
    """Download dataset if not already cached."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{name}.hdf5"

    if path.exists():
        print(f"Using cached dataset: {path}")
        return path

    url = DATASETS[name]["url"]
    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved to {path}")
    return path


def compute_ground_truth_l2(
    vectors: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Compute brute-force L2 KNN ground truth."""
    gt = np.zeros((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        distances = np.linalg.norm(vectors - q, axis=1)
        gt[i] = np.argsort(distances)[:k]
    return gt


def compute_ground_truth_cosine(
    vectors: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Compute brute-force cosine KNN ground truth."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    vectors_norm = vectors / norms

    gt = np.zeros((len(queries), k), dtype=np.int32)
    for i, q in enumerate(queries):
        q_norm = q / np.linalg.norm(q) if np.linalg.norm(q) > 0 else q
        # Cosine similarity = dot product of normalized vectors
        # Higher is better, so we negate for argsort
        similarities = vectors_norm @ q_norm
        gt[i] = np.argsort(-similarities)[:k]
    return gt


def create_subset(name: str) -> None:
    """Create a 10K subset from a full dataset."""
    config = DATASETS[name]
    path = download_dataset(name)

    print(f"\nCreating subset for {name}...")

    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])

    print(f"  Full dataset: {train.shape[0]:,} vectors, {train.shape[1]}D")
    print(f"  Query vectors: {test.shape[0]:,}")

    # Take first SUBSET_SIZE vectors and NUM_QUERIES queries
    vectors = train[:SUBSET_SIZE].astype(np.float32)
    queries = test[:NUM_QUERIES].astype(np.float32)

    print(f"  Subset: {len(vectors):,} vectors, {len(queries):,} queries")

    # Compute ground truth for subset
    print(f"  Computing brute-force ground truth (k={GT_K})...")
    if config["metric"] == "l2":
        ground_truth = compute_ground_truth_l2(vectors, queries, GT_K)
    else:
        ground_truth = compute_ground_truth_cosine(vectors, queries, GT_K)

    # Create synthetic metadata for filtered search
    # Categories 0-9 based on index
    metadata = (np.arange(len(vectors)) % 10).astype(np.int32)

    # Compute filtered ground truth (categories 0-4, ~50% selectivity)
    print("  Computing filtered ground truth (category < 5)...")
    mask = metadata < 5
    filtered_indices = np.where(mask)[0]
    filtered_vectors = vectors[mask]

    if config["metric"] == "l2":
        filtered_gt_local = compute_ground_truth_l2(filtered_vectors, queries, GT_K)
    else:
        filtered_gt_local = compute_ground_truth_cosine(filtered_vectors, queries, GT_K)

    # Map local indices back to original indices
    filtered_ground_truth = filtered_indices[filtered_gt_local]

    # Save as NPZ
    output_path = DATA_DIR / config["output"]
    np.savez_compressed(
        output_path,
        vectors=vectors,
        queries=queries,
        ground_truth=ground_truth,
        metadata=metadata,
        filtered_ground_truth=filtered_ground_truth,
        metric=config["metric"],
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f}MB)")


def main():
    print("Creating 10K subset files for CI validation")
    print("=" * 60)

    for name in DATASETS:
        create_subset(name)

    print("\n" + "=" * 60)
    print("Done! Subset files created in benchmarks/data/")
    print("These files should be committed to the repo for CI.")


if __name__ == "__main__":
    main()
