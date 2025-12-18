# OmenDB ann-benchmarks Integration

Integration with [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for standardized ANN comparison.

## Quick Start

```bash
# Clone ann-benchmarks
git clone https://github.com/erikbern/ann-benchmarks.git
cd ann-benchmarks
pip install -r requirements.txt

# Copy OmenDB algorithm
cp -r /path/to/omendb/benchmarks/ann-benchmarks/algorithms/omendb ann_benchmarks/algorithms/

# Build Docker image
python install.py --algorithm omendb

# Run benchmark
python run.py --algorithm omendb --dataset sift-128-euclidean --runs 3

# Generate plots
python plot.py --dataset sift-128-euclidean
```

## Standalone Test

Test against real datasets without full ann-benchmarks setup:

```bash
cd omendb/benchmarks
pip install h5py
python ann_dataset_test.py --dataset fashion-mnist-784-euclidean
```

## Supported Datasets

| Dataset                     | Dimensions | Vectors | Metric | Status    |
| --------------------------- | ---------- | ------- | ------ | --------- |
| sift-128-euclidean          | 128        | 1M      | L2     | Supported |
| fashion-mnist-784-euclidean | 784        | 60K     | L2     | Supported |
| glove-25-angular            | 25         | 1.2M    | Cosine | Supported |
| glove-100-angular           | 100        | 1.2M    | Cosine | Supported |
| gist-960-euclidean          | 960        | 1M      | L2     | Supported |

Note: Angular datasets map to OmenDB's cosine metric. Results may vary due to implementation differences in how angular distance is computed.

## Configurations

| Config            | Description                                |
| ----------------- | ------------------------------------------ |
| `omendb-m-16`     | Standard (M=16, ef_construction=100)       |
| `omendb-m-24`     | Higher quality (M=24, ef_construction=100) |
| `omendb-sq8-m-16` | SQ8 quantized with rescore                 |

## Results

Fashion-MNIST (60K vectors, 784D, M3 Max):

| ef  | QPS   | Recall@10 |
| --- | ----- | --------- |
| 10  | 7,192 | 97.4%     |
| 20  | 8,905 | 99.3%     |
| 100 | 3,275 | 99.9%     |
| 200 | 2,089 | 100.0%    |

## Files

```
algorithms/omendb/
├── module.py      # BaseANN implementation
├── config.yml     # Parameter combinations
└── Dockerfile     # Build environment
```
