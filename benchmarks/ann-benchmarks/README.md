# OmenDB ann-benchmarks Integration

Integration with [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for standardized ANN comparison.

## Quick Start

### Option 1: Run standalone test (recommended first)

Test against real datasets without full ann-benchmarks setup:

```bash
cd omendb/benchmarks
python ann_dataset_test.py --dataset sift-128-euclidean
```

### Option 2: Full ann-benchmarks integration

```bash
# Clone ann-benchmarks
git clone https://github.com/erikbern/ann-benchmarks.git
cd ann-benchmarks
pip install -r requirements.txt

# Copy OmenDB algorithm
cp -r ../omendb/benchmarks/ann-benchmarks/algorithms/omendb ann_benchmarks/algorithms/

# Build Docker image
python install.py --algorithm omendb

# Run benchmark
python run.py --algorithm omendb --dataset sift-128-euclidean --runs 3

# Generate plots
python plot.py --dataset sift-128-euclidean
```

## Datasets

| Dataset                     | Dimensions | Train | Test | Metric |
| --------------------------- | ---------- | ----- | ---- | ------ |
| sift-128-euclidean          | 128        | 1M    | 10K  | L2     |
| glove-25-angular            | 25         | 1.2M  | 10K  | Cosine |
| glove-100-angular           | 100        | 1.2M  | 10K  | Cosine |
| fashion-mnist-784-euclidean | 784        | 60K   | 10K  | L2     |
| gist-960-euclidean          | 960        | 1M    | 1K   | L2     |

Download: `python -c "from ann_benchmarks.datasets import get_dataset; get_dataset('sift-128-euclidean')"`

## Configurations

- `omendb-m-16`: Standard config (M=16, ef_construction=100)
- `omendb-m-24`: Higher quality (M=24, ef_construction=100)
- `omendb-sq8-m-16`: SQ8 quantized with rescore

## Results

Results stored in `results/` directory. View with:

```bash
python plot.py --dataset sift-128-euclidean --output results/sift.png
```
