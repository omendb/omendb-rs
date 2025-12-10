# OmenDB Benchmarks

Structured benchmark infrastructure with JSONL history.

## Quick Start

```bash
cd python  # Has numpy in venv

# Run benchmarks
uv run python ../benchmarks/run.py

# Quick run (fewer iterations)
uv run python ../benchmarks/run.py --quick

# Show history
uv run python ../benchmarks/run.py --history

# Compare last 2 runs
uv run python ../benchmarks/run.py --compare

# Add notes
uv run python ../benchmarks/run.py --notes "after optimization X"
```

## Output

Each run appends one line to `history.jsonl`:

```json
{"ts": "2025-12-10 06:42:55", "sys": {"cpu": "Apple M3 Max", "cores": 16, ...}, "git": {"commit": "bd053bd", "branch": "main", "dirty": true}, "results": {"128D": {"s": 8170, "b": 78652}, ...}}
```

## Agent-Friendly

```bash
# Raw data
cat benchmarks/history.jsonl

# Filter by commit
grep "bd053bd" benchmarks/history.jsonl | jq

# Get latest single QPS for 128D
tail -1 benchmarks/history.jsonl | jq '.results["128D"].s'
```

## History View

```
===========================================================================
Recent Benchmarks (Single / Batch QPS)
===========================================================================
| Date       | Commit  | Host     |          128D |          768D |         1536D |
|------------|---------|----------|---------------|---------------|---------------|
| 2025-12-10 | bd053bd | Apple    | 8,170/78,652 | 2,511/12,935 | 1,167/ 4,472 |
```
