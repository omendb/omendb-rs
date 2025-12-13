# OmenDB Release Checklist

Run before any release. Validates versions, tests, docs, and performance.

## Instructions

Execute ALL of these checks in order. Stop and fix any failures before proceeding.

### 1. Version Check (9 locations)

```bash
# Run the sync script to verify ALL 9 locations match
./scripts/sync-version.sh --check

# Compare to published
PYPI=$(curl -sf https://pypi.org/pypi/omendb/json | jq -r '.info.version')
echo "PyPI: $PYPI, Code: $(cat VERSION)"
```

**VERIFY**: All 9 locations match AND version is higher than PyPI.

### 2. CI Runner Check

```bash
# Check for retired runners in workflows
grep -r "macos-1[0-3]" .github/workflows/ && echo "ERROR: Retired macOS runner!" || echo "OK: No retired runners"
grep -r "ubuntu-[12][0-9]" .github/workflows/ | grep -v "ubuntu-22\|ubuntu-24" && echo "WARNING: Check Ubuntu version" || echo "OK"
```

### 3. Test Suite

```bash
cargo test --lib
cd python && uv run pytest tests/ -x -q
```

### 4. Doc Examples Test

```bash
# Test that hybrid search API works as documented
cd python && uv run python -c "
import omendb
db = omendb.open(':memory:', dimensions=4)
db.set([{'id': 'doc1', 'vector': [0.1, 0.2, 0.3, 0.4], 'text': 'hello world'}])
r = db.search_hybrid([0.1, 0.2, 0.3, 0.4], 'hello', k=1)
assert len(r) > 0, 'Hybrid search failed'
print('Doc examples: OK')
"
```

### 5. README Accuracy Check

```bash
# Verify README API matches actual implementation
echo "=== Checking README accuracy ==="

# Storage format (should show .omen + .wal)
grep -n "\.omen\|\.wal" README.md || echo "WARNING: No .omen/.wal docs"

# Quantization API (should NOT show old bits syntax)
grep -n "quantization=" README.md
echo "VERIFY: Should show True/\"sq8\"/\"rabitq\", NOT numbers like 2/4/8"

# Check for removed APIs
grep -n "\.save()" README.md && echo "ERROR: save() removed, use flush()" || echo "OK: No save()"

# Verify params match docstrings
cd python && uv run python -c "
import omendb
# Check open() params are documented
sig = str(omendb.open.__doc__)
for param in ['alpha', 'rescore', 'oversample', 'quantization']:
    if param in sig:
        print(f'OK: {param} documented')
    else:
        print(f'WARN: {param} missing from docstring')
"
```

### 6. Benchmark Quick Check

```bash
cd python && uv run python -c "
import omendb, numpy as np, time
db = omendb.open(':memory:', dimensions=128)
db.set([{'id': f'd{i}', 'vector': np.random.rand(128).astype(np.float32).tolist()} for i in range(10000)])
q = np.random.rand(128).astype(np.float32).tolist()
for _ in range(100): db.search(q, k=10)
start = time.time()
for _ in range(1000): db.search(q, k=10)
qps = 1000/(time.time()-start)
print(f'QPS: {qps:,.0f} (target: >5000)')
assert qps > 3000, f'Performance regression: {qps} QPS'
"
```

### 7. Git Status

```bash
git status
git log --oneline -3
```

**VERIFY**: Working directory is clean, on correct branch.

### 8. Ready to Release

If all checks pass:

```bash
./scripts/bump-version.sh      # or --minor / --major --force
git diff                       # Review ALL changes
git commit -am "chore: Bump version to X.Y.Z"
git push
gh workflow run Release
```

Then monitor: `gh run watch`

### 9. Post-Release

After CI completes:

```bash
# Create tag and GitHub release
git tag -a vX.Y.Z -m "vX.Y.Z: Brief description"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z" --notes "## Changes\n- ..."

# Verify published
pip install omendb==X.Y.Z --upgrade
python -c "import omendb; print(omendb.__version__)"
```
