# Release Process

## Quick Release (Patch)

```bash
./scripts/bump-version.sh      # Auto-bumps patch version
git diff                       # Review changes
git commit -am "chore: Bump version to X.Y.Z"
git push
gh workflow run Release
```

## Version Bumping

```bash
./scripts/bump-version.sh             # Patch: 0.0.5 -> 0.0.6 (default)
./scripts/bump-version.sh --minor     # Minor: 0.0.5 -> 0.1.0
./scripts/bump-version.sh --major --force  # Major: 0.0.5 -> 1.0.0
```

The script automatically:

- Fetches current published versions from PyPI/crates.io
- Validates sequential versioning
- Updates all version files in sync

## Pre-Release Checklist

### 1. Local Testing

```bash
# Rust
cargo test --lib
cargo clippy -- -D warnings

# Python
cd python
uv run maturin develop --release
uv run pytest tests/ -v
uv run ruff check .

# Quick benchmark (10K vectors, 128D)
uv run python -c "
import omendb, numpy as np, time
db = omendb.open(':memory:', dimensions=128)
db.set([{'id': f'd{i}', 'vector': np.random.rand(128).tolist()} for i in range(10000)])
q = np.random.rand(128).tolist()
for _ in range(100): db.search(q, k=10)  # warmup
start = time.time()
for _ in range(1000): db.search(q, k=10)
print(f'Single: {1000/(time.time()-start):,.0f} QPS (target: >5000)')
"
```

### 2. Doc Review

```bash
# Check API examples match actual implementation
grep -r "set_with_text\|enable_text_search" ../cloud/ai/  # Should be minimal
grep -r "hybrid_search\|text_search" ../cloud/ai/         # Verify correct method names

# Verify README examples work
python -c "
import omendb
db = omendb.open(':memory:', dimensions=4)
db.set([{'id': 'doc1', 'vector': [0.1, 0.2, 0.3, 0.4], 'text': 'hello world'}])
r = db.search_hybrid([0.1, 0.2, 0.3, 0.4], 'hello', k=1)
print('Hybrid search works:', len(r) > 0)
"
```

### 3. Version Validation

```bash
# Check published versions
curl -s https://pypi.org/pypi/omendb/json | jq -r '.releases | keys | sort | .[-1]'
curl -s https://crates.io/api/v1/crates/omendb | jq -r '.versions[0].num'

# Check code versions match
grep '^version' Cargo.toml python/Cargo.toml
jq .version node/package.json
```

### 3. CI Status

```bash
gh run list --limit 3  # All should be passing
```

## Version Files (6 locations)

| File                        | Registry  | Notes                |
| --------------------------- | --------- | -------------------- |
| `Cargo.toml`                | crates.io | Main Rust crate      |
| `python/Cargo.toml`         | PyPI      | Python bindings      |
| `node/Cargo.toml`           | -         | Node bindings (Rust) |
| `node/package.json`         | npm       | @omendb/omendb       |
| `node/wrapper/package.json` | npm       | omendb wrapper + dep |
| `README.md`                 | Docs      | Version badge text   |

The bump script updates all 6 locations and verifies they match.

## CI Safeguards

The release workflow automatically validates:

- All version files match
- Version is higher than published
- Version isn't already published

## Troubleshooting

**Release cancelled/failed?**

```bash
gh run view <run-id> --log | tail -100
```

**Version mismatch error?**

```bash
./scripts/bump-version.sh  # Re-run to fix
```

**Already published error?**

- Check actual published versions (see above)
- Ensure code version is `published + 1`
