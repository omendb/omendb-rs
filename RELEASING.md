# Releasing OmenDB

## Quick Release

```bash
# 1. Update CHANGELOG.md with release notes
# 2. Update VERSION file to new version
echo "0.0.19" > VERSION

# 3. Sync all version locations
./scripts/sync-version.sh

# 4. Review and commit
git diff
git add -A && git commit -m "chore: Bump to 0.0.19"
git push

# 5. Trigger release (GitHub UI or CLI)
gh workflow run release.yml

# 6. After successful release, tag the commit
git tag v0.0.19
git push origin v0.0.19
```

## Pre-Release Checklist

- [ ] All tests pass (`cargo test --lib`)
- [ ] Clippy clean (`cargo clippy --lib -- -D warnings`)
- [ ] CHANGELOG.md updated with release notes
- [ ] Version synced across all locations

## Version Locations

All 8 locations must match the VERSION file:

| #   | File                        | What                               |
| --- | --------------------------- | ---------------------------------- |
| 1   | `VERSION`                   | Source of truth                    |
| 2   | `Cargo.toml`                | Main Rust crate                    |
| 3   | `python/Cargo.toml`         | Python bindings                    |
| 4   | `python/omendb/__init__.py` | Python package version             |
| 5   | `src/ffi.rs`                | FFI version constant               |
| 6   | `node/Cargo.toml`           | Node bindings                      |
| 7   | `node/package.json`         | npm package version + optionalDeps |
| 8   | `node/wrapper/package.json` | npm wrapper version + @omendb dep  |

## Scripts

```bash
# Sync all versions to VERSION file
./scripts/sync-version.sh

# Bump to specific version
./scripts/sync-version.sh 0.0.10

# Check versions match (CI uses this)
./scripts/sync-version.sh --check
```

## Release Workflow

The GitHub Actions release workflow (`release.yml`):

1. **Verify** - Checks all 9 version locations match VERSION
2. **Check not published** - Ensures version isn't already on PyPI/crates.io
3. **Lint & Test** - fmt, clippy, cargo test
4. **Build** - Python wheels (Linux/macOS), Node binaries
5. **Publish** - crates.io → PyPI → npm (sequential)

### Dry Run

Test the build without publishing:

```bash
gh workflow run release.yml -f dry-run=true
```

## Troubleshooting

### Version mismatch error

```bash
./scripts/sync-version.sh
git add -A && git commit --amend --no-edit
git push --force-with-lease
```

### Already published error

Version already exists on registry. Bump to next version:

```bash
./scripts/sync-version.sh 0.0.11
git add -A && git commit -m "chore: Bump to 0.0.11"
git push
```

### Build fails

1. Check CI logs for specific error
2. Fix locally and push
3. Re-trigger release workflow

## Versioning

- Use `0.0.x` until API stabilizes
- Bump patch for any release (breaking changes OK in 0.x)
- Sequential only: 0.0.8 → 0.0.9 (not 0.0.8 → 0.1.0)
