# Releasing OmenDB

## Quick Release

```bash
# 1. Add a ## vX.Y.Z section to CHANGELOG.md
# 2. Update VERSION file to new version
echo "0.0.27" > VERSION

# 3. Sync all version locations
./scripts/sync-version.sh

# 4. Preview GitHub release notes
./scripts/release-notes.sh 0.0.27 --body-only

# 5. Review and commit
git diff
git add -A && git commit -m "chore: Bump to 0.0.27"
git push

# 6. Trigger release (GitHub UI or CLI)
gh workflow run release.yml
```

## Pre-Release Checklist

- [ ] Rust tests pass, 441+ (`cargo test --lib`)
- [ ] Clippy clean (`cargo clippy --lib -- -D warnings`)
- [ ] Python tests pass, 313+ (`cd python && uv run pytest tests/ -x`)
- [ ] Node tests pass, 104+ (`cd node && bun test`)
- [ ] `CHANGELOG.md` has a `## vX.Y.Z` section for the release
- [ ] Version synced across all locations

## Version Locations

All locations must match the VERSION file:

| #   | File                        | What                               |
| --- | --------------------------- | ---------------------------------- |
| 1   | `VERSION`                   | Source of truth                    |
| 2   | `Cargo.toml`                | Main Rust crate                    |
| 3   | `python/Cargo.toml`         | Python bindings                    |
| 4   | `python/omendb/__init__.py` | Python package version             |
| 5   | `omendb-ffi/Cargo.toml`     | C FFI bindings                     |
| 6   | `node/Cargo.toml`           | Node bindings                      |
| 7   | `node/package.json`         | npm package version + optionalDeps |
| 8   | `node/wrapper/package.json` | npm wrapper version + @omendb dep  |

## Scripts

```bash
# Sync all versions to VERSION file
./scripts/sync-version.sh

# Bump to specific version
./scripts/sync-version.sh 0.0.27

# Check versions match (CI uses this)
./scripts/sync-version.sh --check
```

## Release Workflow

The GitHub Actions release workflow (`release.yml`):

1. **Verify** - Checks all version locations match VERSION
2. **Validate release notes** - Extracts the current version section from `CHANGELOG.md`
3. **Check not published** - Ensures version isn't already on PyPI/crates.io
4. **Lint & Test** - fmt, clippy, cargo test
5. **Build** - Python wheels (Linux/macOS), Node binaries
6. **Publish** - crates.io → PyPI → npm (sequential)
7. **Tag + release** - Creates the git tag and GitHub release from `CHANGELOG.md`

### Dry Run

Test the build without publishing:

```bash
gh workflow run release.yml -f dry-run=true
```

Preview the exact GitHub release body locally:

```bash
./scripts/release-notes.sh 0.0.27 --body-only
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
./scripts/sync-version.sh 0.0.28
git add -A && git commit -m "chore: Bump to 0.0.28"
git push
```

### Build fails

1. Check CI logs for specific error
2. Fix locally and push
3. Re-trigger release workflow

## Versioning

- Use `0.0.x` until API stabilizes
- Bump patch for any release (breaking changes OK in 0.x)
- Sequential only: 0.0.26 → 0.0.27 (not 0.0.26 → 0.1.0)
