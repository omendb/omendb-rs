#!/bin/bash
# Sync version from VERSION file to all package files
#
# Usage:
#   ./scripts/sync-version.sh              # Sync all files to VERSION
#   ./scripts/sync-version.sh 0.0.10       # Bump to specific version
#   ./scripts/sync-version.sh --check      # Verify all versions match (no changes)
#
# Version Locations (8 files):
#   1. VERSION                    - Source of truth
#   2. Cargo.toml                 - Main Rust crate (version)
#   3. python/Cargo.toml          - Python bindings crate
#   4. python/omendb/__init__.py  - Python __version__
#   5. omendb-ffi/Cargo.toml      - C FFI crate
#   6. node/Cargo.toml            - Node bindings crate
#   7. node/package.json          - npm @omendb/omendb package
#   8. node/wrapper/package.json  - npm omendb wrapper (version + dep)

set -e

cd "$(dirname "$0")/.."

# Handle arguments
CHECK_ONLY=false
NEW_VERSION=""

if [ "$1" = "--check" ]; then
    CHECK_ONLY=true
elif [ -n "$1" ]; then
    NEW_VERSION="$1"
fi

# Get current or new version
if [ -n "$NEW_VERSION" ]; then
    echo "$NEW_VERSION" > VERSION
    VERSION="$NEW_VERSION"
else
    VERSION=$(cat VERSION | tr -d '\n')
fi

echo "Version: $VERSION"
echo ""

ERRORS=0

check_or_update() {
    local file=$1
    local current=$2

    if [ "$current" = "$VERSION" ]; then
        echo "  [OK] $file"
    elif [ "$CHECK_ONLY" = true ]; then
        echo "  [MISMATCH] $file: $current (expected $VERSION)"
        ERRORS=$((ERRORS + 1))
    else
        echo "  [UPDATED] $file: $current -> $VERSION"
    fi
}

if [ "$CHECK_ONLY" = true ]; then
    echo "Checking versions..."
else
    echo "Syncing versions..."
fi
echo ""

# 1. VERSION (source of truth - already handled)
echo "  [OK] VERSION"

# 2. Cargo.toml (main package version)
CARGO_V=$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)
if [ "$CHECK_ONLY" = false ] && [ "$CARGO_V" != "$VERSION" ]; then
    sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" Cargo.toml
fi
check_or_update "Cargo.toml (version)" "$CARGO_V"

# 3. python/Cargo.toml
PYTHON_V=$(grep '^version = ' python/Cargo.toml | head -1 | cut -d'"' -f2)
if [ "$CHECK_ONLY" = false ] && [ "$PYTHON_V" != "$VERSION" ]; then
    sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" python/Cargo.toml
fi
check_or_update "python/Cargo.toml" "$PYTHON_V"

# 5. python/omendb/__init__.py (__version__)
INIT_V=$(grep '__version__' python/omendb/__init__.py | cut -d'"' -f2)
if [ "$CHECK_ONLY" = false ] && [ "$INIT_V" != "$VERSION" ]; then
    sed -i '' "s/__version__ = \"[^\"]*\"/__version__ = \"$VERSION\"/" python/omendb/__init__.py
fi
check_or_update "python/omendb/__init__.py" "$INIT_V"

# 6. omendb-ffi/Cargo.toml (FFI crate)
FFI_V=$(grep '^version = ' omendb-ffi/Cargo.toml | head -1 | cut -d'"' -f2)
if [ "$CHECK_ONLY" = false ] && [ "$FFI_V" != "$VERSION" ]; then
    sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" omendb-ffi/Cargo.toml
    # Also update the omendb dependency version
    sed -i '' "s/omendb = { version = \"[^\"]*\"/omendb = { version = \"$VERSION\"/" omendb-ffi/Cargo.toml
fi
check_or_update "omendb-ffi/Cargo.toml" "$FFI_V"

# 7. node/Cargo.toml
NODE_CARGO_V=$(grep '^version = ' node/Cargo.toml | head -1 | cut -d'"' -f2)
if [ "$CHECK_ONLY" = false ] && [ "$NODE_CARGO_V" != "$VERSION" ]; then
    sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" node/Cargo.toml
fi
check_or_update "node/Cargo.toml" "$NODE_CARGO_V"

# 8. node/package.json (version + optionalDependencies)
NODE_V=$(jq -r .version node/package.json)
NODE_OPT=$(jq -r '.optionalDependencies["@omendb/omendb-darwin-arm64"]' node/package.json)
if [ "$CHECK_ONLY" = false ] && ([ "$NODE_V" != "$VERSION" ] || [ "$NODE_OPT" != "$VERSION" ]); then
    jq ".version = \"$VERSION\" |
        .optionalDependencies[\"@omendb/omendb-darwin-x64\"] = \"$VERSION\" |
        .optionalDependencies[\"@omendb/omendb-darwin-arm64\"] = \"$VERSION\" |
        .optionalDependencies[\"@omendb/omendb-linux-x64-gnu\"] = \"$VERSION\" |
        .optionalDependencies[\"@omendb/omendb-linux-arm64-gnu\"] = \"$VERSION\" |
        .optionalDependencies[\"@omendb/omendb-win32-x64-msvc\"] = \"$VERSION\"" \
        node/package.json > tmp.json && mv tmp.json node/package.json
fi
check_or_update "node/package.json (version)" "$NODE_V"
check_or_update "node/package.json (optionalDeps)" "$NODE_OPT"

# 9. node/wrapper/package.json (version + @omendb dep)
WRAPPER_V=$(jq -r .version node/wrapper/package.json)
WRAPPER_DEP=$(jq -r '.dependencies["@omendb/omendb"]' node/wrapper/package.json)
if [ "$CHECK_ONLY" = false ] && ([ "$WRAPPER_V" != "$VERSION" ] || [ "$WRAPPER_DEP" != "$VERSION" ]); then
    jq ".version = \"$VERSION\" | .dependencies[\"@omendb/omendb\"] = \"$VERSION\"" \
        node/wrapper/package.json > tmp.json && mv tmp.json node/wrapper/package.json
fi
check_or_update "node/wrapper/package.json (version)" "$WRAPPER_V"
check_or_update "node/wrapper/package.json (@omendb)" "$WRAPPER_DEP"

echo ""

# Update lockfiles (not in check mode)
if [ "$CHECK_ONLY" = false ]; then
    echo "Updating lockfiles..."
    cargo check --quiet 2>/dev/null || true
    echo "  [OK] Cargo.lock"
    (cd python && cargo check --quiet 2>/dev/null || true)
    echo "  [OK] python/Cargo.lock"
    echo ""
fi

# Final status
if [ "$ERRORS" -gt 0 ]; then
    echo "ERROR: $ERRORS version mismatch(es) found!"
    echo ""
    echo "Run: ./scripts/sync-version.sh"
    exit 1
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "All 8 version locations match: $VERSION"
else
    echo "All files synced to version $VERSION"
    echo ""
    echo "Next steps:"
    echo "  1. git diff"
    echo "  2. git add -A && git commit -m 'chore: Bump to $VERSION'"
    echo "  3. git push"
    echo "  4. gh workflow run release.yml (or trigger via GitHub UI)"
fi
