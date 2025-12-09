#!/bin/bash
# Sync version from VERSION file to all package files
# Usage: ./scripts/sync-version.sh

set -e

cd "$(dirname "$0")/.."

VERSION=$(cat VERSION | tr -d '\n')

echo "Syncing version: $VERSION"
echo ""

# Cargo.toml (main)
sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" Cargo.toml
echo "  Cargo.toml"

# python/Cargo.toml
sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" python/Cargo.toml
echo "  python/Cargo.toml"

# node/Cargo.toml
sed -i '' "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" node/Cargo.toml
echo "  node/Cargo.toml"

# node/package.json
jq ".version = \"$VERSION\"" node/package.json > tmp.json && mv tmp.json node/package.json
echo "  node/package.json"

# node/wrapper/package.json (version + @omendb dep)
jq ".version = \"$VERSION\" | .dependencies[\"@omendb/omendb\"] = \"$VERSION\"" node/wrapper/package.json > tmp.json && mv tmp.json node/wrapper/package.json
echo "  node/wrapper/package.json"

# Update Cargo.lock
cargo check --quiet 2>/dev/null || true
echo "  Cargo.lock"

echo ""
echo "All files synced to version $VERSION"
echo ""
echo "Next: git diff && git commit -am 'chore: Bump to $VERSION' && git push"
