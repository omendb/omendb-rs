#!/bin/bash
# Bump version across all package files
# Usage:
#   ./scripts/bump-version.sh             # Patch bump (0.0.5 -> 0.0.6)
#   ./scripts/bump-version.sh --minor     # Minor bump (0.0.5 -> 0.1.0)
#   ./scripts/bump-version.sh --major --force  # Major bump (0.0.5 -> 1.0.0)

set -e

cd "$(dirname "$0")/.."

# Parse arguments
BUMP_TYPE="patch"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minor)
            BUMP_TYPE="minor"
            shift
            ;;
        --major)
            BUMP_TYPE="major"
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--minor] [--major --force]"
            echo ""
            echo "Options:"
            echo "  (default)         Patch bump (0.0.5 -> 0.0.6)"
            echo "  --minor           Minor bump (0.0.5 -> 0.1.0)"
            echo "  --major --force   Major bump (0.0.5 -> 1.0.0)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Major bump requires --force
if [ "$BUMP_TYPE" = "major" ] && [ "$FORCE" = "false" ]; then
    echo "ERROR: Major version bump requires --force flag"
    echo "Usage: $0 --major --force"
    exit 1
fi

# Get published version (source of truth)
echo "Fetching published versions..."
PYPI_VERSION=$(curl -sf https://pypi.org/pypi/omendb/json | jq -r '.info.version' 2>/dev/null || echo "0.0.0")
CRATES_VERSION=$(curl -sf https://crates.io/api/v1/crates/omendb | jq -r '.versions[0].num' 2>/dev/null || echo "0.0.0")

echo "  PyPI:   $PYPI_VERSION"
echo "  crates: $CRATES_VERSION"

# Use highest published version as base
if [ "$(printf '%s\n' "$PYPI_VERSION" "$CRATES_VERSION" | sort -V | tail -1)" = "$PYPI_VERSION" ]; then
    BASE_VERSION="$PYPI_VERSION"
else
    BASE_VERSION="$CRATES_VERSION"
fi

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$BASE_VERSION"

# Calculate new version
case "$BUMP_TYPE" in
    major)
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    minor)
        NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
        ;;
    patch)
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
esac

echo ""
echo "Bump type: $BUMP_TYPE"
echo "Bumping:   $BASE_VERSION -> $NEW_VERSION"
echo ""

# Update all files
echo "Updating version files..."

sed -i '' "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" Cargo.toml
echo "  Cargo.toml"

sed -i '' "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" python/Cargo.toml
echo "  python/Cargo.toml"

sed -i '' "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" node/Cargo.toml
echo "  node/Cargo.toml"

sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" node/package.json
echo "  node/package.json"

# node/wrapper needs both version AND dependency updated
sed -i '' "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" node/wrapper/package.json
sed -i '' "s/\"@omendb\/omendb\": \"[^\"]*\"/\"@omendb\/omendb\": \"$NEW_VERSION\"/" node/wrapper/package.json
echo "  node/wrapper/package.json"

sed -i '' "s/> \*\*v[0-9.]*\*\*:/> **v$NEW_VERSION**:/" README.md
echo "  README.md"

# Verify ALL versions match
echo ""
echo "Verification:"
CARGO_V=$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)
PYTHON_V=$(grep '^version = ' python/Cargo.toml | head -1 | cut -d'"' -f2)
NODE_CARGO_V=$(grep '^version = ' node/Cargo.toml | head -1 | cut -d'"' -f2)
NODE_V=$(grep '"version"' node/package.json | head -1 | cut -d'"' -f4)
WRAPPER_V=$(grep '"version"' node/wrapper/package.json | head -1 | cut -d'"' -f4)
WRAPPER_DEP_V=$(jq -r '.dependencies["@omendb/omendb"]' node/wrapper/package.json)

ALL_MATCH=true
echo "  Cargo.toml:              $CARGO_V"
echo "  python/Cargo.toml:       $PYTHON_V"
echo "  node/Cargo.toml:         $NODE_CARGO_V"
echo "  node/package.json:       $NODE_V"
echo "  node/wrapper version:    $WRAPPER_V"
echo "  node/wrapper @omendb:    $WRAPPER_DEP_V"

for v in "$CARGO_V" "$PYTHON_V" "$NODE_CARGO_V" "$NODE_V" "$WRAPPER_V" "$WRAPPER_DEP_V"; do
    if [ "$v" != "$NEW_VERSION" ]; then
        ALL_MATCH=false
    fi
done

if [ "$ALL_MATCH" = "true" ]; then
    echo ""
    echo "All 6 version locations updated to $NEW_VERSION"
else
    echo ""
    echo "ERROR: Version mismatch detected!"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  git diff"
echo "  git commit -am 'chore: Bump version to $NEW_VERSION'"
echo "  git push"
echo "  gh workflow run Release"
