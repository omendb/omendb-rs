#!/bin/bash
set -e
cd /Users/nick/github/omendb/omendb/node

for pkg in darwin-arm64 darwin-x64 linux-arm64-gnu linux-x64-gnu; do
  echo "=== Publishing @omendb/omendb-$pkg ==="
  cd npm/$pkg
  npm publish --access public --tag alpha
  cd ../..
done

echo "=== All packages published! ==="
