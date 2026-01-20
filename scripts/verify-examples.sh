#!/usr/bin/env bash
set -euo pipefail

echo "Running Python quickstart"
(
  cd python
  uv run python examples/quickstart.py
)

echo "Running Node.js quickstart"
(
  cd node
  if [ ! -d node_modules ]; then
    npm install
  fi
  if [ ! -f omendb.node ]; then
    npm run build
  fi
  node examples/quickstart.js
)
