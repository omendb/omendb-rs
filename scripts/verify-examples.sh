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
		bun install --frozen-lockfile
	fi
	if [ ! -f omendb.node ]; then
		bun run build
	fi
	bun run examples/quickstart.js
)
