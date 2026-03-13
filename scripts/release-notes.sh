#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    echo "Usage: ./scripts/release-notes.sh <version> [--body-only]" >&2
    echo "Example: ./scripts/release-notes.sh 0.0.32 --body-only" >&2
    exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
fi

VERSION="$1"
BODY_ONLY="${2:-}"

if [[ "$VERSION" != v* ]]; then
    VERSION="v$VERSION"
fi

if [ -n "$BODY_ONLY" ] && [ "$BODY_ONLY" != "--body-only" ]; then
    usage
fi

awk -v section="## $VERSION" -v body_only="$BODY_ONLY" '
BEGIN {
    found = 0
    printing = 0
    first_line = 1
}
$0 == section {
    found = 1
    printing = 1
}
printing {
    if ($0 ~ /^## / && $0 != section) {
        exit
    }

    if (body_only == "--body-only" && first_line) {
        first_line = 0
        next
    }

    print
    first_line = 0
}
END {
    if (!found) {
        exit 2
    }
}
' CHANGELOG.md
