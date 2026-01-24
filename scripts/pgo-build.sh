#!/bin/bash
# PGO (Profile-Guided Optimization) build for OmenDB
# Provides ~9% build improvement, ~7% search improvement
#
# Prerequisites:
#   rustup component add llvm-tools
#
# Usage:
#   ./scripts/pgo-build.sh           # Build Rust library with PGO
#   ./scripts/pgo-build.sh python    # Build Python bindings with PGO
set -e

PROFILE_DIR="/tmp/omendb-pgo-profiles"
MERGED_PROFILE="/tmp/omendb-pgo.profdata"

# Find llvm-profdata from Rust toolchain
LLVM_PROFDATA=$(find "$(rustc --print sysroot)/lib/rustlib" -name "llvm-profdata" 2>/dev/null | head -1)
if [ -z "$LLVM_PROFDATA" ]; then
    echo "Error: llvm-profdata not found. Run: rustup component add llvm-tools"
    exit 1
fi

echo "=== PGO Build for OmenDB ==="
echo "llvm-profdata: $LLVM_PROFDATA"

# Step 1: Clean previous profiles
rm -rf "$PROFILE_DIR" "$MERGED_PROFILE"
mkdir -p "$PROFILE_DIR"

# Step 2: Build with instrumentation
echo ""
echo "Step 1/4: Building with instrumentation..."
RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" cargo build --release --lib 2>&1 | tail -3

# Step 3: Run profiling workload
echo ""
echo "Step 2/4: Running profiling workload..."
if [ "$1" = "python" ]; then
    # Python bindings - use Python workload
    cd python
    RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" uv run maturin develop --release 2>&1 | tail -3
    for i in 1 2 3; do
        UV_PYTHON=3.13 uv run python pgo_workload.py
    done
    cd ..
else
    # Rust library - use test workload
    RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" cargo test --release --lib -- \
        test_parallel_build \
        test_basic_search \
        test_sq8_basic \
        2>&1 | grep -E "(running|test result|passed)" || true
fi

# Step 4: Merge profiles
echo ""
echo "Step 3/4: Merging profiles..."
"$LLVM_PROFDATA" merge -o "$MERGED_PROFILE" "$PROFILE_DIR"/*.profraw
echo "Profile size: $(du -h $MERGED_PROFILE | cut -f1)"

# Step 5: Rebuild with PGO
echo ""
echo "Step 4/4: Building with PGO..."
if [ "$1" = "python" ]; then
    cd python
    RUSTFLAGS="-Cprofile-use=$MERGED_PROFILE" uv run maturin develop --release 2>&1 | tail -5
    cd ..
else
    RUSTFLAGS="-Cprofile-use=$MERGED_PROFILE" cargo build --release --lib 2>&1 | tail -3
fi

echo ""
echo "=== PGO Build Complete ==="
echo "Run benchmarks to verify: python benchmark.py"
