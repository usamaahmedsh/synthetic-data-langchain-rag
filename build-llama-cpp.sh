#!/usr/bin/env bash
# build_llama_cpp.sh
# Build llama.cpp (llama-server) with CMake, CPU/GPU agnostic.

set -euo pipefail

# ============================================
# Configuration
# ============================================

# Root of llama.cpp (change if needed)
LLAMA_ROOT="${LLAMA_ROOT:-$HOME/llama.cpp}"

# Build directory
BUILD_DIR="${BUILD_DIR:-$LLAMA_ROOT/build}"

# Number of build threads
JOBS="${JOBS:-8}"

# ============================================
# Helper: detect CUDA
# ============================================

detect_cuda() {
  # Prefer nvidia-smi if available
  if command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  # Fallback: check common CUDA library paths
  if ldconfig -p 2>/dev/null | grep -qi "libcuda.so"; then
    return 0
  fi

  # macOS / others: no CUDA
  return 1
}

# ============================================
# Start
# ============================================

echo "════════════════════════════════════════"
echo "  Building llama.cpp (llama-server)"
echo "════════════════════════════════════════"
echo ""

if [ ! -d "$LLAMA_ROOT" ]; then
  echo "✗ Error: LLAMA_ROOT directory does not exist: $LLAMA_ROOT"
  echo "  Set LLAMA_ROOT or clone llama.cpp, e.g.:"
  echo "    git clone https://github.com/ggml-org/llama.cpp.git \$HOME/llama.cpp"
  exit 1
fi

cd "$LLAMA_ROOT"

# Decide on CUDA
CMAKE_EXTRA_FLAGS=""
if detect_cuda; then
  echo "✓ Detected NVIDIA GPU / CUDA environment"
  CMAKE_EXTRA_FLAGS="-DGGML_CUDA=ON"
else
  echo "ℹ No CUDA detected, building CPU-only version"
  CMAKE_EXTRA_FLAGS="-DGGML_CUDA=OFF"
fi

echo ""
echo "Using configuration:"
echo "  LLAMA_ROOT     = $LLAMA_ROOT"
echo "  BUILD_DIR      = $BUILD_DIR"
echo "  JOBS           = $JOBS"
echo "  CMAKE FLAGS    = $CMAKE_EXTRA_FLAGS"
echo ""

mkdir -p "$BUILD_DIR"

# Configure
echo "Running CMake configure..."
cmake -S "$LLAMA_ROOT" -B "$BUILD_DIR" $CMAKE_EXTRA_FLAGS

# Build llama-server target
echo ""
echo "Building llama-server (Release)..."
cmake --build "$BUILD_DIR" --config Release -j "$JOBS" --target llama-server

# Result
SERVER_BIN="$BUILD_DIR/bin/llama-server"

echo ""
if [ -x "$SERVER_BIN" ]; then
  echo "✓ Build complete."
  echo "  llama-server binary: $SERVER_BIN"
  echo ""
  echo "You can add it to PATH, e.g.:"
  echo "  export PATH=\"$BUILD_DIR/bin:\$PATH\""
else
  echo "✗ Build finished but llama-server not found at: $SERVER_BIN"
  echo "  Check CMake output above for errors."
  exit 1
fi
