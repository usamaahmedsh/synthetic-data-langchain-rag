#!/bin/bash
# run_llama_server.sh
# Optimized llama.cpp server configuration for M4 Pro Mac
# For SynData query generation pipeline

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# Model path - UPDATE THIS to your actual model path
MODEL_PATH="$HOME/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Server settings
HOST="127.0.0.1"
PORT=8080

# Context and parallel settings
CONTEXT_SIZE=8192        # Total KV cache size
PARALLEL_REQUESTS=12     # Max simultaneous requests (-np)
BATCH_SIZE=2048          # Logical batch size for token processing (-b)
UBATCH_SIZE=512          # Physical batch size for memory efficiency (-ub)

# GPU settings (Metal on Mac)
GPU_LAYERS=99            # Offload all layers to Metal GPU (-ngl)

# Performance tuning
THREADS=0                # 0 = auto-detect optimal thread count

# Model name for OpenAI API compatibility
MODEL_ALIAS="llama-3.1-8b-q4_k_m"

# ============================================
# Validation
# ============================================

echo "════════════════════════════════════════"
echo "  LLaMA.cpp Server Launcher"
echo "════════════════════════════════════════"
echo ""

# Check if llama-server exists
if ! command -v llama-server &> /dev/null; then
    if [ -f "./llama-server" ]; then
        LLAMA_SERVER="./llama-server"
        echo "✓ Found llama-server in current directory"
    elif [ -f "./llama.cpp/llama-server" ]; then
        LLAMA_SERVER="./llama.cpp/llama-server"
        echo "✓ Found llama-server in ./llama.cpp/"
    else
        echo "✗ Error: llama-server not found!"
        echo "  Please ensure llama-server is in your PATH or current directory"
        echo "  Build it with: cd llama.cpp && make llama-server"
        exit 1
    fi
else
    LLAMA_SERVER="llama-server"
    echo "✓ Found llama-server in PATH"
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Error: Model not found at: $MODEL_PATH"
    echo "  Please update MODEL_PATH in this script"
    exit 1
fi
echo "✓ Model found: $MODEL_PATH"
echo ""

# ============================================
# System info
# ============================================

echo "Configuration:"
echo "──────────────────────────────────────"
echo "  Host:              $HOST:$PORT"
echo "  Model:             $(basename $MODEL_PATH)"
echo "  Context size:      $CONTEXT_SIZE tokens"
echo "  Parallel requests: $PARALLEL_REQUESTS"
echo "  Batch size:        $BATCH_SIZE"
echo "  UBatch size:       $UBATCH_SIZE"
echo "  GPU layers:        $GPU_LAYERS (Metal)"
echo "  Threads:           $THREADS (auto)"
echo ""

# Calculate effective parallelism
TOKENS_PER_REQUEST=$((CONTEXT_SIZE / PARALLEL_REQUESTS))
echo "Note: With these settings, you can process:"
echo "  → $PARALLEL_REQUESTS requests of ~$TOKENS_PER_REQUEST tokens each"
echo "  → Or fewer requests with longer contexts"
echo ""

# ============================================
# Launch server
# ============================================

echo "Starting llama.cpp server..."
echo "──────────────────────────────────────"
echo "Press Ctrl+C to stop"
echo ""

$LLAMA_SERVER \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CONTEXT_SIZE" \
  --parallel "$PARALLEL_REQUESTS" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --threads "$THREADS" \
  --n-gpu-layers "$GPU_LAYERS" \
  --cont-batching \
  --metrics \
  --alias "$MODEL_ALIAS"

# ============================================
# Notes on flags
# ============================================
# --model (-m)           : Path to GGUF model file
# --host                 : Server host address
# --port                 : Server port
# --ctx-size (-c)        : Total KV cache size (context window)
# --parallel (-np)       : Max parallel sequences/requests
# --batch-size (-b)      : Logical batch size for processing
# --ubatch-size (-ub)    : Physical batch size (memory control)
# --threads (-t)         : CPU threads (0 = auto)
# --n-gpu-layers (-ngl)  : GPU layers to offload (99 = all)
# --cont-batching (-cb)  : Enable continuous batching (critical!)
# --metrics              : Enable /metrics endpoint for monitoring
# --log-format           : Log format (text/json)
# --alias                : Model name for OpenAI API compatibility
