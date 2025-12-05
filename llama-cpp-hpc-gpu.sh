#!/bin/bash
# run_llama_server_hpc_gpu.sh
# Llama.cpp server launcher for HPC (GPU offload, auto-download model, background)

set -euo pipefail

# ============================================
# Configuration
# ============================================

# Model repo + filename on Hugging Face
HF_REPO="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
HF_FILENAME="Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

# Local model path
MODEL_DIR="$HOME/models"
MODEL_PATH="$MODEL_DIR/$HF_FILENAME"

# Server settings
HOST="127.0.0.1"
PORT=8080

# Context and parallel settings (more conservative for GPU stability)
CONTEXT_SIZE=4096        # Total KV cache size
PARALLEL_REQUESTS=4      # Max simultaneous requests (-np)
BATCH_SIZE=512           # Logical batch size (-b)
UBATCH_SIZE=256          # Physical batch size (-ub)

# GPU offload
GPU_LAYERS=99            # Offload all layers (tune down if VRAM is tight)

# Threads: default to SLURM_CPUS_PER_TASK if set, else auto (0)
THREADS="${SLURM_CPUS_PER_TASK:-0}"

# Model alias for OpenAI-compatible APIs
MODEL_ALIAS="llama-3.1-8b-q6_k_l"

# Where to log server stdout/stderr
LOG_DIR="${LOG_DIR:-$HOME/llama_logs}"
LOG_FILE="$LOG_DIR/llama_server_gpu_${PORT}.log"

# ============================================
# Helper: download model if missing
# ============================================

download_model() {
  echo "Model not found at: $MODEL_PATH"
  echo "Attempting to download from Hugging Face: $HF_REPO/$HF_FILENAME"
  mkdir -p "$MODEL_DIR"

  if command -v curl >/dev/null 2>&1; then
    curl -L \
      "https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}" \
      -o "$MODEL_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_PATH" \
      "https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}"
  else
    echo "✗ Error: neither curl nor wget is available on this system."
    exit 1
  fi

  echo "✓ Downloaded model to: $MODEL_PATH"
}

# ============================================
# Validation
# ============================================

echo "════════════════════════════════════════"
echo "  LLaMA.cpp Server Launcher (HPC GPU)"
echo "════════════════════════════════════════"
echo ""

# Check llama-server
if ! command -v llama-server &> /dev/null; then
  if [ -f "./llama-server" ]; then
    LLAMA_SERVER="./llama-server"
    echo "✓ Found llama-server in current directory"
  elif [ -f "./llama.cpp/llama-server" ]; then
    LLAMA_SERVER="./llama.cpp/llama-server"
    echo "✓ Found llama-server in ./llama.cpp/"
  else
    echo "✗ Error: llama-server not found!"
    echo "  Ensure llama-server is in PATH or current directory."
    echo "  Build it with: cd llama.cpp && make llama-server"
    exit 1
  fi
else
  LLAMA_SERVER="llama-server"
  echo "✓ Found llama-server in PATH"
fi

# Ensure model exists (download if not)
if [ ! -f "$MODEL_PATH" ]; then
  download_model
fi
echo "✓ Model ready: $MODEL_PATH"
echo ""

mkdir -p "$LOG_DIR"

# ============================================
# System info
# ============================================

echo "Configuration:"
echo "──────────────────────────────────────"
echo "  Host:              $HOST:$PORT"
echo "  Model:             $(basename "$MODEL_PATH")"
echo "  Context size:      $CONTEXT_SIZE tokens"
echo "  Parallel requests: $PARALLEL_REQUESTS"
echo "  Batch size:        $BATCH_SIZE"
echo "  UBatch size:       $UBATCH_SIZE"
echo "  GPU layers:        $GPU_LAYERS"
echo "  Threads:           $THREADS (0 = auto / SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK)"
echo "  Alias:             $MODEL_ALIAS"
echo "  Log file:          $LOG_FILE"
echo ""

TOKENS_PER_REQUEST=$((CONTEXT_SIZE / PARALLEL_REQUESTS))
echo "Note: With these settings, you can process:"
echo "  → $PARALLEL_REQUESTS requests of ~$TOKENS_PER_REQUEST tokens each"
echo ""

# ============================================
# Launch server in background
# ============================================

echo "Starting llama.cpp server (GPU) in background..."
echo "Logs: $LOG_FILE"
echo ""

nohup "$LLAMA_SERVER" \
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
  --alias "$MODEL_ALIAS" \
  > "$LOG_FILE" 2>&1 &

SERVER_PID=$!

echo "✓ llama-server (GPU) started with PID $SERVER_PID"
echo "Waiting a few seconds to ensure it is up..."
sleep 5

echo "Your Python pipeline can now call:"
echo "  http://$HOST:$PORT/v1/chat/completions"
echo ""
