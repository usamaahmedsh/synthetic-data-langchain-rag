#!/bin/bash
#SBATCH --job-name=syn-100k-gpu
#SBATCH --partition=gpu              # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python/3.11/3.11.4
module load cuda12/12.4.1                   # or cuda12/12.5
module load cmake                    # if needed

source "$HOME/Documents/Github/synthetic-data-langchain-rag/syn/bin/activate"

cd "$HOME/Documents/Github/synthetic-data-langchain-rag"

# Ensure llama-server from CMake build is on PATH
export PATH="$HOME/llama.cpp/build/bin:$PATH"

mkdir -p "$HOME/llama_logs"
mkdir -p logs

# 1) Start llama.cpp server (CPU/GPU auto-detect)
bash $HOME/Documents/Github/synthetic-data-langchain-rag/llama-cpp-hpc.sh

# 2) Run multi-topic generator for ~100k final queries
cd $HOME/Documents/Github/synthetic-data-langchain-rag/scripts

python3 run_multi_topics.py \
  --topics \
    "Albert Einstein" \
    "Imran Khan" \
    "Donald Trump" \
    "Machine learning" \
    "Deep learning" \
    "Large language models" \
    "Information retrieval" \
    "Neural networks" \
    "Reinforcement learning" \
    "Natural language processing" \
  --num-final-queries 100000
