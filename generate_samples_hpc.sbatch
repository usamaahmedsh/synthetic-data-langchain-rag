#!/bin/bash
#SBATCH --job-name=syn-100k-gpu
#SBATCH --account=liangmingpan
#SBATCH --partition=gpu_standard              
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


module load python/3.11/3.11.4
module load cuda12/12.4.1             
module load cmake                    

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
    "Einstein" \
    "Newton" \
    "Tesla" \
    "Beethoven" \
    "Shakespeare" \
    "Napoleon" \
    "Hitler" \
    "Gandhi" \
    "Mandela" \
    "Cleopatra" \
    "Imran Khan" \
    "Donald Trump" \
    "Barack Obama" \
    "Joe Biden" \
    "Putin" \
    "Stalin" \
    "World War I" \
    "World War II" \
    "Cold War" \
    "French Revolution" \
    "Industrial Revolution" \
    "Renaissance" \
    "Moon landing" \
    "9/11 attacks" \
    "Chernobyl disaster" \
    "COVID-19 pandemic" \
    "Olympic Games" \
    "Black Death" \
    "Great Depression" \
    "Machine learning" \
    "Deep learning" \
    "Cryptocurrency" \
    "Blockchain" \
    "Internet" \
    "Smartphones" \
    "Facebook" \
    "Google" \
    "Apple" \
    "Amazon" \
    "Python" \
    "Linux" \
    "GitHub" \
    "YouTube" \
    "Instagram" \
    "TikTok" \
    "Climate change" \
    "Globalization" \
    "Democracy" \
    "Capitalism" \
  --num-final-queries 20000

