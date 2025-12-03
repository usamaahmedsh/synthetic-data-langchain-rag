"""Global configuration and constants for the pipeline."""

import os
from pathlib import Path
import platform


# ==========================================
# System Detection
# ==========================================

IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

if IS_APPLE_SILICON:
    print("✓ Detected Apple Silicon")
    # Set environment variables for optimal M-series performance
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ==========================================
# Wikipedia Corpus Settings
# ==========================================

DEFAULT_MAX_PAGES = 2000
DEFAULT_CAT_DEPTH = 2
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "LocalLLMTopicPipeline/0.1 (research)"

# Network settings - optimized for Apple Silicon
if IS_APPLE_SILICON:
    RATE_LIMIT_PER_SECOND = 10  # M4 can handle more
    MAX_CONCURRENT_REQUESTS = 20
else:
    RATE_LIMIT_PER_SECOND = 5
    MAX_CONCURRENT_REQUESTS = 10

DEFAULT_TIMEOUT = 30.0


# ==========================================
# Topic Modeling Settings
# ==========================================

MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 5
CHUNK_WORDS = 150
CHUNK_OVERLAP = 0
BASE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_NEIGHBORS = 10
UMAP_COMPONENTS = 10
FINETUNE_EMBEDDINGS = False


# ==========================================
# Query Generation Settings
# ==========================================

QUERIES_PER_TOPIC_PER_CATEGORY = 3
HF_GENERATION_MODEL = "llama-3.2-3b-instruct"

# LLM Server
LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080")
GENERATION_TEMPERATURE = 0.6
GENERATION_TOP_P = 0.9
GENERATION_MAX_NEW_TOKENS = 80

# Over-generation strategy
OVER_GENERATION_STRATEGY = "adaptive"  # Options: "global", "adaptive"
GLOBAL_OVER_GENERATION_FACTOR = 5.0

# Category-specific over-generation factors
CATEGORY_OVER_GENERATION = {
    "informational": 2.5,
    "exploratory": 3.5,
    "navigational": 2.0,
    "comparative": 4.0,
    "transactional": 3.5,
    "commercial_investigation": 4.0,
}

# Taxonomy
DEFAULT_TAXONOMY = [
    "informational",
    "exploratory",
    "navigational",
    "comparative",
    "transactional",
    "commercial_investigation",
]


# ==========================================
# Query Scoring Settings (Heuristic)
# ==========================================

SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MAX_SIM = 0.85
LEXICAL_MIN_DIST = 0.30

# BM25 Settings
BM25_TOP_N = 5
MAX_CONTEXT_CHARS = 2000


# ==========================================
# Query Scoring Settings (LLM Judge)
# ==========================================

JUDGE_MODEL_PATH = os.environ.get(
    "JUDGE_MODEL_PATH",
    str(Path.home() / "models" / "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
)
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_MAX_TOKENS = 80
JUDGE_BATCH_SIZE = 24
JUDGE_MAX_CONCURRENT = 12


# ==========================================
# Rejection Sampling Settings
# ==========================================

INITIAL_PERCENTILE = 0.90
PERCENTILE_STEP = 0.05
MIN_PERCENTILE = 0.50
MIN_PER_TOPIC_PER_TYPE = 1
ENFORCE_DEDUP = True
RANDOM_SEED = 42


# ==========================================
# Performance Settings (Hardware-Optimized)
# ==========================================

# Detect GPU/MPS availability
USE_GPU = True  # Will auto-detect CUDA, MPS, or fallback to CPU

# Platform-specific optimizations
if IS_APPLE_SILICON:
    # M4 Pro optimizations (24GB unified memory)
    EMBEDDING_BATCH_SIZE = 256        # Larger batches for unified memory
    MAX_PARALLEL_TOPICS = 8           # More concurrent topics
    MAX_PARALLEL_CATEGORIES = 6       # More concurrent categories
    DEFAULT_BATCH_SIZE = 256          # General batch size
    
    # Memory management
    MAX_MEMORY_PERCENT = 0.85         # Use up to 85% of available memory
    
    print(f"  → Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"  → Max parallel topics: {MAX_PARALLEL_TOPICS}")
    print(f"  → Max parallel categories: {MAX_PARALLEL_CATEGORIES}")
else:
    # Standard settings (CUDA GPU or CPU)
    EMBEDDING_BATCH_SIZE = 128
    MAX_PARALLEL_TOPICS = 5
    MAX_PARALLEL_CATEGORIES = 6
    DEFAULT_BATCH_SIZE = 128
    MAX_MEMORY_PERCENT = 0.75

# Cache settings
USE_CACHE = True
CACHE_DIR = ARTIFACTS_DIR / "cache"


# ==========================================
# Query Validation Settings
# ==========================================

# Validation thresholds
MIN_QUERY_LENGTH = 5
MAX_QUERY_LENGTH = 150
MIN_QUERY_WORDS = 2
MAX_QUERY_WORDS = 20
REQUIRE_TOPIC_RELEVANCE = True


# ==========================================
# Deduplication Settings
# ==========================================

# Global deduplication threshold
GLOBAL_DEDUP_SIMILARITY_THRESHOLD = 0.90
KEEP_HIGHEST_SCORE = True


# ==========================================
# Output Organization Settings
# ==========================================

# Output directory structure
USE_TIMESTAMPED_RUNS = True           # Create timestamp subfolder for each run
SAVE_INTERMEDIATE_ARTIFACTS = True    # Save candidates, scored queries, etc.
CREATE_LATEST_SYMLINK = True          # Create 'latest' symlink to most recent run

# Output filenames
FINAL_QUERIES_FILENAME = "final_queries.jsonl"
CANDIDATE_QUERIES_FILENAME = "candidate_queries.jsonl"
SCORED_QUERIES_FILENAME = "scored_queries.jsonl"
TOPICS_FILENAME = "topics.json"
TOPICS_DEDUPED_FILENAME = "topics_deduped.json"
PIPELINE_METRICS_FILENAME = "pipeline_metrics.json"
RUN_CONFIG_FILENAME = "config.json"
PIPELINE_LOG_FILENAME = "pipeline.log"


# ==========================================
# Feature Flags
# ==========================================

# Enable/disable pipeline features
ENABLE_CHECKPOINTS = True             # Save checkpoints for resume
ENABLE_VALIDATION = True              # Validate queries before scoring
ENABLE_GLOBAL_DEDUP = True            # Global deduplication across categories
ENABLE_PROFILING = True               # Track performance metrics
ENABLE_PROGRESS_BARS = True           # Show progress bars (tqdm)


# ==========================================
# Logging Settings
# ==========================================

LOG_LEVEL = "INFO"                    # Options: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True                    # Save logs to file
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==========================================
# Debug Settings
# ==========================================

DEBUG_MODE = False                    # Enable debug output
VERBOSE = True                        # Verbose console output
SAVE_DEBUG_ARTIFACTS = False          # Save extra debug files


# ==========================================
# Model Download Settings
# ==========================================

HF_CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional: for private models


# ==========================================
# Summary on Import
# ==========================================

def print_config_summary():
    """Print configuration summary on import."""
    print("\n" + "=" * 60)
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"Platform:           {platform.system()} {platform.machine()}")
    print(f"Apple Silicon:      {IS_APPLE_SILICON}")
    print(f"Use GPU:            {USE_GPU}")
    print(f"Batch Size:         {EMBEDDING_BATCH_SIZE}")
    print(f"Max Parallel:       {MAX_PARALLEL_TOPICS} topics, {MAX_PARALLEL_CATEGORIES} categories")
    print(f"Caching:            {USE_CACHE}")
    print(f"Validation:         {ENABLE_VALIDATION}")
    print(f"Global Dedup:       {ENABLE_GLOBAL_DEDUP}")
    print(f"Checkpoints:        {ENABLE_CHECKPOINTS}")
    print(f"Project Root:       {PROJECT_ROOT}")
    print("=" * 60 + "\n")


# Optionally print summary on import (comment out if too verbose)
if VERBOSE and __name__ != "__main__":
    print_config_summary()


# ==========================================
# Backward Compatibility Aliases
# ==========================================

# For code that references old variable names
DEFAULT_BATCH_SIZE = EMBEDDING_BATCH_SIZE
