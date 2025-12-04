# synthetic-data-langchain-rag

Synthetic query generation pipeline for **RAG** and **search / IR evaluation**.

This project builds a Wikipedia-based corpus for a given topic, discovers sub-topics, generates LLM-based search queries, and then **scores, samples, and deduplicates** them into a high-quality, diverse final set.

The design balances:

- Practical usefulness (easy to run, reusable outputs)
- IR-style grounding (BM25, passage retrieval)
- Modern NLP (embeddings, topic models, LLM generation)

---

## Features

- **Automatic corpus building** from Wikipedia for any person/topic
- **Topic discovery + deduplication** with BERTopic
- **Topic quality scoring** to focus on rich, well-covered topics
- **LLM-based query generation** via a local **llama.cpp** HTTP server
- **Heuristic query scoring** (BM25 + length + lexical + semantic diversity)
- **Rejection sampling** with percentile-based thresholds per query type
- **Global semantic deduplication**
- **Optional semantic diversity enforcement** (MMR-style)
- **Passage-level BM25 indexing** for finer relevance signals
- **Checkpoints, metrics, and profiling** for iterative runs

---

## Pipeline Overview

Given a topic like **“Albert Einstein”**, the pipeline:

1. **Builds a Wikipedia corpus** for the topic’s neighborhood.
2. **Discovers sub-topics** (e.g., relativity, Nobel Prize, biography, politics).
3. **Deduplicates and selects** the richest topics based on coverage and quality.
4. **Generates candidate queries** per topic and query type:
   - informational
   - exploratory
   - navigational
   - comparative
   - transactional
   - commercial_investigation
5. **Builds a passage-level BM25 index** over the corpus.
6. **Scores queries** with a heuristic that combines:
   - BM25 relevance to passages
   - length-based quality
   - lexical diversity
   - semantic diversity
7. **Runs rejection sampling** to hit the target number of queries per type using percentile thresholds.
8. **Deduplicates semantically** across all queries.
9. **Optionally enforces semantic diversity** (MMR-style) if you generated more than you need.

The end result is a **`final_queries.jsonl`** file containing realistic, grounded, and diverse queries for your topic.

---

## Technologies

- **Python** 3.12+
- **Topic modeling**: BERTopic
- **Sparse retrieval**: rank-bm25
- **Dense embeddings**: sentence-transformers
- **LLM inference**: llama.cpp HTTP server (local)
- **Deep learning backend**: PyTorch (CPU, CUDA, or Apple MPS)
- **Orchestration / utilities**: httpx, requests, tqdm, dotenv

---

## Repository Structure

Key components:

- `scripts/main.py`  
  Main entry point. Orchestrates the full pipeline:
  - corpus → topics → topic selection → query generation → BM25 → scoring → sampling → dedup → output

- `scripts/tools/langchain_tools.py`  
  LangChain-style tool wrappers:
  - `fetch_wikipedia_corpus`
  - `build_topics`
  - `dedupe_topics`
  - `generate_queries`
  - `heuristic_score_queries`

- `scripts/core/`  
  - `query_generator.py` – Asynchronous query generation via llama.cpp (`BuildQueries`)
  - `query_scorer.py` – Heuristic scoring (BM25 + length + lexical/semantic diversity)
  - `topic_scorer.py` – Topic quality scoring and ranking

- `scripts/postprocessing/`  
  - `sampling.py` – Rejection sampling + semantic diversity (MMR-style)
  - `deduplication.py` – Global semantic deduplication

- `scripts/utils/`  
  - `bm25_utils.py` – Passage-level BM25 index and retrieval
  - `text_utils.py` – Tokenization, normalization, simple text stats
  - `output_manager.py` – Output directories, JSON/JSONL saving, run summaries
  - `checkpoint.py`, `metrics.py`, `profiler.py` – Checkpoints, metrics, timing

- `config/settings.py`  
  Central configuration:
  - page limits
  - taxonomy (query types)
  - over-generation strategy
  - model names
  - paths and performance flags

- `data/`  
  Local Wikipedia corpus and artifacts.

- `outputs/`  
  Per-run outputs:
  - `topics.json`, `topics_deduped.json`
  - `candidate_queries.jsonl`, `scored_queries.jsonl`, `final_queries.jsonl`
  - `metrics.json`, `config.json`
  - human-readable run summary

---

## Installation

From the repo root:

```r
cd synthetic-data-langchain-rag

```


### 1. Create and activate a virtual environment

```r
python3 -m venv syn
source syn/bin/activate # macOS / Linux

```

or the equivalent command on your platform

### 2. Install dependencies

```r
pip install -r requirements.txt

```


---

## Starting the LLM Server (llama.cpp)

There is a **bash script** in this repository that starts the llama.cpp server with the correct model and flags.

From the repo root (or wherever the script lives), run:

```r
bash llama-cpp.sh

```

This script should:

- Point to your GGUF model (for example, `llama-3.2-3b-instruct.Q4_K_M.gguf`)
- Configure context length, GPU offload, port (usually `8080`), etc.

The query generator (`BuildQueries`) expects the server to be reachable at the URL configured in `config/settings.py` (commonly `http://127.0.0.1:8080`).

---

## Configuration

Most knobs live in `config/settings.py`. Important ones:

- **Corpus / topics**
  - `DEFAULT_MAX_PAGES` – Maximum number of Wikipedia pages to fetch
  - `DATA_DIR`, `ARTIFACTS_DIR` – Where to store corpus and artifacts

- **Query generation**
  - `HF_GENERATION_MODEL` – Logical model name (for logging)
  - `DEFAULT_TAXONOMY` – List of query types
  - `QUERIES_PER_TOPIC_PER_CATEGORY` – Base number of queries per topic × type

- **Over-generation**
  - `OVER_GENERATION_STRATEGY` – `"global"` or `"adaptive"`
  - `GLOBAL_OVER_GENERATION_FACTOR`
  - `CATEGORY_OVER_GENERATION` – Per-type multipliers

- **Scoring & pipeline behavior**
  - `USE_PASSAGE_LEVEL_BM25`
  - `USE_TOPIC_QUALITY_SCORING`
  - `USE_SEMANTIC_DIVERSITY`
  - `USE_GLOBAL_DEDUP`
  - `USE_CHECKPOINTS`

- **Performance**
  - `EMBEDDING_BATCH_SIZE`
  - `MAX_PARALLEL_TOPICS`
  - `MAX_PARALLEL_CATEGORIES`
  - `USE_GPU`, `IS_APPLE_SILICON`

You can treat these as experimentation controls to trade off **quality**, **diversity**, and **runtime**.

---

## How to Run

From the repo root:

### 1. Activate the virtual environment

```r
source syn/bin/activate

```

### 2. Start the LLM server

```r
bash llama-cpp.sh

```


Wait until the server is up and listening on the configured port.

### 3. Run the pipeline

```r
cd scripts
python3 main.py

```


You will be prompted:

```r

Enter topic/person for corpus (e.g. 'Imran Khan'): Albert Einstein
Enter desired TOTAL number of final queries (e.g. 30): 100

```


The pipeline will then:

- Build or reuse the Wikipedia corpus
- Build topics and deduplicate them
- Select rich topics for generation
- Generate candidate queries via the LLM
- Build a passage-level BM25 index
- Score queries (quality + diversity)
- Run rejection sampling and global deduplication
- Optionally enforce semantic diversity
- Save all outputs under `outputs/<topic_slug>/<timestamp>/`

At the end, it prints a summary and the top queries by score.

---

## Outputs

For a run on **“Albert Einstein”**, you’ll typically see:

- `outputs/albert_einstein/<timestamp>/topics.json`  
  All topics from BERTopic.

- `outputs/albert_einstein/<timestamp>/topics_deduped.json`  
  Deduplicated topics.

- `outputs/albert_einstein/<timestamp>/candidate_queries.jsonl`  
  Raw candidate queries from the LLM.

- `outputs/albert_einstein/<timestamp>/scored_queries.jsonl`  
  Candidate queries annotated with:
  - `quality_score`
  - `lex_div_score`
  - `sem_div_score`
  - `total_score`
  - and other metadata.

- `outputs/albert_einstein/<timestamp>/final_queries.jsonl`  
  Final queries selected after scoring, sampling, dedup, and optional diversity pruning.

- `outputs/albert_einstein/<timestamp>/metrics.json`  
  Metrics and timing per stage, score distribution, per-type counts.

- `outputs/albert_einstein/<timestamp>/config.json`  
  Configuration snapshot for reproducibility.

---

## Example Use Cases

- **RAG evaluation**
  - Use `final_queries.jsonl` as realistic query sets to benchmark retrieval and RAG pipelines.

- **Search / ranking experiments**
  - Inject synthetic queries into your index to test ranking, coverage, and robustness.

- **Ablation studies**
  - Toggle:
    - topic quality scoring on/off
    - passage vs. document-level BM25
    - over-generation factors
    - semantic diversity enforcement  
  - Compare outputs and metrics across runs.

- **Query analysis**
  - Study how different query types (informational, exploratory, comparative, etc.) distribute across topics and corpus segments.

---

## Notes / Future Directions

- The pipeline uses **heuristic scoring only** (no LLM-as-judge) for transparency and reproducibility.
- Easy extensions:
  - Swap embedding models
  - Add neural rerankers
  - Add an optional LLM judge as a final filter

The codebase is structured so you can tweak individual components (generation, scoring, sampling) and re-run the pipeline to see how the final query set changes.