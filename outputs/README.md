# Pipeline Outputs

    This directory contains organized outputs from query generation pipeline runs.

    ## Directory Structure

    outputs/
    ├── {topic_slug}/ # One folder per topic
    │ ├── {timestamp}/ # Timestamped run (YYYYMMDD_HHMMSS)
    │ │ ├── final_queries.jsonl # ✅ Final selected queries
    │ │ ├── candidate_queries.jsonl # All generated candidates
    │ │ ├── scored_queries.jsonl # Scored candidates
    │ │ ├── topics.json # Discovered topics
    │ │ ├── topics_deduped.json # Deduplicated topics
    │ │ ├── pipeline_metrics.json # Run statistics
    │ │ ├── config.json # Configuration used
    │ │ └── logs/
    │ │ └── pipeline.log # Detailed logs
    │ │
    │ └── latest -> {timestamp}/ # Symlink to most recent run

    text

    ## File Descriptions

    ### Core Outputs

    - **final_queries.jsonl**: Final queries after scoring and rejection sampling
    - Format: One query per line (JSONL)
    - Fields: `query`, `query_type`, `topic_id`, `total_score`, etc.

    - **candidate_queries.jsonl**: All queries generated before filtering
    - Useful for analysis of what was generated

    - **scored_queries.jsonl**: All candidates with scores
    - Includes quality, diversity, and total scores

    ### Topic Metadata

    - **topics.json**: Topics discovered by BERTopic
    - **topics_deduped.json**: Topics after deduplication

    ### Run Metadata

    - **pipeline_metrics.json**: Statistics about the run
    - Number of queries generated/selected
    - Timing information
    - Score distributions

    - **config.json**: Exact configuration used for the run
    - Model names
    - Hyperparameters
    - Over-generation factors

    ## Usage

    ### Access Latest Run

    cd outputs/{topic_slug}/latest
    cat final_queries.jsonl

    text

    ### Compare Runs

    List all runs for a topic
    ls -lt outputs/{topic_slug}/

    Compare metrics
    diff outputs/{topic_slug}/20250203_140530/pipeline_metrics.json     outputs/{topic_slug}/20250203_151020/pipeline_metrics.json

    text

    ### Load Queries in Python

    import json
    from pathlib import Path

    Load latest queries
    latest_dir = Path("outputs/imran_khan/latest")
    queries = []
    with open(latest_dir / "final_queries.jsonl") as f:
    for line in f:
    queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries")

    text

    ## Retention Policy

    Consider periodically cleaning up old runs to save disk space:

    Keep only last 5 runs per topic
    find outputs//2 -maxdepth 0 -type d | sort -r | tail -n +6 | xargs rm -rf

    text

    ## Analyzing Results

    ### View Metrics Summary

    python -m json.tool outputs/{topic_slug}/latest/pipeline_metrics.json

    text

    ### Compare Query Quality Across Runs

    import json
    from pathlib import Path

    def load_metrics(run_dir):
    with open(run_dir / "pipeline_metrics.json") as f:
    return json.load(f)

    topic_dir = Path("outputs/imran_khan")
    runs = sorted([d for d in topic_dir.iterdir() if d.is_dir() and d.name != "latest"])

    for run in runs[-5:]: # Last 5 runs
    metrics = load_metrics(run)
    print(f"{run.name}: {metrics['queries']['final_selected']} queries, "
    f"avg score: {metrics['scores']['mean']:.3f}")

    text

    ## File Formats

    ### JSONL Format

    Each line is a valid JSON object:

    {"query": "what is python", "query_type": "informational", "total_score": 0.85}
    {"query": "python vs java", "query_type": "comparative", "total_score": 0.78}

    text

    ### Metrics Format

    {
    "run_timestamp": "2025-02-03T15:30:00",
    "duration_seconds": 245.3,
    "queries": {
    "candidates_generated": 90,
    "final_selected": 30
    },
    "scores": {
    "min": 0.42,
    "max": 0.95,
    "mean": 0.73
    }
    }

    text
    