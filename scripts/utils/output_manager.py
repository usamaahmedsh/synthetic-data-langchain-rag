"""Output directory and file management utilities."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class OutputManager:
    """
    Manages organized output structure for pipeline runs.
    
    Creates structure:
        outputs/{topic_slug}/{timestamp}/
            ├── final_queries.jsonl
            ├── candidate_queries.jsonl
            ├── scored_queries.jsonl
            ├── topics.json
            ├── topics_deduped.json
            ├── pipeline_metrics.json
            ├── config.json
            └── logs/
                └── pipeline.log
    """

    def __init__(
        self,
        topic_slug: str,
        base_output_dir: Path,
        use_timestamp: bool = True,
        create_symlink: bool = True,
    ):
        """
        Initialize OutputManager.
        
        Args:
            topic_slug: Slugified topic name
            base_output_dir: Base outputs directory
            use_timestamp: Whether to create timestamped subdirectory
            create_symlink: Whether to create 'latest' symlink
        """
        self.topic_slug = topic_slug
        self.base_output_dir = Path(base_output_dir)
        self.use_timestamp = use_timestamp
        self.create_symlink = create_symlink
        
        # Create topic directory
        self.topic_dir = self.base_output_dir / topic_slug
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory (timestamped or not)
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.topic_dir / timestamp
        else:
            self.run_dir = self.topic_dir
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs subdirectory
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup symlink to latest run
        if use_timestamp and create_symlink:
            self._create_latest_symlink()
    
    def _create_latest_symlink(self):
        """Create or update 'latest' symlink to current run."""
        latest_link = self.topic_dir / "latest"
        
        # Remove existing symlink if it exists
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            # It's a real directory/file, don't touch it
            print(f"  Note: 'latest' exists as a real path, not creating symlink")
            return
        
        # Create new symlink (relative path for portability)
        try:
            latest_link.symlink_to(self.run_dir.name)
        except (OSError, NotImplementedError):
            # Symlinks may not work on all systems (e.g., Windows without admin)
            print(f"  Note: Could not create 'latest' symlink (OS limitation)")
    
    def get_path(self, filename: str) -> Path:
        """
        Get full path for a given filename in the run directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to file in run directory
        """
        return self.run_dir / filename
    
    def get_log_path(self, filename: str) -> Path:
        """
        Get full path for a given filename in the logs directory.
        
        Args:
            filename: Name of the log file
            
        Returns:
            Full path to file in logs directory
        """
        return self.logs_dir / filename
    
    def save_json(self, data: Any, filename: str, indent: int = 2):
        """
        Save data as JSON to run directory.
        
        Args:
            data: Data to save (must be JSON-serializable)
            filename: Name of the output file
            indent: JSON indentation level
        """
        path = self.get_path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        print(f"  ✓ Saved: {path}")
    
    def save_jsonl(self, rows: List[Dict], filename: str):
        """
        Save list of dicts as JSONL to run directory.
        
        Args:
            rows: List of dictionaries to save
            filename: Name of the output file
        """
        path = self.get_path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"  ✓ Saved: {path}")
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save pipeline configuration for reproducibility.
        
        Args:
            config: Configuration dictionary
        """
        self.save_json(config, "config.json")
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """
        Save pipeline metrics/statistics.
        
        Args:
            metrics: Metrics dictionary
        """
        self.save_json(metrics, "pipeline_metrics.json")
    
    def get_summary(self) -> str:
        """
        Get formatted summary of output locations and files.
        
        Returns:
            Formatted summary string
        """
        summary = f"""
{'=' * 60}
Output Summary
{'=' * 60}
Topic:       {self.topic_slug}
Run dir:     {self.run_dir}
Latest link: {self.topic_dir / 'latest' if self.use_timestamp else 'N/A'}

Files saved:
"""
        # List all files in run directory
        files = []
        for item in self.run_dir.iterdir():
            if item.is_file():
                size = item.stat().st_size
                files.append((item.name, size))
        
        # Sort by name
        files.sort()
        
        for name, size in files:
            summary += f"  - {name:30s} {self._format_size(size):>10s}\n"
        
        return summary
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


def create_outputs_readme(outputs_dir: Path):
    """
    Create README.md in outputs directory explaining structure.
    
    Args:
        outputs_dir: Path to outputs directory
    """
    readme_path = outputs_dir / "README.md"
    
    if readme_path.exists():
        return  # Don't overwrite existing README
    
    content = """# Pipeline Outputs

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
    diff outputs/{topic_slug}/20250203_140530/pipeline_metrics.json \
    outputs/{topic_slug}/20250203_151020/pipeline_metrics.json

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
    """
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content, encoding='utf-8')
    print(f"  ✓ Created README at {readme_path}")