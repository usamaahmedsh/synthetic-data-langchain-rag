#!/usr/bin/env python3
# -*- coding: utf-8 -*-




"""
Run the synthetic query pipeline for multiple topics in one go.

- Each topic gets the full requested number of final queries.
- Failures for one topic do NOT stop the whole batch.
- Optionally aggregates all final queries into a single global JSONL file.
"""

from pathlib import Path
from typing import List, Dict, Any
import sys
import json
import traceback

# Ensure scripts/ is on the path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from main import _run_pipeline
from config.settings import OUTPUTS_DIR


def _topic_to_slug(topic: str) -> str:
    """Simple slugification for topic names."""
    return topic.strip().lower().replace(" ", "_")


def find_latest_run_dir(topic_slug: str) -> Path | None:
    """Find the latest run directory for a given topic_slug under OUTPUTS_DIR."""
    root = Path(OUTPUTS_DIR) / topic_slug
    if not root.exists():
        return None
    run_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    # Sort by name (timestamp-based naming) and pick last
    run_dirs.sort()
    return run_dirs[-1]


def append_to_global(topic_slug: str, run_dir: Path, global_out: Path) -> None:
    """Append final queries for this topic to a global JSONL file."""
    final_path = run_dir / "final_queries.jsonl"
    if not final_path.exists():
        print(f"  ⚠ No final_queries.jsonl found for {topic_slug} in {run_dir}")
        return

    with final_path.open("r") as fin, global_out.open("a") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["topic_slug"] = topic_slug
            fout.write(json.dumps(obj) + "\n")


def run_for_topics(
    topics: List[str],
    num_final_queries: int,
    aggregate_global: bool = True,
) -> None:
    """Run the pipeline for each topic, continue on errors, optionally aggregate outputs."""
    failed: List[Dict[str, Any]] = []

    global_out: Path | None = None
    if aggregate_global:
        global_out = Path(OUTPUTS_DIR) / "global_final_queries.jsonl"
        global_out.parent.mkdir(parents=True, exist_ok=True)
        if global_out.exists():
            global_out.unlink()  # start fresh

    for idx, topic in enumerate(topics):
        print("\n" + "=" * 80)
        print(f"[{idx+1}/{len(topics)}] Running pipeline for topic: {topic}")
        print("=" * 80 + "\n")

        try:
            _run_pipeline(topic_name=topic, num_final_queries=num_final_queries)

            if aggregate_global and global_out is not None:
                topic_slug = _topic_to_slug(topic)
                run_dir = find_latest_run_dir(topic_slug)
                if run_dir:
                    append_to_global(topic_slug, run_dir, global_out)
                else:
                    print(f"  ⚠ Could not find run directory for topic_slug='{topic_slug}'")
        except KeyboardInterrupt:
            print("\n✗ Interrupted by user. Stopping batch run.")
            break
        except Exception as e:
            print(f"\n✗ Error while processing topic '{topic}': {e}")
            traceback.print_exc()
            failed.append({"topic": topic, "error": str(e)})
            print("→ Skipping this topic and continuing with the next.\n")
            continue

    if failed:
        print("\n" + "=" * 80)
        print("Summary of failed topics")
        print("=" * 80)
        for f in failed:
            print(f"  - {f['topic']}: {f['error']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run pipeline for multiple topics.")
    parser.add_argument(
        "--topics",
        nargs="+",
        required=True,
        help="List of topics/persons, e.g. --topics 'Albert Einstein' 'Imran Khan'",
    )
    parser.add_argument(
        "--num-final-queries",
        type=int,
        default=100,
        help="Desired TOTAL number of final queries per topic",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Disable global aggregation into outputs/global_final_queries.jsonl",
    )
    args = parser.parse_args()

    run_for_topics(
        topics=args.topics,
        num_final_queries=args.num_final_queries,
        aggregate_global=not args.no_aggregate,
    )


if __name__ == "__main__":
    main()
