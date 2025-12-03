#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Main orchestrator for the LangChain-based query-generation pipeline.


Steps:
  1) Build Wikipedia corpus for a user topic (async tool, capped at 500 pages).
  2) Build BERTopic topics (with caching).
  3) Deduplicate topics.
  4) Generate candidate queries via HF model (LangChain tool, async under the hood).
  5) Score queries with heuristic metrics (LangChain tool).
  6) Adaptive percentile-based rejection sampling to keep final queries.


All heavy logic lives in client modules and LangChain tools:
  - tools_pipeline.fetch_wikipedia_corpus
  - tools_pipeline.build_topics
  - tools_pipeline.dedupe_topics
  - tools_pipeline.generate_queries
  - tools_pipeline.heuristic_score_queries
  - rejection_sampler.RejectionSampler
"""


import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


from tools_pipeline import (
    fetch_wikipedia_corpus,
    build_topics,
    dedupe_topics,
    generate_queries,
    judge_queries,
    heuristic_score_queries,
)
from queries_builder import BuildQueries
from rejection_sampler import RejectionSampler


# Safely import load_dotenv; if python-dotenv is not installed, provide a no-op fallback
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        # dotenv is not available in the environment; behave as a no-op.
        return False


load_dotenv()


# -------------------------
# Configuration
# -------------------------


MAX_WIKI_PAGES = 2000
QUERIES_PER_TOPIC_PER_CATEGORY = 3
HF_GENERATION_MODEL = "llama-3.2-3b-instruct"
JUDGE_MODEL_PATH = "/Users/usamaahmedsh/models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

# Over-generation strategy
OVER_GENERATION_STRATEGY = "adaptive"  # Options: "global", "adaptive"
GLOBAL_OVER_GENERATION_FACTOR = 3.0

# Category-specific over-generation factors (for "adaptive" strategy)
# Some categories are harder to generate quality queries for
CATEGORY_OVER_GENERATION = {
    "informational": 2.5,           # Easier - factual queries
    "exploratory": 3.5,             # Medium - needs breadth
    "navigational": 2.0,            # Easier - specific destinations
    "comparative": 4.0,             # Harder - needs valid comparisons
    "transactional": 3.5,           # Medium - needs actionable intent
    "commercial_investigation": 4.0, # Harder - needs evaluation context
}


# -------------------------
# Simple helpers
# -------------------------


def ask_topic() -> str:
    topic = input("Enter topic/person for corpus (e.g. 'Imran Khan'): ").strip()
    while not topic:
        topic = input("Topic cannot be empty. Enter topic/person: ").strip()
    return topic


def ask_num_final_queries() -> int:
    raw = input("Enter desired TOTAL number of final queries (e.g. 30): ").strip()
    while not raw.isdigit() or int(raw) <= 0:
        raw = input("Please enter a positive integer (e.g. 30): ").strip()
    return int(raw)


def compute_per_type_targets(total: int, types: List[str]) -> Dict[str, int]:
    """
    Split a GLOBAL target 'total' across taxonomy types as evenly as possible.
    """
    n = max(1, len(types))
    base = total // n
    leftover = total % n
    targets = {
        t: base + (1 if i < leftover else 0)
        for i, t in enumerate(types)
    }
    return targets


def get_over_generation_factor(
    category: str,
    strategy: str = "global",
    global_factor: float = 3.0,
    category_factors: Dict[str, float] = None,
) -> float:
    """
    Get over-generation factor based on strategy.
    
    Args:
        category: Query category/type
        strategy: "global" (same for all) or "adaptive" (per-category)
        global_factor: Factor to use for global strategy
        category_factors: Dict of category-specific factors
        
    Returns:
        Over-generation factor for this category
    """
    if strategy == "global":
        return global_factor
    elif strategy == "adaptive" and category_factors:
        return category_factors.get(category, global_factor)
    else:
        return global_factor


def compute_adaptive_topic_distribution(
    num_final_queries: int,
    query_types: List[str],
    strategy: str,
    global_factor: float,
    category_factors: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute how many candidates to generate per category with adaptive over-generation.
    
    Returns:
        Dict mapping category -> {
            "target": final queries needed,
            "over_gen_factor": over-generation factor,
            "candidates": total candidates to generate
        }
    """
    # Split final queries evenly across categories
    base_per_type = compute_per_type_targets(num_final_queries, query_types)
    
    distribution = {}
    for qtype in query_types:
        target = base_per_type[qtype]
        factor = get_over_generation_factor(
            qtype, strategy, global_factor, category_factors
        )
        distribution[qtype] = {
            "target": target,
            "over_gen_factor": factor,
            "candidates": int(target * factor),
        }
    
    return distribution


def topic_richness(topic: Dict[str, Any], docs_root: Path) -> int:
    """
    Simple richness score: sum of character lengths of all relevant documents.
    Fallback to #representative_docs if no files found.
    """
    rel_docs = topic.get("relevant_documents", []) or []
    total_chars = 0
    for fname in rel_docs:
        fpath = docs_root / fname
        if not fpath.exists():
            continue
        try:
            txt = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        total_chars += len(txt)

    if total_chars == 0:
        reps = topic.get("representative_docs", []) or []
        return max(1, len(reps))
    return total_chars


def select_rich_topics(
    topics: List[Dict[str, Any]],
    docs_dir: Path,
    num_topics: int,
) -> List[Dict[str, Any]]:
    """
    Select top N richest topics by document coverage.
    """
    if not topics:
        return []

    scored: List[Tuple[Dict[str, Any], int]] = []
    for t in topics:
        score = topic_richness(t, docs_dir)
        scored.append((t, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = min(num_topics, len(scored))
    return [t for t, _ in scored[:selected]]


def calculate_optimal_topics(
    distribution: Dict[str, Dict[str, Any]],
    num_deduped_topics: int,
    queries_per_topic_per_category: int,
    num_categories: int,
) -> int:
    """
    Calculate optimal number of topics to use based on adaptive distribution.
    
    Args:
        distribution: Per-category distribution from compute_adaptive_topic_distribution
        num_deduped_topics: Total deduped topics available
        queries_per_topic_per_category: Queries generated per topic per category
        num_categories: Number of taxonomy categories
        
    Returns:
        Optimal number of topics to select
    """
    if num_deduped_topics == 0:
        return 0
    
    queries_per_topic = queries_per_topic_per_category * num_categories
    
    # Total candidates needed across all categories
    total_candidates = sum(info["candidates"] for info in distribution.values())
    
    # How many topics needed to hit target? (ceiling division)
    topics_needed = max(1, (total_candidates + queries_per_topic - 1) // queries_per_topic)
    
    # Use whatever is smaller: what we need or what we have
    optimal = min(topics_needed, num_deduped_topics)
    
    return optimal


# -------------------------
# Orchestrator (sync, tools hide async)
# -------------------------


def main() -> None:
    print("=== LangChain Query Generation Pipeline ===\n")

    # 0) User inputs
    topic_name = ask_topic()
    num_final_queries = ask_num_final_queries()

    query_types = BuildQueries.DEFAULT_TAXONOMY

    print("\nConfiguration:")
    print(f"  Topic name:         {topic_name}")
    print(f"  Final query target: {num_final_queries}")
    print(f"  Max Wikipedia pages:{MAX_WIKI_PAGES}")
    print(f"  Taxonomy categories:{len(query_types)} ({', '.join(query_types)})")
    print(f"  Queries per topic per category: {QUERIES_PER_TOPIC_PER_CATEGORY}")
    print(f"  Generation HF model:{HF_GENERATION_MODEL}")
    print(f"  Over-generation strategy: {OVER_GENERATION_STRATEGY}")
    if OVER_GENERATION_STRATEGY == "global":
        print(f"  Over-generation factor: {GLOBAL_OVER_GENERATION_FACTOR}x (global)")
    else:
        print(f"  Over-generation factors: adaptive per category")
    print(f"  Judge GGUF model:   {JUDGE_MODEL_PATH}")
    print("=" * 60 + "\n")

    base_data_dir = Path("data/raw")
    base_artifacts_dir = Path("artifacts")
    base_data_dir.mkdir(parents=True, exist_ok=True)
    base_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Corpus: Wikipedia (tool wraps async client)
    print("STEP 1: Building corpus from Wikipedia (LangChain tool)...")
    corpus_info = fetch_wikipedia_corpus.invoke(
        {
            "seed_title": topic_name,
            "max_pages": MAX_WIKI_PAGES,
            "cat_depth": 2,
            "output_dir": str(base_data_dir),
        }
    )
    topic_slug = corpus_info["topic_slug"]
    topic_name = corpus_info["topic_name"]
    corpus_topic_dir = Path(corpus_info["output_dir"]) / topic_slug / "pages"
    print(f"  ✓ Corpus ready at {corpus_topic_dir}\n")

    # 2) Topics: BERTopic (tool)
    print("STEP 2: Building topics with BERTopic (LangChain tool)...")
    topics_result = build_topics.invoke(
        {
            "pages_dir": str(corpus_topic_dir),
            "topic_name": topic_name,
            "topic_slug": topic_slug,
            "artifacts_root": str(base_artifacts_dir),
        }
    )
    topics = topics_result["topics"]
    topics_json_path = Path(f"topics_{topic_slug}.json")
    with topics_json_path.open("w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Topics built. Saved to {topics_json_path}\n")

    # 3) Topic dedup (tool)
    print("STEP 3: Deduplicating topics (LangChain tool)...")
    topics_deduped = dedupe_topics.invoke({"topics": topics})
    topics_deduped_path = Path(f"topics_{topic_slug}_deduped.json")
    with topics_deduped_path.open("w", encoding="utf-8") as f:
        json.dump(topics_deduped, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Topics deduped. Saved to {topics_deduped_path}")
    print(f"  Total deduped topics: {len(topics_deduped)}\n")

    # 4) Generate queries (tool, async under the hood) - ADAPTIVE TOPIC SELECTION
    print("STEP 4: Generating candidate queries (LangChain tool)...")

    # Compute adaptive distribution
    distribution = compute_adaptive_topic_distribution(
        num_final_queries=num_final_queries,
        query_types=query_types,
        strategy=OVER_GENERATION_STRATEGY,
        global_factor=GLOBAL_OVER_GENERATION_FACTOR,
        category_factors=CATEGORY_OVER_GENERATION,
    )

    # Display distribution
    print(f"\n  Over-generation strategy: {OVER_GENERATION_STRATEGY}")
    print("  Per-category targets:")
    for qtype, info in distribution.items():
        print(
            f"    {qtype:30s}: {info['target']:3d} final → "
            f"{info['candidates']:3d} candidates ({info['over_gen_factor']:.1f}x)"
        )
    print()

    # Dynamically calculate optimal number of topics
    optimal_topics = calculate_optimal_topics(
        distribution=distribution,
        num_deduped_topics=len(topics_deduped),
        queries_per_topic_per_category=QUERIES_PER_TOPIC_PER_CATEGORY,
        num_categories=len(query_types),
    )

    rich_topics = select_rich_topics(
        topics=topics_deduped,
        docs_dir=corpus_topic_dir,
        num_topics=optimal_topics,
    )

    total_candidates = sum(info["candidates"] for info in distribution.values())
    overgen_avg = total_candidates / num_final_queries if num_final_queries > 0 else 0

    print(
        f"  → Using {len(rich_topics)} topics (out of {len(topics_deduped)}) "
        f"[dynamically selected for {num_final_queries} final queries]."
    )
    print(f"  → Total candidate queries: {total_candidates} ({overgen_avg:.1f}x average over-generation)")

    candidate_rows = generate_queries.invoke(
        {
            "topics": rich_topics,
            "docs_root_dir": str(corpus_topic_dir),
            "num_queries_per_type": QUERIES_PER_TOPIC_PER_CATEGORY,
            "hf_model": HF_GENERATION_MODEL,
        }
    )
    print(f"  ✓ Generated {len(candidate_rows)} raw candidate queries.\n")

    if not candidate_rows:
        print("No candidate queries generated; exiting.")
        return

    # 5) Score queries with heuristic metrics
    print("STEP 5: Scoring queries with heuristic metrics (LangChain tool)...")
    scored_all: List[Dict[str, Any]] = []

    for qtype in query_types:
        rows_for_type = [r for r in candidate_rows if r.get("query_type") == qtype]
        if not rows_for_type:
            continue

        print(f"  Scoring {len(rows_for_type)} queries of type '{qtype}'...")
        scored = heuristic_score_queries.invoke(
            {
                "rows": rows_for_type,
                "category": qtype,
                "topics_json_path": str(topics_json_path),
                "docs_dir": str(corpus_topic_dir),
                # You can override defaults if you like:
                # "semantic_model_name": "...",
                # "semantic_max_sim": 0.85,
                # "lexical_min_dist": 0.3,
            }
        )
        scored_all.extend(scored)
    
    print(f"  ✓ Scoring done. Total scored rows: {len(scored_all)}\n")

    if not scored_all:
        print("✗ No scored rows available, exiting.")
        return

    # 6) Adaptive percentile-based rejection sampling (plain Python)
    print("STEP 6: Selecting queries with dynamic percentile thresholding...")

    # Extract final targets from distribution
    target_per_type = {qtype: info["target"] for qtype, info in distribution.items()}
    print(f"  Target per type: {target_per_type}")

    sampler = RejectionSampler(
        min_per_topic_per_type=1,
        enforce_dedup=True,
        random_seed=42,
        initial_percentile=0.90,
        percentile_step=0.05,
        min_percentile=0.50,
    )

    final_rows = sampler.run(
        scored_rows=scored_all,
        target_per_type=target_per_type,
    )

    print(f"\n  ✓ Selected {len(final_rows)} queries total.\n")

    # Save final queries
    out_dir = Path("outputs") / topic_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = out_dir / f"final_queries_{topic_slug}.jsonl"
    with final_json_path.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print top 10 queries by score
    print("\nTop 10 queries by score:")
    top10 = sorted(
        [r for r in final_rows if r.get("total_score") is not None],
        key=lambda r: r["total_score"],
        reverse=True,
    )[:10]
    for i, r in enumerate(top10, 1):
        score = r.get("total_score", 0.0)
        q = r.get("query", "")
        qtype = r.get("query_type", "")
        print(f"  {i:2d}. [{score:.3f}] [{qtype:20s}] {q}")

    print(f"\n{'=' * 60}")
    print("✓ Pipeline complete!")
    print(f"{'=' * 60}")
    print(f"Final queries written to: {final_json_path}")
    print(f"Total queries: {len(final_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        raise
