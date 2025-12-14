#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Main orchestrator for the LangChain-based query-generation pipeline.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import traceback
import torch

# Import LangChain tools
from tools.langchain_tools import (
    fetch_wikipedia_corpus,
    build_topics,
    dedupe_topics,
    generate_queries,
    heuristic_score_queries,
)

# Import core components
from core.query_filter import AdvancedQueryFilter, deduplicate_near_duplicates
from core.topic_scorer import TopicQualityScorer

# Import postprocessing
from postprocessing.sampling import RejectionSampler, ensure_semantic_diversity
from postprocessing.deduplication import QueryDeduplicator

# Import utilities
from utils.output_manager import OutputManager, create_outputs_readme
from utils.profiler import SimpleProfiler
from utils.checkpoint import PipelineCheckpoint
from utils.metrics import PipelineMetrics
from utils.bm25_utils import build_bm25_with_passages

# Import configuration
from config.settings import (
    DEFAULT_MAX_PAGES,
    QUERIES_PER_TOPIC_PER_CATEGORY,
    HF_GENERATION_MODEL,
    JUDGE_MODEL_PATH,
    OVER_GENERATION_STRATEGY,
    GLOBAL_OVER_GENERATION_FACTOR,
    CATEGORY_OVER_GENERATION,
    DEFAULT_TAXONOMY,
    DATA_DIR,
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
    EMBEDDING_BATCH_SIZE,
    MAX_PARALLEL_TOPICS,
    MAX_PARALLEL_CATEGORIES,
    USE_GPU,
    IS_APPLE_SILICON,
)

# Safely import load_dotenv
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()


# ==========================================
# Configuration Flags
# ==========================================

USE_CHECKPOINTS = True
USE_VALIDATION = True
USE_ADVANCED_FILTERING = True
USE_GLOBAL_DEDUP = True
USE_SEMANTIC_DIVERSITY = True
USE_TOPIC_QUALITY_SCORING = True
USE_PASSAGE_LEVEL_BM25 = True


# ==========================================
# Helper Functions
# ==========================================

def ask_topic() -> str:
    topic = input("Enter topic/person for corpus (e.g. 'Albert Einstein'): ").strip()
    while not topic:
        topic = input("Topic cannot be empty. Enter topic/person: ").strip()
    return topic


def ask_num_final_queries() -> int:
    raw = input("Enter desired APPROXIMATE number of final queries (e.g. 30): ").strip()
    while not raw.isdigit() or int(raw) <= 0:
        raw = input("Please enter a positive integer (e.g. 30): ").strip()
    return int(raw)


def compute_per_type_targets(total: int, types: List[str]) -> Dict[str, int]:
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
    use_quality_scoring: bool = True,
) -> List[Dict[str, Any]]:
    if not topics:
        return []

    if use_quality_scoring:
        print("  Using advanced topic quality scoring...")
        scorer = TopicQualityScorer()
        ranked_topics = scorer.rank_topics(topics, docs_dir, top_k=num_topics)
        print(f"  ✓ Selected {len(ranked_topics)} topics by quality score")
        return ranked_topics
    else:
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
    if num_deduped_topics == 0:
        return 0
    queries_per_topic = queries_per_topic_per_category * num_categories
    total_candidates = sum(info["candidates"] for info in distribution.values())
    topics_needed = max(1, (total_candidates + queries_per_topic - 1) // queries_per_topic)
    optimal = min(topics_needed, num_deduped_topics)
    return optimal


# ==========================================
# Main Pipeline
# ==========================================

def _run_pipeline(topic_name: str, num_final_queries: int) -> None:
    """Run the full pipeline for a single topic and target query count."""
    print("=" * 70)
    print("Query Generation Pipeline")
    print("=" * 70)
    print()

    profiler = SimpleProfiler()
    metrics = PipelineMetrics()

    # Detect hardware
    if USE_GPU:
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU Detected: {gpu_name}")
        elif IS_APPLE_SILICON and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print(f"✓ Apple Silicon GPU (MPS) Detected")
        else:
            device = "cpu"
            print(f"ℹ Using CPU (no GPU detected)")
    else:
        device = "cpu"
        print(f"ℹ GPU disabled, using CPU")

    query_types = DEFAULT_TAXONOMY

    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"  Topic name:              {topic_name}")
    print(f"  Final query target:      {num_final_queries}")
    print(f"  Max Wikipedia pages:     {DEFAULT_MAX_PAGES}")
    print(f"  Taxonomy categories:     {len(query_types)} ({', '.join(query_types)})")
    print(f"  Queries per topic/cat:   {QUERIES_PER_TOPIC_PER_CATEGORY}")
    print(f"  Generation model:        {HF_GENERATION_MODEL}")
    print(f"  Over-generation:         {OVER_GENERATION_STRATEGY}")
    print(f"  Device:                  {device}")
    print(f"  Batch size:              {EMBEDDING_BATCH_SIZE}")
    print(f"  Max parallel topics:     {MAX_PARALLEL_TOPICS}")
    print()
    print(f"  Quality Features:")
    print(f"    - Advanced Filtering:    {USE_ADVANCED_FILTERING}")
    print(f"    - Passage-level BM25:    {USE_PASSAGE_LEVEL_BM25}")
    print(f"    - Topic Quality Score:   {USE_TOPIC_QUALITY_SCORING}")
    print(f"    - Semantic Diversity:    {USE_SEMANTIC_DIVERSITY}")
    print(f"    - Global Dedup:          {USE_GLOBAL_DEDUP}")
    print(f"    - Checkpoints:           {USE_CHECKPOINTS}")
    print("=" * 70 + "\n")

    base_data_dir = Path(DATA_DIR)
    base_artifacts_dir = Path(ARTIFACTS_DIR)
    base_data_dir.mkdir(parents=True, exist_ok=True)
    base_artifacts_dir.mkdir(parents=True, exist_ok=True)

    create_outputs_readme(Path(OUTPUTS_DIR))

    # STEP 1: Corpus
    print("\n" + "=" * 70)
    print("STEP 1: Building Wikipedia Corpus")
    print("=" * 70)
    try:
        with profiler.profile("corpus_building"):
            corpus_info = fetch_wikipedia_corpus.invoke(
                {
                    "seed_title": topic_name,
                    "max_pages": DEFAULT_MAX_PAGES,
                    "cat_depth": 2,
                    "output_dir": str(base_data_dir),
                }
            )
    except Exception as e:
        print(f"✗ Corpus step failed for topic '{topic_name}': {e}")
        traceback.print_exc()
        return

    topic_slug = corpus_info["topic_slug"]
    topic_name_resolved = corpus_info["topic_name"]
    corpus_topic_dir = Path(corpus_info["output_dir"]) / topic_slug / "pages"
    print(f"  ✓ Corpus ready at {corpus_topic_dir}\n")

    output_mgr = OutputManager(
        topic_slug=topic_slug,
        base_output_dir=Path(OUTPUTS_DIR),
        use_timestamp=True,
        create_symlink=True,
    )
    print(f"  ✓ Output directory: {output_mgr.run_dir}\n")

    if USE_CHECKPOINTS:
        checkpoint = PipelineCheckpoint(Path("checkpoints") / topic_slug)
    else:
        checkpoint = None

    metrics.record_stage("corpus", {
        "pages_directory": str(corpus_topic_dir),
        "topic_name": topic_name_resolved,
        "topic_slug": topic_slug,
    })

    # STEP 2–3.5: Topics, dedup, selection
    try:
        print("\n" + "=" * 70)
        print("STEP 2: Building Topics with BERTopic")
        print("=" * 70)

        with profiler.profile("topic_modeling"):
            if checkpoint and checkpoint.has_step("topics"):
                print("  ⚡ Loading topics from checkpoint...")
                topics_result = checkpoint.load_step("topics")
            else:
                topics_result = build_topics.invoke(
                    {
                        "pages_dir": str(corpus_topic_dir),
                        "topic_name": topic_name_resolved,
                        "topic_slug": topic_slug,
                        "artifacts_root": str(base_artifacts_dir),
                    }
                )
                if checkpoint:
                    checkpoint.save_step("topics", topics_result)

        topics = topics_result["topics"]
        output_mgr.save_json(topics, "topics.json")
        print(f"  ✓ Topics built and saved ({len(topics)} topics).\n")

        metrics.record_stage("topics", {"discovered": len(topics)})

        print("\n" + "=" * 70)
        print("STEP 3: Deduplicating Topics")
        print("=" * 70)

        with profiler.profile("topic_deduplication"):
            if checkpoint and checkpoint.has_step("topics_deduped"):
                print("  ⚡ Loading deduped topics from checkpoint...")
                topics_deduped = checkpoint.load_step("topics_deduped")
            else:
                topics_deduped = dedupe_topics.invoke({"topics": topics})
                if checkpoint:
                    checkpoint.save_step("topics_deduped", topics_deduped)

        output_mgr.save_json(topics_deduped, "topics_deduped.json")
        print(f"  ✓ Topics deduped. Total: {len(topics_deduped)}\n")

        metrics.record_stage("topics_dedup", {
            "before": len(topics),
            "after": len(topics_deduped),
        })

        print("\n" + "=" * 70)
        print("STEP 3.5: Topic Selection")
        print("=" * 70)

        distribution = compute_adaptive_topic_distribution(
            num_final_queries=num_final_queries,
            query_types=query_types,
            strategy=OVER_GENERATION_STRATEGY,
            global_factor=GLOBAL_OVER_GENERATION_FACTOR,
            category_factors=CATEGORY_OVER_GENERATION,
        )

        print(f"\n  Over-generation strategy: {OVER_GENERATION_STRATEGY}")
        print("  Per-category targets:")
        for qtype, info in distribution.items():
            print(
                f"    {qtype:30s}: {info['target']:3d} final → "
                f"{info['candidates']:3d} candidates ({info['over_gen_factor']:.1f}x)"
            )
        print()

        optimal_topics = calculate_optimal_topics(
            distribution=distribution,
            num_deduped_topics=len(topics_deduped),
            queries_per_topic_per_category=QUERIES_PER_TOPIC_PER_CATEGORY,
            num_categories=len(query_types),
        )

        with profiler.profile("topic_selection"):
            rich_topics = select_rich_topics(
                topics=topics_deduped,
                docs_dir=corpus_topic_dir,
                num_topics=optimal_topics,
                use_quality_scoring=USE_TOPIC_QUALITY_SCORING,
            )

        total_candidates = sum(info["candidates"] for info in distribution.values())
        overgen_avg = total_candidates / num_final_queries if num_final_queries > 0 else 0

        print(
            f"  → Using {len(rich_topics)} topics (out of {len(topics_deduped)}) "
            f"[optimized for {num_final_queries} final queries]"
        )
        print(f"  → Total candidate queries: {total_candidates} ({overgen_avg:.1f}x average over-generation)\n")
    except Exception as e:
        print(f"✗ Topic modeling/selection failed for topic '{topic_name}': {e}")
        traceback.print_exc()
        return

    # STEP 4–4.5: Query generation + advanced filtering
    try:
        print("\n" + "=" * 70)
        print("STEP 4: Generating Candidate Queries (Parallel)")
        print("=" * 70)

        with profiler.profile("query_generation"):
            if checkpoint and checkpoint.has_step("candidate_rows"):
                print("  ⚡ Loading candidate queries from checkpoint...")
                candidate_rows = checkpoint.load_step("candidate_rows")
            else:
                candidate_rows = generate_queries.invoke(
                    {
                        "topics": rich_topics,
                        "docs_root_dir": str(corpus_topic_dir),
                        "num_queries_per_type": QUERIES_PER_TOPIC_PER_CATEGORY,
                        "hf_model": HF_GENERATION_MODEL,
                    }
                )
                if checkpoint:
                    checkpoint.save_step("candidate_rows", candidate_rows)

        print(f"  ✓ Generated {len(candidate_rows)} raw candidate queries.\n")

        if not candidate_rows:
            print("✗ No candidate queries generated; exiting for this topic.")
            return

        metrics.record_stage("query_generation", {
            "topics_used": len(rich_topics),
            "candidates_generated": len(candidate_rows),
        })

        print("\n" + "=" * 70)
        print("STEP 4.5: Advanced Query Filtering")
        print("=" * 70)

        if USE_ADVANCED_FILTERING:
            adv_filter = AdvancedQueryFilter(
                min_length=5,
                max_length=150,
                min_words=2,
                max_words=20,
                max_repetition_ratio=0.5,
                min_entropy=1.5,
            )

            filtered_rows = []
            rejection_counts: Dict[str, int] = {}

            for row in candidate_rows:
                q = str(row.get("query", "")).strip()
                ok, reason = adv_filter.filter_query(q)
                if ok:
                    filtered_rows.append(row)
                else:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            print(f"  Advanced filtering: {len(filtered_rows)} passed, {len(candidate_rows) - len(filtered_rows)} rejected")
            if rejection_counts:
                print("  Rejection breakdown:")
                for reason, cnt in rejection_counts.items():
                    print(f"    - {reason}: {cnt}")

            candidate_rows = filtered_rows
        else:
            print(f"  ℹ Advanced filtering disabled - keeping all {len(candidate_rows)} queries")

        # Near-duplicate removal within candidate set
        texts = [r["query"] for r in candidate_rows]
        deduped_texts = deduplicate_near_duplicates(texts)
        # Map back: keep only rows whose query is in deduped_texts (first occurrence wins)
        seen = set()
        deduped_rows = []
        for r in candidate_rows:
            q = r["query"]
            if q in deduped_texts and q not in seen:
                deduped_rows.append(r)
                seen.add(q)
        candidate_rows = deduped_rows

        print(f"  ✓ After near-duplicate removal: {len(candidate_rows)} candidates")

        if not candidate_rows:
            print("✗ No candidates left after filtering; exiting for this topic.")
            return

        if USE_CHECKPOINTS:
            checkpoint_path = output_mgr.run_dir / "candidate_queries.jsonl"
            with open(checkpoint_path, 'w') as f:
                for row in candidate_rows:
                    f.write(json.dumps(row) + '\n')
            print(f"  ✓ Saved: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Query generation/filtering failed for topic '{topic_name}': {e}")
        traceback.print_exc()
        return

    # STEP 5–7.6: BM25, scoring, sampling, dedup, diversity
    try:
        # STEP 5: BM25
        if USE_PASSAGE_LEVEL_BM25:
            print("\n" + "=" * 70)
            print("STEP 5: Building Passage-Level BM25 Index")
            print("=" * 70)

            with profiler.profile("bm25_indexing"):
                print("  Building BM25 index over passages...")
                bm25_index, passage_metadata = build_bm25_with_passages(
                    docs_dir=corpus_topic_dir,
                    passage_size=200,
                    overlap=50,
                )

                if bm25_index:
                    print(f"  ✓ BM25 index built with {len(passage_metadata)} passages\n")
                    bm25_indices = {0: (bm25_index, passage_metadata)}
                else:
                    print("  ⚠ Failed to build BM25 index, will use default\n")
                    bm25_indices = None
        else:
            bm25_indices = None

        # STEP 6: Scoring
        print("\n" + "=" * 70)
        print("STEP 6: Scoring Queries (Heuristic + Diversity)")
        print("=" * 70)

        scored_all: List[Dict[str, Any]] = []
        topics_json_for_scoring = output_mgr.get_path("topics.json")

        with profiler.profile("query_scoring"):
            for qtype in query_types:
                rows_for_type = [r for r in candidate_rows if r.get("query_type") == qtype]
                if not rows_for_type:
                    continue

                print(f"\n  Scoring {len(rows_for_type)} queries of type '{qtype}'...")
                scored = heuristic_score_queries.invoke(
                    {
                        "rows": rows_for_type,
                        "category": qtype,
                        "topics_json_path": str(topics_json_for_scoring),
                        "docs_dir": str(corpus_topic_dir),
                        "bm25_indices": bm25_indices,
                    }
                )
                scored_all.extend(scored)

        print(f"\n  ✓ Scoring done. Total scored rows: {len(scored_all)}\n")
        output_mgr.save_jsonl(scored_all, "scored_queries.jsonl")

        if not scored_all:
            print("✗ No scored rows available; exiting for this topic.")
            return

        metrics.record_stage("scoring", {"scored": len(scored_all)})

        # STEP 7: Rejection sampling
        print("\n" + "=" * 70)
        print("STEP 7: Selecting Queries (Dynamic Percentile Thresholding)")
        print("=" * 70)

        target_per_type = {qtype: info["target"] for qtype, info in distribution.items()}
        print(f"  Target per type: {target_per_type}\n")

        with profiler.profile("rejection_sampling"):
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

        # STEP 7.5: Global dedup
        if USE_GLOBAL_DEDUP:
            print("\n" + "=" * 70)
            print("STEP 7.5: Global Deduplication")
            print("=" * 70)

            with profiler.profile("global_deduplication"):
                deduplicator = QueryDeduplicator(similarity_threshold=0.90)
                final_rows = deduplicator.deduplicate(final_rows, keep_highest_score=True)

            print(f"  ✓ After global dedup: {len(final_rows)} unique queries\n")

        # STEP 7.6: Semantic diversity
        if USE_SEMANTIC_DIVERSITY and len(final_rows) > num_final_queries:
            print("\n" + "=" * 70)
            print("STEP 7.6: Enforcing Semantic Diversity")
            print("=" * 70)

            with profiler.profile("semantic_diversity"):
                from sentence_transformers import SentenceTransformer

                print("  Loading embedding model for diversity check...")
                diversity_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

                print(f"  Selecting {num_final_queries} diverse queries from {len(final_rows)}...")
                final_rows = ensure_semantic_diversity(
                    queries=final_rows,
                    target_count=num_final_queries,
                    model=diversity_model,
                    diversity_weight=0.3,
                )

            print(f"  ✓ Semantic diversity enforced: {len(final_rows)} queries selected\n")

        output_mgr.save_jsonl(final_rows, "final_queries.jsonl")

        # Finalization
        print("\n" + "=" * 70)
        print("Finalizing Results")
        print("=" * 70)

        scores = [r.get("total_score", 0) for r in final_rows if "total_score" in r]

        final_metrics = {
            "run_timestamp": metrics.start_time.isoformat(),
            "topic_name": topic_name_resolved,
            "topic_slug": topic_slug,
            "device": device,
            "configuration": {
                "max_pages": DEFAULT_MAX_PAGES,
                "queries_per_topic_per_category": QUERIES_PER_TOPIC_PER_CATEGORY,
                "over_generation_strategy": OVER_GENERATION_STRATEGY,
                "global_factor": GLOBAL_OVER_GENERATION_FACTOR,
                "category_factors": CATEGORY_OVER_GENERATION,
                "hf_model": HF_GENERATION_MODEL,
                "device": device,
                "batch_size": EMBEDDING_BATCH_SIZE,
                "advanced_filtering": USE_ADVANCED_FILTERING,
                "passage_bm25": USE_PASSAGE_LEVEL_BM25,
                "topic_quality_scoring": USE_TOPIC_QUALITY_SCORING,
                "semantic_diversity": USE_SEMANTIC_DIVERSITY,
                "global_dedup": USE_GLOBAL_DEDUP,
            },
            "corpus": {"pages_directory": str(corpus_topic_dir)},
            "topics": {
                "discovered": len(topics),
                "after_dedup": len(topics_deduped),
                "used_for_generation": len(rich_topics),
            },
            "queries": {
                "candidates_generated": metrics.metrics["stages"]
                    .get("query_generation", {})
                    .get("candidates_generated", 0),
                "after_filtering": len(candidate_rows),
                "candidates_scored": len(scored_all),
                "final_selected": len(final_rows),
                "target": num_final_queries,
                "over_generation_factor": overgen_avg,
            },
            "scores": metrics.get_quality_distribution(scores),
            "distribution": {
                qtype: {
                    "target": info["target"],
                    "actual": sum(1 for r in final_rows if r.get("query_type") == qtype),
                    "over_gen_factor": info["over_gen_factor"],
                }
                for qtype, info in distribution.items()
            },
            "timing": profiler.timings,
        }

        output_mgr.save_metrics(final_metrics)

        config = {
            "max_wiki_pages": DEFAULT_MAX_PAGES,
            "queries_per_topic_per_category": QUERIES_PER_TOPIC_PER_CATEGORY,
            "hf_generation_model": HF_GENERATION_MODEL,
            "judge_model_path": JUDGE_MODEL_PATH,
            "over_generation_strategy": OVER_GENERATION_STRATEGY,
            "global_over_generation_factor": GLOBAL_OVER_GENERATION_FACTOR,
            "category_over_generation": CATEGORY_OVER_GENERATION,
            "taxonomy": query_types,
            "device": device,
            "features": {
                "advanced_filtering": USE_ADVANCED_FILTERING,
                "passage_level_bm25": USE_PASSAGE_LEVEL_BM25,
                "topic_quality_scoring": USE_TOPIC_QUALITY_SCORING,
                "semantic_diversity": USE_SEMANTIC_DIVERSITY,
                "global_dedup": USE_GLOBAL_DEDUP,
                "checkpoints": USE_CHECKPOINTS,
            },
        }
        output_mgr.save_config(config)

        print("\n" + "=" * 70)
        print("Top 10 Queries by Score")
        print("=" * 70)
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

        profiler.print_report()
        print(output_mgr.get_summary())

        duration = sum(profiler.timings.values())
        print("\n" + "=" * 70)
        print("✅ Pipeline Complete!")
        print("=" * 70)
        print(f"  Run directory:    {output_mgr.run_dir}")
        print(f"  Duration:         {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Final queries:    {len(final_rows)}")
        print(f"  Quality (mean):   {final_metrics['scores'].get('mean', 0):.3f}")
        print(f"  Quality (median): {final_metrics['scores'].get('median', 0):.3f}")
        print(f"  Device used:      {device}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"✗ Post-processing (BM25/scoring/sampling) failed for topic '{topic_name}': {e}")
        traceback.print_exc()
        return


def main() -> None:
    """Interactive entrypoint for single-topic runs."""
    topic_name = ask_topic()
    num_final_queries = ask_num_final_queries()
    _run_pipeline(topic_name, num_final_queries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        traceback.print_exc()
        raise
