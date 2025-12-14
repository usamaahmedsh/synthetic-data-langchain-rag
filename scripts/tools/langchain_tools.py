"""LangChain tool wrappers for the query generation pipeline."""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.tools import tool

# Import core business logic
from core.corpus_builder import WikipediaClient, CorpusBuilder
from core.topic_modeler import BuildTopics
from core.query_generator import BuildQueries
from core.query_scorer import HeuristicQueryScorer, LLMQueryScorer

# Import postprocessing
from postprocessing.deduplication import BuildTopicsDeduper

# Import utilities
from utils.text_utils import slugify
from utils.data_utils import load_topics, precompute_topic_cache

# Import settings
from config.settings import (
    DEFAULT_MAX_PAGES,
    DEFAULT_CAT_DEPTH,
    DATA_DIR,
)


# =========================
# Wikipedia / Corpus Tools
# =========================

@tool("fetch_wikipedia_corpus", return_direct=True)
def fetch_wikipedia_corpus(
    seed_title: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    cat_depth: int = DEFAULT_CAT_DEPTH,
    output_dir: str = str(DATA_DIR),
) -> Dict[str, Any]:
    """
    Build a Wikipedia corpus for a given seed title.
    
    Args:
        seed_title: Topic/person to fetch corpus for
        max_pages: Maximum pages to fetch
        cat_depth: Category traversal depth
        output_dir: Root directory for output
        
    Returns:
        Dict with topic_slug, topic_name, output_dir, manifest_path
    """
    async def _run() -> Dict[str, Any]:
        topic_name = seed_title.strip()
        topic_slug = slugify(topic_name)
        out_dir = Path(output_dir)
        pages_dir = out_dir / topic_slug / "pages"
        manifest_path = out_dir / topic_slug / "manifest.jsonl"

        # Short-circuit if corpus already exists
        if pages_dir.exists() and manifest_path.exists():
            print(f"  ✓ Corpus already exists at {pages_dir}, skipping fetch.")
            return {
                "topic_slug": topic_slug,
                "topic_name": topic_name,
                "output_dir": str(out_dir),
                "manifest_path": str(manifest_path),
            }

        async with WikipediaClient() as client:
            builder = CorpusBuilder(client=client, output_dir=output_dir)
            titles = await builder.discover_titles(
                seed_title=topic_name,
                cat_depth=cat_depth,
                max_pages=max_pages,
            )

            await builder.download_titles(
                topic_slug=topic_slug,
                topic_name=topic_name,
                titles=titles,
            )

        return {
            "topic_slug": topic_slug,
            "topic_name": topic_name,
            "output_dir": str(out_dir),
            "manifest_path": str(manifest_path),
        }

    return asyncio.run(_run())


# =========================
# Topic Modeling Tools
# =========================

@tool("build_topics", return_direct=True)
def build_topics(
    pages_dir: str,
    topic_name: str,
    topic_slug: str,
    artifacts_root: str = "artifacts",
) -> Dict[str, Any]:
    """
    Run BERTopic over .txt files in pages_dir.
    
    Args:
        pages_dir: Directory containing .txt files
        topic_name: Human-readable topic name
        topic_slug: Slugified topic name
        artifacts_root: Root directory for artifacts
        
    Returns:
        Dict with topics, model_dir, embeddings_path, docs_path
    """
    bt = BuildTopics(artifacts_root=artifacts_root)
    res = bt.run(
        input_dir=pages_dir,
        topic_name=topic_name,
        topic_slug=topic_slug,
    )
    return res


@tool("dedupe_topics", return_direct=True)
def dedupe_topics(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate topics using overlap in top_words and relevant_documents.
    
    Args:
        topics: List of topic dicts from build_topics
        
    Returns:
        Deduplicated list of topics
    """
    deduper = BuildTopicsDeduper()
    return deduper.run(topics)


# =========================
# Query Generation Tool
# =========================

@tool("generate_queries", return_direct=True)
def generate_queries(
    topics: List[Dict[str, Any]],
    docs_root_dir: str,
    num_queries_per_type: int = 10,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> List[Dict[str, Any]]:
    """
    Generate search queries for each topic using BuildQueries.
    
    Args:
        topics: List of topic dicts (deduped)
        docs_root_dir: Base directory containing {topic_slug}/pages/*.txt
        num_queries_per_type: Target queries per taxonomy type
        hf_model: Hugging Face model repo id
        
    Returns:
        List of rows with fields: topic_id, topic_name, query_type, query, etc.
    """
    bq = BuildQueries(hf_model=hf_model)

    async def _run_all() -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        docs_dir = Path(docs_root_dir)

        # BuildQueries.run_for_topics_async handles parallel processing
        rows = await bq.run_for_topics_async(
            topics=topics,
            docs_dir=docs_dir,
            num_queries_per_type=num_queries_per_type,
        )
        all_rows.extend(rows)
        return all_rows

    return asyncio.run(_run_all())


# =========================
# Query Scoring Tools
# =========================

@tool("heuristic_score_queries", return_direct=True)
def heuristic_score_queries(
    rows: List[Dict[str, Any]],
    category: str,
    topics_json_path: str,
    docs_dir: str,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_max_sim: float = 0.85,
    lexical_min_dist: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Score queries with simple heuristics + diversity (no LLM).
    
    Args:
        rows: List of query dicts from generate_queries
        category: Taxonomy label
        topics_json_path: Path to topics.json
        docs_dir: Directory with topic documents
        semantic_model_name: SentenceTransformer model
        semantic_max_sim: Max semantic similarity
        lexical_min_dist: Min lexical distance
        
    Returns:
        Rows with added fields: total_score, quality_score, lex_div_score, sem_div_score
    """
    scorer = HeuristicQueryScorer(
        semantic_model_name=semantic_model_name,
        semantic_max_sim=semantic_max_sim,
        lexical_min_dist=lexical_min_dist,
    )

    return scorer.score_queries(
        rows=rows,
        category=category,
        topics_json_path=Path(topics_json_path),
        docs_dir=Path(docs_dir),
    )


@tool("judge_queries", return_direct=True)
def judge_queries(
    rows: List[Dict[str, Any]],
    category: str,
    topics_json_path: str,
    docs_dir: str,
    model_path: str,
    batch_size: int = 24,
    max_concurrent: int = 12,
) -> List[Dict[str, Any]]:
    """
    Score queries with LLM-as-a-judge.
    
    Args:
        rows: List of query rows to score
        category: Taxonomy category
        topics_json_path: Path to topics.json
        docs_dir: Directory with documents
        model_path: Path to local gguf judge model
        batch_size: Batch size for scoring
        max_concurrent: Max concurrent requests
        
    Returns:
        Rows with total_score and per-criterion scores
    """
    topics_by_id = load_topics(Path(topics_json_path))
    topic_ids = {int(r["topic_id"]) for r in rows if "topic_id" in r}
    topic_cache = precompute_topic_cache(topic_ids, topics_by_id, Path(docs_dir))

    judge = LLMQueryScorer(
        model_path=model_path,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
    )

    return judge.score_queries(
        rows=rows,
        category=category,
        topics_by_id=topics_by_id,
        topic_cache=topic_cache,
    )
