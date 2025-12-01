# tools_pipeline.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

# LangChain core
from langchain_core.tools import tool

# Your modules
from corpus_wikipedia import WikipediaClient, CorpusBuilder, slugify, DEFAULT_MAX_PAGES  # type: ignore
from topics_builder import BuildTopics  # type: ignore
from topics_deduper import BuildTopicsDeduper  # type: ignore
from queries_builder import BuildQueries  # new HF-based version
from judge_mlx_client import (
    JudgeQueriesMLX,
    load_topics,
    precompute_topic_cache,
)  # LangChain+ChatLlamaCpp version


# =========================
# Wikipedia / corpus tools
# =========================

@tool("fetch_wikipedia_corpus", return_direct=True)
def fetch_wikipedia_corpus(
    seed_title: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    cat_depth: int = 2,
    output_dir: str = "data/raw",
) -> Dict[str, Any]:
    """
    Build a Wikipedia corpus for a given seed title.

    Returns:
    - topic_slug: slugified title
    - topic_name: original title
    - output_dir: root directory where pages + manifest are stored
    - manifest_path: path to manifest.jsonl
    """
    async def _run() -> Dict[str, Any]:
        topic_name = seed_title.strip()
        topic_slug = slugify(topic_name)
        out_dir = Path(output_dir)
        pages_dir = out_dir / topic_slug / "pages"
        manifest_path = out_dir / topic_slug / "manifest.jsonl"

        # NEW: short-circuit if corpus already exists
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
# Topic modeling tools
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

    Returns the dictionary from BuildTopics.run:
    - topics: List[topic_dict]
    - model_dir, embeddings_path, docs_path
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

    Input: list of topic dicts (BuildTopics.run()['topics'])
    Output: new list of topic dicts.
    """
    deduper = BuildTopicsDeduper()
    return deduper.run(topics)


# =========================
# Query generation tool
# =========================

@tool("generate_queries", return_direct=True)
def generate_queries(
    topics: List[Dict[str, Any]],
    docs_root_dir: str,
    num_queries_per_type: int = 10,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> List[Dict[str, Any]]:
    """
    Generate search queries for each topic using the HF-based BuildQueries.

    Inputs:
    - topics: list of topic dicts (deduped)
    - docs_root_dir: base directory containing {topic_slug}/pages/*.txt
    - num_queries_per_type: target queries per taxonomy type
    - hf_model: Hugging Face model repo id for query generation

    Output:
    - list of rows with fields:
      topic_id, topic_name, query_type, query, model, top_words, relevant_files
    """
    bq = BuildQueries(hf_model=hf_model)

    async def _run_all() -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        docs_dir = Path(docs_root_dir)

        # BuildQueries.run_for_topics_async already does sequential topics + parallel qtypes
        rows = await bq.run_for_topics_async(
            topics=topics,
            docs_dir=docs_dir,
            num_queries_per_type=num_queries_per_type,
        )
        all_rows.extend(rows)
        return all_rows

    return asyncio.run(_run_all())


# =========================
# Judge / scoring tool
# =========================

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
    Score queries with LLM-as-a-judge for a given taxonomy category.

    Inputs:
    - rows: list of query rows (from generate_queries) to score
    - category: one of the taxonomy labels used in SPECS
    - topics_json_path: path to a topics.json file (for topic metadata)
    - docs_dir: directory containing topic-specific documents (for BM25 context)
    - model_path: path to local gguf judge model
    - batch_size, max_concurrent: performance tuning knobs

    Output:
    - rows with extra fields: total_score, per-criterion scores, and error (if any)
    """
    topics_by_id = load_topics(Path(topics_json_path))
    topic_ids = {int(r["topic_id"]) for r in rows if "topic_id" in r}
    topic_cache = precompute_topic_cache(topic_ids, topics_by_id, Path(docs_dir))

    judge = JudgeQueriesMLX(
        model_path=model_path,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
    )

    # Use sync wrapper (internally runs async)
    scored_rows = judge.score_rows(
        rows=rows,
        category=category,
        topics_by_id=topics_by_id,
        topic_cache=topic_cache,
    )
    return scored_rows
