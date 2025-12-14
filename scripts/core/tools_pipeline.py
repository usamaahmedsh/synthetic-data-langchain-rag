# tools_pipeline.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

# LangChain core
from langchain_core.tools import tool

# Your modules
from scripts.core.corpus_builder import WikipediaClient, CorpusBuilder, slugify, DEFAULT_MAX_PAGES  # type: ignore
from scripts.core.topic_modeler import BuildTopics  # type: ignore
from backups.topics_deduper import BuildTopicsDeduper  # type: ignore
from scripts.core.query_generator import BuildQueries  # new HF-based version
from backups.judge_mlx_client import (
    JudgeQueriesMLX,
    load_topics,
    precompute_topic_cache,
)  # LangChain+ChatLlamaCpp version


from sentence_transformers import SentenceTransformer  # or your existing encoder
import numpy as np
from pathlib import Path

from scripts.core.corpus_builder import load_topics, precompute_topic_cache  # adjust imports
from rank_bm25 import BM25Okapi
import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())

def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / (len(A | B) + 1e-6)

def build_bm25_for_topic_docs(docs_dir: Path) -> tuple[BM25Okapi, list[str]]:
    docs: list[str] = []
    for p in docs_dir.glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if txt.strip():
            docs.append(txt)
    if not docs:
        docs = [""]
    tokenized = [tokenize(d) for d in docs]
    return BM25Okapi(tokenized), docs


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



@tool("heuristic_score_queries", return_direct=True)
def heuristic_score_queries(
    rows: list[dict[str, object]],
    category: str,
    topics_json_path: str,
    docs_dir: str,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_max_sim: float = 0.85,
    lexical_min_dist: float = 0.3,
) -> list[dict[str, object]]:
    """
    Score queries with simple heuristics + diversity (no LLM).

    Inputs:
      - rows: list of query dicts from generate_queries
      - category: taxonomy label (informational, exploratory, ...)
      - topics_json_path: path to topics.json for the corpus
      - docs_dir: directory with topic-specific documents

    Output:
      - rows with added fields:
          total_score, quality_score, lex_div_score, sem_div_score
    """
    if not rows:
        return []

    topics_by_id = load_topics(Path(topics_json_path))
    docs_dir_path = Path(docs_dir)

    # Pre-build BM25 per topic_id (coarse, doc-level)
    bm25_by_topic: dict[int, BM25Okapi] = {}
    for tid in {int(r["topic_id"]) for r in rows if "topic_id" in r}:
        t_docs_dir = docs_dir_path  # single dir per run in your pipeline
        bm25, _ = build_bm25_for_topic_docs(t_docs_dir)
        bm25_by_topic[tid] = bm25

    # Prepare embeddings for all queries
    model = SentenceTransformer(semantic_model_name)
    queries = [str(r["query"]) for r in rows]
    embs = model.encode(queries, normalize_embeddings=True)
    embs = np.asarray(embs)

    # Tokenize all queries once
    tokens_all = [tokenize(q) for q in queries]

    # Group rows by (topic_id, query_type)
    grouped: dict[tuple[int, str], list[int]] = {}
    for idx, r in enumerate(rows):
        try:
            tid = int(r["topic_id"])
        except Exception:
            continue
        qtype = str(r.get("query_type", ""))
        grouped.setdefault((tid, qtype), []).append(idx)

    # Helper: simple quality score per query
    def quality_score(idx: int, topic_id: int) -> float:
        q_tokens = tokens_all[idx]
        n = len(q_tokens)
        if n < 2:
            return 0.0
        # length prior ~ Gaussian around 6 tokens
        s_len = float(np.exp(-((n - 6) ** 2) / 18.0))

        # BM25 relevance (normalized per topic bucket later if needed)
        bm25 = bm25_by_topic.get(topic_id)
        if bm25 is None:
            s_bm25 = 0.0
        else:
            s = bm25.get_scores(q_tokens)
            s_bm25 = float(np.max(s)) if len(s) > 0 else 0.0

        # We can just log BM25 raw; for now, simple combination
        return 0.6 * s_bm25 + 0.4 * s_len

    # Compute scores per group with diversity
    for (tid, qtype), idxs in grouped.items():
        if not idxs:
            continue

        # Precompute raw quality scores in this group
        q_raw = {i: quality_score(i, tid) for i in idxs}
        # Normalize quality within group for stability
        vals = np.array(list(q_raw.values()), dtype=float)
        q_min, q_max = float(vals.min()), float(vals.max())
        for i in idxs:
            if q_max > q_min:
                rows[i]["quality_score"] = (q_raw[i] - q_min) / (q_max - q_min)
            else:
                rows[i]["quality_score"] = 0.0

        # Greedy selection for diversity-aware scores (we still score all rows)
        selected: list[int] = []

        # Process in descending quality order
        for i in sorted(idxs, key=lambda j: rows[j]["quality_score"], reverse=True):
            e_i = embs[i]
            t_i = tokens_all[i]

            if not selected:
                # First one is both lex/sem-diverse by definition
                rows[i]["lex_div_score"] = 1.0
                rows[i]["sem_div_score"] = 1.0
                # total_score = quality only for now
                rows[i]["total_score"] = rows[i]["quality_score"]
                selected.append(i)
                continue

            # Compute lexical diversity vs selected
            lex_sims = [jaccard(t_i, tokens_all[j]) for j in selected]
            max_lex_sim = max(lex_sims)
            d_lex_min = 1.0 - max_lex_sim

            # Compute semantic similarity vs selected
            sem_sims = [float(np.dot(e_i, embs[j])) for j in selected]
            max_sem_sim = max(sem_sims)
            d_sem = 1.0 - max_sem_sim

            rows[i]["lex_div_score"] = d_lex_min
            rows[i]["sem_div_score"] = d_sem

            # Diversity-aware score (but do not hard-reject here; let sampler decide)
            # You can weight these however you like:
            total = (
                0.6 * rows[i]["quality_score"] +
                0.2 * d_lex_min +
                0.2 * d_sem
            )
            rows[i]["total_score"] = total

    return rows

