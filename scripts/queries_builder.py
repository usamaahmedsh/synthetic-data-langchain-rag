# queries_builder.py

import os
import re
import json
import asyncio
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import requests
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# ------------------------
# Generation hyperparams
# ------------------------

TEMPERATURE = 0.6
TOP_P = 0.9
MAX_NEW_TOKENS = 80

# Context management - optimized for <1024 tokens per prompt
MAX_CONTEXT_CHARS = 2000  # ~500 tokens for context
MAX_CHUNK_SIZE = 400      # Smaller chunks for tighter prompts
CHUNK_OVERLAP = 80

# Parallel processing settings
MAX_CONCURRENT_REQUESTS = 8
BATCH_SIZE = 16

# ------------------------
# Text + retrieval helpers
# ------------------------

def safe_read(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"[[ERROR reading {path.name}: {e}]]"

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Smaller chunks for shorter prompts; overlap to preserve coherence."""
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks

def build_bm25_index(docs: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [word_tokenize(d.lower()) for d in docs]
    return BM25Okapi(tokenized), tokenized

def retrieve_top_passages(
    bm25: BM25Okapi,
    tokenized_docs: List[List[str]],
    docs: List[str],
    query: str,
    top_n: int = 3,
) -> List[str]:
    q_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(q_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [docs[i] for i in ranked_indices]

# ------------------------
# JSON helpers
# ------------------------

def extract_json_list(text: str) -> List[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        snippet = text[start : end + 1]
        snippet = re.sub(r",\s*\]", "]", snippet)
        snippet = re.sub(r"[\n\r]+", " ", snippet)
        try:
            return json.loads(snippet)
        except Exception:
            return []

def normalize_query(q: str) -> str:
    """Normalize for deduplication: lowercase, strip punctuation & extra spaces."""
    q = q.strip().strip("-•\"' ").rstrip(".?!")
    q = re.sub(r"\s+", " ", q)
    return q.lower()

# ------------------------
# Query builder container
# ------------------------

class BuildQueries:
    """
    Container for generating search-like queries per topic with parallel processing.

    Uses a local llama.cpp HTTP server (OpenAI-compatible /v1/chat/completions)
    for query generation.
    """

    DEFAULT_TAXONOMY = [
        "informational",
        "exploratory",
        "navigational",
        "comparative",
        "transactional",
        "commercial_investigation",
    ]

    def __init__(
        self,
        *,
        hf_model: str = "llama-3.1-8b-q6_k_l",
        query_types: List[str] | None = None,
        sleep_between_calls: float = 0.0,
        max_attempts_per_topic: int = 10,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """
        hf_model: Logical model name as seen by your llama.cpp server.
        Environment:
          LLAMA_CPP_URL  - base URL of local server (e.g. http://127.0.0.1:8080)
        """
        self.hf_model = hf_model
        self.query_types = query_types or self.DEFAULT_TAXONOMY
        self.sleep_between_calls = sleep_between_calls
        self.max_attempts_per_topic = max_attempts_per_topic
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size

        # Local llama.cpp HTTP endpoint
        self.base_url = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080")

    # ---- internal helpers ----

    def _type_instruction(self, qtype: str) -> str:
        """Shortened instructions to reduce token count."""
        mapping = {
            "informational": "Ask for facts, definitions, or details.",
            "exploratory": "Ask broad 'what', 'why', or 'how' questions.",
            "comparative": "Compare entities using 'vs', 'difference', 'better/worse'.",
            "transactional": "Ask with intent to buy, download, or take action.",
            "commercial_investigation": "Ask for reviews, pros/cons, or recommendations.",
            "navigational": "Ask to reach a specific site, brand, or page.",
        }
        return mapping.get(qtype, mapping["informational"])

    def _make_prompt(
        self,
        topic: Dict[str, Any],
        qtype: str,
        context: str,
        num_queries_for_type: int,
    ) -> str:
        """Optimized prompt template - targets ~800-1000 tokens total."""
        top_words = ", ".join(topic.get("top_words", [])[:8])
        topic_name = topic.get("topic_name") or f"Topic {topic['topic_id']}"
        type_instr = self._type_instruction(qtype)

        return textwrap.dedent(
            f"""
            Generate {num_queries_for_type} realistic search queries as users type them.

            Topic: {topic_name}
            Key terms: {top_words}

            Context:
            \"\"\"{context}\"\"\"

            Requirements:
            - Query type: {qtype} - {type_instr}
            - Include natural imperfections: typos, casual grammar, incomplete phrasing
            - 3-10 words each
            - Output ONLY a JSON list: ["query1", "query2", ...]
            """
        ).strip()

    def _build_topic_corpus(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
    ) -> Tuple[List[str], BM25Okapi, List[List[str]]]:
        rel_files = topic.get("relevant_documents", []) or []
        corpus_chunks: List[str] = []

        for fname in rel_files:
            fpath = docs_dir / fname
            if not fpath.exists():
                continue
            text = clean_text(safe_read(fpath))
            if not text:
                continue
            corpus_chunks.extend(chunk_text(text))

        if not corpus_chunks:
            raise ValueError(f"No relevant docs found for topic {topic.get('topic_id')}.")

        bm25, tokenized = build_bm25_index(corpus_chunks)
        return corpus_chunks, bm25, tokenized

    def _build_context(
        self,
        topic: Dict[str, Any],
        corpus_chunks: List[str],
        bm25: BM25Okapi,
        tokenized: List[List[str]],
    ) -> str:
        topic_name = topic.get("topic_name") or f"Topic {topic.get('topic_id')}"
        query_string = f"{topic_name}. keywords: {' '.join(topic.get('top_words', [])[:8])}"
        top_passages = retrieve_top_passages(
            bm25, tokenized, corpus_chunks, query_string, top_n=3
        )
        context = "\n\n".join(top_passages)[:MAX_CONTEXT_CHARS]
        return context

    # ---- llama.cpp HTTP client ----

    def _call_llama_sync(self, prompt: str) -> str:
        """
        Synchronous call to local llama.cpp /v1/chat/completions endpoint.
        Expects an OpenAI-compatible API (llama.cpp server -c).
        """
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.hf_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-style: choices[0].message.content
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"llama.cpp request failed: {e}")

    # ---- async batch processing via llama.cpp ----

    async def _generate_one_async(self, prompt: str) -> Dict[str, Any]:
        """
        Call local llama.cpp asynchronously for a single prompt (thread off sync HTTP).
        """
        try:
            text = await asyncio.to_thread(self._call_llama_sync, prompt)
            return {"success": True, "content": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_batch_async(
        self,
        prompts_with_meta: List[Tuple[str, str, Dict[str, Any]]],  # (prompt, qtype, topic)
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts in parallel with concurrency control.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _wrapped_call(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self._generate_one_async(prompt)

        tasks = []
        for prompt, qtype, topic in prompts_with_meta:
            tasks.append((_wrapped_call(prompt), qtype, topic))

        results: List[Dict[str, Any]] = []
        for coro, qtype, topic in tasks:
            result = await coro
            results.append(
                {
                    "result": result,
                    "qtype": qtype,
                    "topic": topic,
                }
            )

        return results

    # ---- public API (async versions) ----

    async def run_for_topic_async(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
        num_queries_per_type: int,
    ) -> List[Dict[str, Any]]:
        docs_dir = Path(docs_dir)
        topic_id = topic.get("topic_id")
        topic_name = topic.get("topic_name") or f"Topic {topic_id}"
        rel_files = topic.get("relevant_documents", []) or []

        try:
            corpus_chunks, bm25, tokenized = self._build_topic_corpus(topic, docs_dir)
            context = self._build_context(topic, corpus_chunks, bm25, tokenized)
        except ValueError as e:
            print(f"[WARN] {e}")
            return []

        prompts_with_meta: List[Tuple[str, str, Dict[str, Any]]] = []
        for qtype in self.query_types:
            if num_queries_per_type <= 0:
                continue
            prompt = self._make_prompt(topic, qtype, context, num_queries_per_type)
            prompts_with_meta.append((prompt, qtype, topic))

        batch_results = await self._generate_batch_async(prompts_with_meta)

        rows: List[Dict[str, Any]] = []
        seen_norm: Set[str] = set()

        for batch_item in batch_results:
            result = batch_item["result"]
            qtype = batch_item["qtype"]

            if not result["success"]:
                print(f"[WARN] Generation failed for topic {topic_id}, type {qtype}: {result.get('error')}")
                continue

            try:
                queries = extract_json_list(result["content"])
            except Exception as e:
                print(f"[WARN] JSON parse failed for topic {topic_id}, type {qtype}: {e}")
                continue

            if not isinstance(queries, list):
                continue

            if len(queries) > num_queries_per_type:
                queries = queries[:num_queries_per_type]

            for q in queries:
                if not isinstance(q, str):
                    continue
                q_clean = q.strip().strip("-•\"' ").rstrip(".?!")
                if not q_clean:
                    continue

                norm = normalize_query(q_clean)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)

                rows.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "query_type": qtype,
                        "query": q_clean,
                        "model": self.hf_model,
                        "top_words": "; ".join(topic.get("top_words", [])[:10]),
                        "relevant_files": "; ".join(rel_files),
                    }
                )

        return rows

    async def run_for_topics_async(
        self,
        topics: List[Dict[str, Any]],
        docs_dir: Path,
        num_queries_per_type: int,
    ) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        total_topics = len(topics)

        for idx, topic in enumerate(topics, 1):
            print(f"Processing topic {idx}/{total_topics}: {topic.get('topic_name', 'Unknown')}")
            try:
                rows = await self.run_for_topic_async(topic, docs_dir, num_queries_per_type)
                all_rows.extend(rows)
                print(f"  Generated {len(rows)} queries")
            except Exception as e:
                print(f"[WARN] Skipping topic {topic.get('topic_id')}: {e}")

        return all_rows

    # ---- Sync wrappers ----

    def run_for_topic(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
        num_queries_per_type: int,
    ) -> List[Dict[str, Any]]:
        return asyncio.run(self.run_for_topic_async(topic, docs_dir, num_queries_per_type))

    def run_for_topics(
        self,
        topics: List[Dict[str, Any]],
        docs_dir: Path,
        num_queries_per_type: int,
    ) -> List[Dict[str, Any]]:
        return asyncio.run(self.run_for_topics_async(topics, docs_dir, num_queries_per_type))
