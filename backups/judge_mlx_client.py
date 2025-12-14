import json
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Iterable

import pandas as pd
from rank_bm25 import BM25Okapi

# LangChain + llama.cpp (local)
from langchain_community.chat_models import ChatLlamaCpp  # requires llama-cpp-python
from langchain_core.messages import HumanMessage

# -------------------------
# Local llama.cpp judge config
# -------------------------
TEMP = 0.0
TOP_P = 1.0
MAX_TOKENS = 80
TIMEOUT = 180
DEFAULT_BATCH_SIZE = 12

# Parallel processing settings
MAX_CONCURRENT_REQUESTS = 12  # Match your llama.cpp threads
BATCH_SIZE = 24               # Process more at once since judge calls are fast

# Path to your local gguf judge model
DEFAULT_JUDGE_MODEL_PATH = "$HOME/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# -------------------------
# BM25 / context settings - optimized
# -------------------------
CHUNK_WORDS = 600
CHUNK_OVERLAP = 100
BM25_TOP_N = 5
MAX_CONTEXT_CHARS = 2000

# -------------------------
# Simple tokenizer
# -------------------------
_token_re = re.compile(r"[A-Za-z0-9_]+(?:\'[A-Za-z0-9_]+)?")
def simple_tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())

# -------------------------
# JSON extraction helper
# -------------------------

def extract_json_object(text: str) -> Optional[dict]:
    """
    Extract a JSON object from a model response.

    Strategy:
    - Prefer content between <json> and </json> tags (case-insensitive).
    - Fallback: first '{' to last '}' in the string.
    - Apply light cleanup before final parse attempt.

    Returns:
    - Parsed dict on success, or None on failure.
    """
    if not text:
        return None

    # Prefer explicit markers
    lower = text.lower()
    start_tag = lower.find("<json>")
    end_tag = lower.find("</json>")
    inner: str

    if start_tag != -1 and end_tag != -1 and end_tag > start_tag:
        inner = text[start_tag + len("<json>") : end_tag].strip()
    else:
        # Fallback: best-effort object extraction
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        inner = text[start : end + 1]

    # First parse attempt
    try:
        return json.loads(inner)
    except Exception:
        pass

    # Light cleanup and retry
    cleaned = inner.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return None

# -------------------------
# I/O + corpus helpers
# -------------------------

def load_topics(json_path: Path) -> Dict[int, Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topics = data["topics"] if isinstance(data, dict) and "topics" in data else data
    by_id: Dict[int, Dict[str, Any]] = {}
    for t in topics:
        tid = int(t.get("topic_id"))
        by_id[tid] = t
    return by_id

def safe_read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"[[ERROR reading {p.name}: {e}]]"

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text_by_words(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = text.split()
    out, i = [], 0
    step = max(1, chunk_words - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + chunk_words]))
        i += step
    return out

def build_topic_corpus(topic: Dict[str, Any], docs_dir: Path) -> List[str]:
    chunks: List[str] = []
    for fname in topic.get("relevant_documents", []):
        fpath = docs_dir / fname
        if not fpath.exists():
            continue
        raw = clean_text(safe_read(fpath))
        chunks.extend(chunk_text_by_words(raw, CHUNK_WORDS, CHUNK_OVERLAP))
    return chunks

def bm25_index(chunks: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [simple_tokenize(c) for c in chunks]
    return BM25Okapi(tokenized), tokenized

def retrieve_context(chunks: List[str], bm25: BM25Okapi, query: str) -> str:
    q_tokens = simple_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_N]
    passages = [chunks[i] for i in ranked]
    context = ("\n\n".join(passages))[:MAX_CONTEXT_CHARS]
    return context

def precompute_topic_cache(
    topic_ids: Iterable[int],
    topics_by_id: Dict[int, Dict[str, Any]],
    docs_dir: Path,
) -> Dict[int, Dict[str, Any]]:
    cache: Dict[int, Dict[str, Any]] = {}
    topic_ids_list = list(topic_ids)
    print(f"Pre-computing BM25 context for {len(topic_ids_list)} topics...")
    for tid in topic_ids_list:
        topic = topics_by_id.get(tid)
        if not topic:
            continue

        title = (
            topic.get("title_15words")
            or topic.get("topic_name")
            or f"Topic {tid}"
        )
        top_words = topic.get("top_words", [])
        corpus = build_topic_corpus(topic, docs_dir)

        if not corpus:
            reps = topic.get("representative_docs", []) or []
            fallback_ctx = "\n\n".join([r.strip() for r in reps[:3]])[:MAX_CONTEXT_CHARS]
            cache[tid] = {
                "title": title,
                "top_words": top_words,
                "context": fallback_ctx,
            }
        else:
            bm25, _tokenized = bm25_index(corpus)
            query_str = f"{title}. {' '.join(top_words[:6])}"
            ctx = retrieve_context(corpus, bm25, query_str)
            cache[tid] = {
                "title": title,
                "top_words": top_words,
                "context": ctx,
            }
    print(f"  ✓ Context cache ready for {len(cache)} topics\n")
    return cache

# -------------------------
# Rubrics for each taxonomy
# -------------------------

INFORMATIONAL_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query clearly express an intent to obtain information, clarification, or an explanation about the topic "
        "(for example facts, definitions, reasons, or mechanisms), rather than to navigate to a specific site or perform "
        "an action like buying, signing up, or downloading?"
    )},
    {"label": "Clarity", "question": (
        "Even if the wording is informal or slightly messy, is it still reasonably clear what information the user is "
        "trying to find or what question they are asking?"
    )},
    {"label": "Relevance", "question": (
        "Is the query directly relevant to the given topic and its key terms, rather than drifting into unrelated subjects "
        "or generic questions that could be about anything?"
    )},
    {"label": "Human Realism", "question": (
        "Does the query sound like something a real human might type into a search engine, including natural imperfections "
        "such as typos, missing words, or casual phrasing, while still feeling non-mechanical and non-robotic?"
    )},
    {"label": "Specificity", "question": (
        "Does the query ask for a reasonably specific piece of information (for example a particular event, concept, or "
        "aspect of the topic) instead of being so broad or vague that it would be hard to answer usefully?"
    )},
    {"label": "Diversity", "question": (
        "If you consider this query alongside other informational queries for the same topic, does it ask for something "
        "meaningfully different (for example a different angle, subtopic, or question) instead of repeating the same "
        "request with only minor wording changes?"
    )},
]

EXPLORATORY_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query clearly express an intent to explore, browse, or gain a broad understanding of the topic "
        "(for example overviews, lists, introductions, or surveys), rather than requesting a single specific fact or "
        "performing a concrete action like buying something?"
    )},
    {"label": "Breadth", "question": (
        "Is the query broad enough that a reasonable answer would cover multiple aspects, examples, or subtopics "
        "(for example causes and effects, major events, key figures), instead of focusing on one extremely narrow detail?"
    )},
    {"label": "Relevance", "question": (
        "Is the exploratory focus of the query clearly centered on the given topic and its key terms, instead of drifting "
        "into loosely related or unrelated areas?"
    )},
    {"label": "Human Realism", "question": (
        "Does the query resemble how a real user might phrase a broad or open-ended search (for example 'overview of…', "
        "'everything about…', 'guide to…'), including some natural informal language or small mistakes, rather than "
        "looking like a perfectly edited prompt?"
    )},
    {"label": "Usefulness", "question": (
        "If you imagine a normal search engine result page, would this query likely return a set of results that help "
        "someone explore the topic in depth (for example articles, guides, timelines, or summaries) rather than results "
        "that feel random or unhelpful?"
    )},
    {"label": "Diversity", "question": (
        "Compared to other exploratory queries for this topic, does this query explore a distinct angle "
        "(for example biographies, timelines, causes, consequences, debates) instead of repeating almost the same "
        "exploration pattern with slightly different words?"
    )},
]

COMPARATIVE_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query clearly express a comparison intent, such as asking about similarities, differences, pros and cons, "
        "or which of two or more entities is better, worse, more important, or more influential?"
    )},
    {"label": "Clear Comparison Target", "question": (
        "Is it reasonably clear which entities, events, or concepts are being compared (for example X vs Y, or A before vs "
        "after B), even if the wording is informal or abbreviated?"
    )},
    {"label": "Relevance", "question": (
        "Are the items being compared genuinely relevant to the given topic and key terms (for example related figures, "
        "ideologies, or events) rather than injecting unrelated comparisons just because they are famous or generic?"
    )},
    {"label": "Human Realism", "question": (
        "Does the comparison query sound like something a typical user might actually type (for example 'X vs Y who was worse', "
        "'difference between A and B'), including casual phrasing, typos, or shorthand such as 'vs'?"
    )},
    {"label": "Comparative Depth", "question": (
        "Does the query set up a comparison that could realistically be answered in a meaningful way (for example by "
        "discussing differences in role, impact, time, or ideology), instead of being so vague that the comparison has no "
        "clear dimension?"
    )},
    {"label": "Diversity", "question": (
        "Across all comparative queries for the same topic, does this query focus on a different comparison pair, time period, "
        "or aspect (for example strategy vs ideology, person vs person, country vs country) rather than repeating the same "
        "comparison over and over?"
    )},
]

NAVIGATIONAL_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query clearly show an intent to reach or locate a specific website, page, profile, document, or known "
        "resource, rather than to broadly explore information or perform a purchase or other transaction?"
    )},
    {"label": "Entity Targeting", "question": (
        "Is the entity or destination the user seems to be trying to reach (for example a person, organization, site, or "
        "document) reasonably clear from the wording of the query, even if it is a bit informal or incomplete?"
    )},
    {"label": "Relevance", "question": (
        "Is the destination or resource that the query appears to target clearly related to the given topic and key terms, "
        "rather than an unrelated site or brand that just happens to share a word?"
    )},
    {"label": "Human Realism", "question": (
        "Does the query resemble how people usually type navigational searches (for example brand names, site names, "
        "person names, or URL-like fragments), including typos or partial names, while still sounding like a real user?"
    )},
    {"label": "Conciseness", "question": (
        "Is the navigational query short and to the point (for example just the key entity or site name plus a small "
        "qualifier), rather than being unnecessarily long or verbose for a navigation task?"
    )},
    {"label": "Diversity", "question": (
        "Within the set of navigational queries for this topic, does this query target a different destination or entity, or "
        "use a distinct phrasing, instead of repeating essentially the same navigational request with only trivial "
        "wording changes?"
    )},
]

TRANSACTIONAL_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query clearly show an intent to take an immediate action such as buying, booking, downloading, subscribing, "
        "registering, watching, or accessing something, instead of merely asking for information or trying to navigate to a site?"
    )},
    {"label": "Action Verb Usage", "question": (
        "Does the query include or strongly imply an action verb or phrase (for example 'buy', 'download', 'book', 'watch', "
        "'subscribe', 'sign up', 'get ticket'), even if the wording is imperfect?"
    )},
    {"label": "Specificity and Actionability", "question": (
        "Is the query specific enough that a search engine could return directly actionable results (for example product pages, "
        "booking forms, download links, streaming pages), rather than only very general informational results?"
    )},
    {"label": "Relevance", "question": (
        "Is the action being requested clearly connected to the given topic (for example buying books, documentaries, courses, "
        "or memorabilia about the topic) rather than an unrelated transaction?"
    )},
    {"label": "Human Realism", "question": (
        "Does the query resemble how a real person would write a transactional search (for example 'cheap X near me', "
        "'buy book about Y online'), including informal language or minor mistakes, instead of sounding like a sterile or "
        "robotic command?"
    )},
    {"label": "Safety and Appropriateness", "question": (
        "Considering the topic, does this transactional intent avoid obviously harmful, illegal, or clearly inappropriate "
        "actions (for example queries that would violate basic safety or content guidelines)?"
    )},
]

COMMERCIAL_INVESTIGATION_SPEC = [
    {"label": "Intent Match", "question": (
        "Does this query show that the user is in a research or comparison phase before taking an action (for example looking "
        "for 'best', 'top', 'reviews', 'ratings', 'pros and cons', or 'recommendations'), rather than just seeking raw facts "
        "or trying to complete a transaction immediately?"
    )},
    {"label": "Evaluation Focus", "question": (
        "Does the query clearly imply that the user wants to evaluate or choose between multiple options (for example which book, "
        "course, documentary, or resource about the topic to pick), instead of asking about a single fixed item?"
    )},
    {"label": "Relevance", "question": (
        "Are the products, resources, or options under consideration clearly connected to the topic (for example learning "
        "materials, media, or tools about the topic), rather than generic items that only share a keyword but are contextually "
        "unrelated?"
    )},
    {"label": "Human Realism", "question": (
        "Does the query match how people usually phrase pre-purchase or pre-action searches (for example 'best X for beginners', "
        "'Y course reviews', 'is X worth it'), including natural imperfections and informal tone?"
    )},
    {"label": "Specificity", "question": (
        "Is the query specific enough that a search engine could plausibly show concrete options to research (for example named "
        "item types, audiences, or use cases), instead of being so vague that it is unclear what kind of options the user wants "
        "to investigate?"
    )},
    {"label": "Diversity", "question": (
        "Within the set of commercial-investigation queries for this topic, does this query target a different kind of item, "
        "audience, or evaluation angle (for example price, quality, depth, difficulty) rather than repeating the same 'best X' "
        "formulation on nearly identical objects?"
    )},
]

SPECS = {
    "informational": INFORMATIONAL_SPEC,
    "exploratory": EXPLORATORY_SPEC,
    "comparative": COMPARATIVE_SPEC,
    "navigational": NAVIGATIONAL_SPEC,
    "transactional": TRANSACTIONAL_SPEC,
    "commercial_investigation": COMMERCIAL_INVESTIGATION_SPEC,
}

def build_generation_summary(category: str, title: str, top_words: List[str]) -> str:
    """Shortened to reduce token count."""
    keywords = ", ".join(top_words[:6])
    return f"Category: {category}\nTopic: {title}\nKeywords: {keywords}"

def build_judge_prompt(
    category: str,
    gen_summary: str,
    query: str,
    spec: List[Dict[str, str]],
) -> str:
    """Prompt: strict JSON only, with <json> markers."""
    rubric_lines = [f"{it['label']}: {it['question']}" for it in spec]
    rubric = "\n".join(rubric_lines)
    labels = [it["label"] for it in spec]
    labels_hint = ", ".join(labels)

    return f"""You are an evaluation assistant. Your task is to score a single search query.

Return STRICTLY a single JSON object between <json> and </json> tags.
Do NOT include any explanation, commentary, or text outside the JSON.
If you are unsure, still return valid JSON with numeric scores (use 0.0 instead of null).

{gen_summary}

Query: "{query}"

Rubric (0=poor, 1=excellent):
{rubric}

JSON schema example (structure only, not actual scores):

<json>
{{
  "scores": {{
    "{labels_hint}": 0.0
  }},
  "total_score": 0.0
}}
</json>""".strip()

# -------------------------
# Async LLM call via ChatLlamaCpp
# -------------------------

async def call_llama_json_async(
    chat_llm: ChatLlamaCpp,
    semaphore: asyncio.Semaphore,
    prompt: str,
) -> Dict[str, Any]:
    """
    Async call to local llama.cpp via LangChain ChatLlamaCpp.
    Attempts to extract a JSON object from possibly noisy output.
    """
    async with semaphore:
        try:
            msg = HumanMessage(content=prompt)
            resp = await chat_llm.ainvoke([msg])
            output_text = resp.content.strip()

            js = extract_json_object(output_text)
            if js is not None:
                return {"success": True, "data": js}

            return {
                "success": False,
                "error": "JSON decode error: could not extract object",
                "output": output_text[:400],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

async def judge_score_async(
    chat_llm: ChatLlamaCpp,
    semaphore: asyncio.Semaphore,
    category: str,
    gen_summary: str,
    query: str,
    spec: List[Dict[str, str]],
) -> Tuple[Optional[float], Dict[str, float], Optional[str]]:
    """
    Async version of judge_score using ChatLlamaCpp.
    Returns: (total_score, per_criterion_scores, error)
    """
    prompt = build_judge_prompt(category, gen_summary, query, spec)
    result = await call_llama_json_async(chat_llm, semaphore, prompt)

    if not result["success"]:
        return None, {}, result.get("error", "unknown error")

    js = result["data"]
    scores = js.get("scores", {})
    total = js.get("total_score", None)

    if total is None or not isinstance(scores, dict):
        return None, {}, "Malformed JSON from judge"

    per = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
    return float(total), per, None

# -------------------------
# Pipeline-friendly client
# -------------------------

class JudgeQueriesMLX:
    """
    LLM-as-a-judge client with async batch processing for high throughput.

    Uses a local llama.cpp gguf model via LangChain ChatLlamaCpp.
    """

    def __init__(
        self,
        *,
        model_path: str = DEFAULT_JUDGE_MODEL_PATH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

        # Configure local ChatLlamaCpp.
        self.chat_llm = ChatLlamaCpp(
            model_path=self.model_path,
            temperature=TEMP,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            n_ctx=4096,          # adjust as needed
            verbose=False,
        )

    async def _score_batch_async(
        self,
        tasks: List[Tuple[int, str, str, str, List[Dict[str, str]]]],
    ) -> List[Tuple[int, Optional[float], Dict[str, float], Optional[str]]]:
        """
        Process a batch of scoring tasks in parallel.
        tasks: [(idx, query, gen_summary, category, spec), ...]
        Returns: [(idx, total_score, per_scores, error), ...]
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async_tasks = []
        for idx, query, gen_summary, category, spec in tasks:
            task = judge_score_async(
                self.chat_llm, semaphore, category, gen_summary, query, spec
            )
            async_tasks.append((idx, task))

        results = []
        for idx, task in async_tasks:
            total, per, error = await task
            results.append((idx, total, per, error))

        return results

    async def score_rows_async(
        self,
        rows: List[Dict[str, Any]],
        category: str,
        topics_by_id: Dict[int, Dict[str, Any]],
        topic_cache: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score queries using async batch processing for maximum throughput (async version).
        """
        if category not in SPECS:
            raise ValueError(f"Unknown category: {category}")
        spec = SPECS[category]
        criterion_labels = [x["label"] for x in spec]

        df = pd.DataFrame(rows)
        if "query" not in df.columns or "topic_id" not in df.columns:
            raise ValueError("Rows must contain 'query' and 'topic_id' fields.")

        results: List[Optional[Dict[str, Any]]] = [None] * len(df)
        tasks: List[Tuple[int, str, str, str, List[Dict[str, str]]]] = []

        # Prepare all tasks
        for idx, row in df.iterrows():
            q = str(row["query"]).strip()
            if not q:
                results[idx] = {"error": "empty_query"}
                continue

            try:
                tid = int(row["topic_id"])
            except Exception:
                results[idx] = {"error": "invalid_topic_id"}
                continue

            cache = topic_cache.get(tid)
            if not cache:
                results[idx] = {"error": f"topic_id {tid} not in cache"}
                continue

            gen_summary = build_generation_summary(
                category=category,
                title=cache["title"],
                top_words=cache["top_words"],
            )
            tasks.append((idx, q, gen_summary, category, spec))

        if not tasks:
            return df.to_dict(orient="records")

        print(f"Scoring {len(tasks)} queries for category '{category}' (max_concurrent={self.max_concurrent})...")
        start_time = time.time()

        # Process all tasks in batches
        all_batch_results = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size

            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} queries)...")
            batch_results = await self._score_batch_async(batch)
            all_batch_results.extend(batch_results)

            elapsed = time.time() - start_time
            completed = len(all_batch_results)
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"  [{completed}/{len(tasks)}] ({rate:.1f} queries/sec)")

        # Collect results
        for idx, total, per, error in all_batch_results:
            if error:
                results[idx] = {"error": error}
            else:
                results[idx] = {"total_score": total, **per}

        elapsed = time.time() - start_time
        if elapsed > 0 and tasks:
            print(f"  ✓ Completed {len(tasks)} queries in {elapsed:.1f}s ({len(tasks)/elapsed:.1f} qps)\n")

        # Merge results back into dataframe
        for idx, res in enumerate(results):
            if not res:
                continue
            for k, v in res.items():
                df.at[idx, k] = v

        # Ensure all columns exist
        for col in ["total_score", "error"] + criterion_labels:
            if col not in df.columns:
                df[col] = None

        return df.to_dict(orient="records")

    # ---- Sync wrapper ----

    def score_rows(
        self,
        rows: List[Dict[str, Any]],
        category: str,
        topics_by_id: Dict[int, Dict[str, Any]],
        topic_cache: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sync wrapper - only use if not in async context."""
        return asyncio.run(self.score_rows_async(rows, category, topics_by_id, topic_cache))
