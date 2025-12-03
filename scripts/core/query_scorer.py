"""Query scoring implementations with performance optimizations."""


import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from tqdm import tqdm


from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# Import utilities
from utils.text_utils import tokenize
from utils.bm25_utils import build_bm25_with_passages, jaccard
from utils.data_utils import load_topics
from utils.embedding_cache import EmbeddingCache, get_embeddings_with_cache



class HeuristicQueryScorer:
    """Score queries with heuristics + diversity (optimized)."""


    def __init__(
        self,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_max_sim: float = 0.85,
        lexical_min_dist: float = 0.3,
        batch_size: int = 256,  # Larger batch for GPU
        use_cache: bool = True,
        use_gpu: bool = True,
    ):
        self.semantic_model_name = semantic_model_name
        self.semantic_max_sim = semantic_max_sim
        self.lexical_min_dist = lexical_min_dist
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.use_gpu = use_gpu
        
        self.model = None
        self.cache = None
        self.device = None


    def _get_device(self) -> str:
        """Get computation device (GPU if available)."""
        if self.device is None:
            if self.use_gpu:
                # Check for CUDA (NVIDIA)
                if torch.cuda.is_available():
                    self.device = "cuda"
                    print(f"  ✓ Using GPU: {torch.cuda.get_device_name(0)}")
                # Check for MPS (Apple Silicon)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                    print(f"  ✓ Using Apple Silicon GPU (MPS)")
                else:
                    self.device = "cpu"
                    print(f"  ℹ Using CPU")
            else:
                self.device = "cpu"
                print(f"  ℹ Using CPU")
        return self.device



    def _get_model(self) -> SentenceTransformer:
        """Lazy load sentence transformer model."""
        if self.model is None:
            device = self._get_device()
            self.model = SentenceTransformer(self.semantic_model_name, device=device)
        return self.model


    def _get_cache(self) -> EmbeddingCache:
        """Lazy load embedding cache."""
        if self.cache is None and self.use_cache:
            from config.settings import ARTIFACTS_DIR
            self.cache = EmbeddingCache(
                cache_dir=Path(ARTIFACTS_DIR) / "embedding_cache",
                model_name=self.semantic_model_name,
            )
        return self.cache


    def score_queries(
        self,
        rows: List[Dict[str, Any]],
        category: str,
        topics_json_path: Path,
        docs_dir: Path,
        bm25_indices: Dict[int, Tuple[BM25Okapi, List[Dict]]] = None,  # Pre-built BM25 indices with passage metadata
    ) -> List[Dict[str, Any]]:
        """Score queries with heuristics + diversity."""
        if not rows:
            return []


        topics_by_id = load_topics(topics_json_path)
        docs_dir_path = Path(docs_dir)


        # Build or use pre-built BM25 indices
        if bm25_indices is None:
            print("  Building BM25 indices...")
            bm25_by_topic: Dict[int, Tuple[BM25Okapi, List[Dict]]] = {}
            for tid in tqdm(set(int(r["topic_id"]) for r in rows if "topic_id" in r), desc="BM25 indexing"):
                # Load documents for this topic
                topic_docs_path = docs_dir_path / f"topic_{tid}_documents.json"
                if topic_docs_path.exists():
                    with open(topic_docs_path, 'r') as f:
                        docs = json.load(f)
                    doc_texts = [doc.get("text", "") for doc in docs]
                    bm25_index, passage_metadata = build_bm25_with_passages(doc_texts)
                    bm25_by_topic[tid] = (bm25_index, passage_metadata)
                else:
                    print(f"  ⚠ Warning: No documents found for topic {tid}")
        else:
            print("  ✓ Using pre-built BM25 indices")
            bm25_by_topic = bm25_indices


        # Get embeddings with cache
        model = self._get_model()
        cache = self._get_cache() if self.use_cache else None
        
        queries = [str(r["query"]) for r in rows]
        print(f"  Computing embeddings for {len(queries)} queries...")
        embs = get_embeddings_with_cache(
            texts=queries,
            model=model,
            cache=cache,
            batch_size=self.batch_size,
            show_progress=True,
        )


        # Tokenize all queries (can be parallelized further)
        print("  Tokenizing queries...")
        tokens_all = [tokenize(q) for q in tqdm(queries, desc="Tokenizing")]


        # Group by (topic_id, query_type)
        grouped: Dict[tuple, List[int]] = {}
        for idx, r in enumerate(rows):
            try:
                tid = int(r["topic_id"])
            except Exception:
                continue
            qtype = str(r.get("query_type", ""))
            grouped.setdefault((tid, qtype), []).append(idx)


        # Quality score helper
        def quality_score(idx: int, topic_id: int) -> float:
            q_tokens = tokens_all[idx]
            n = len(q_tokens)
            if n < 2:
                return 0.0


            s_len = float(np.exp(-((n - 6) ** 2) / 18.0))


            bm25_tuple = bm25_by_topic.get(topic_id)
            if bm25_tuple is None:
                s_bm25 = 0.0
            else:
                bm25, _ = bm25_tuple  # Unpack tuple
                s = bm25.get_scores(q_tokens)
                s_bm25 = float(np.max(s)) if len(s) > 0 else 0.0


            return 0.6 * s_bm25 + 0.4 * s_len


        # Compute scores per group with diversity
        print("  Computing diversity scores...")
        for (tid, qtype), idxs in tqdm(grouped.items(), desc="Scoring groups"):
            if not idxs:
                continue


            q_raw = {i: quality_score(i, tid) for i in idxs}


            vals = np.array(list(q_raw.values()), dtype=float)
            q_min, q_max = float(vals.min()), float(vals.max())
            for i in idxs:
                if q_max > q_min:
                    rows[i]["quality_score"] = (q_raw[i] - q_min) / (q_max - q_min)
                else:
                    rows[i]["quality_score"] = 0.0


            selected: List[int] = []


            for i in sorted(idxs, key=lambda j: rows[j]["quality_score"], reverse=True):
                e_i = embs[i]
                t_i = tokens_all[i]


                if not selected:
                    rows[i]["lex_div_score"] = 1.0
                    rows[i]["sem_div_score"] = 1.0
                    rows[i]["total_score"] = rows[i]["quality_score"]
                    selected.append(i)
                    continue


                lex_sims = [jaccard(t_i, tokens_all[j]) for j in selected]
                max_lex_sim = max(lex_sims)
                d_lex_min = 1.0 - max_lex_sim


                sem_sims = [float(np.dot(e_i, embs[j])) for j in selected]
                max_sem_sim = max(sem_sims)
                d_sem = 1.0 - max_sem_sim


                rows[i]["lex_div_score"] = d_lex_min
                rows[i]["sem_div_score"] = d_sem


                total = (
                    0.6 * rows[i]["quality_score"]
                    + 0.2 * d_lex_min
                    + 0.2 * d_sem
                )
                rows[i]["total_score"] = total


                selected.append(i)


        return rows



class LLMQueryScorer:
    """Score queries using LLM-as-a-judge."""


    def __init__(
        self,
        model_path: str,
        batch_size: int = 24,
        max_concurrent: int = 12,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent


    def score_queries(
        self,
        rows: List[Dict[str, Any]],
        category: str,
        topics_by_id: Dict[int, Dict[str, Any]],
        topic_cache: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score queries with LLM judge."""
        raise NotImplementedError(
            "LLM scoring not yet integrated. "
            "Use heuristic_score_queries or integrate judge_mlx_client.py here."
        )
