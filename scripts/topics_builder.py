# topics_builder.py

import os
import re
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterable, Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from tqdm import tqdm

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import umap
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"


# --------- helpers --------- #

def read_txt_files(root: str) -> List[Tuple[str, str]]:
    """Read text files with parallel processing."""
    from concurrent.futures import ThreadPoolExecutor
    
    files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(".txt")]
    
    def read_single_file(fname):
        path = os.path.join(root, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            txt = txt.replace("\ufeff", " ").strip()
            if txt:
                return (fname, txt)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
        return None
    
    # Parallel file reading
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_single_file, files))
    
    return [r for r in results if r is not None]


def segment_text(text: str, max_words: int = 150) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def clean_for_display(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:max_len] + "…") if len(s) > max_len else s


def build_pseudo_pairs(
    docs: List[str],
    doc2file: List[str],
    pairs_per_doc: int = 4,
    seed: int = 42,
) -> List[InputExample]:
    import random
    rng = random.Random(seed)
    file_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, fn in enumerate(doc2file):
        file_to_indices[fn].append(i)

    examples: List[InputExample] = []
    for fn, idxs in file_to_indices.items():
        if len(idxs) < 2:
            continue
        attempts = min(pairs_per_doc, max(1, len(idxs) // 2))
        for _ in range(attempts):
            a, b = rng.sample(idxs, 2)
            examples.append(InputExample(texts=[docs[a], docs[b]], label=1.0))
    return examples


def reassign_outliers(
    embeddings: np.ndarray,
    topics: List[int],
    valid_topic_ids: List[int],
    min_sim: float = 0.50,
) -> List[int]:
    from sklearn.preprocessing import normalize
    X = normalize(embeddings)
    topic_ids = [t for t in valid_topic_ids if t != -1]
    if not topic_ids:
        return topics

    centroids = {}
    t_arr = np.asarray(topics)
    for tid in topic_ids:
        idx = np.where(t_arr == tid)[0]
        if len(idx) == 0:
            continue
        centroids[tid] = X[idx].mean(axis=0)

    out_idx = np.where(t_arr == -1)[0]
    for i in out_idx:
        sims = {tid: float(X[i] @ centroids[tid].T) for tid in centroids}
        if not sims:
            continue
        best_tid, best_sim = max(sims.items(), key=lambda kv: kv[1])
        if best_sim >= min_sim:
            topics[i] = best_tid
    return topics


def dedup_topics_by_ctfidf(topic_model: BERTopic, sim_thresh: float = 0.85):
    import itertools
    C = topic_model.c_tf_idf_.toarray()
    keep = set(range(len(C)))
    parent = {i: i for i in range(len(C))}
    norms = np.linalg.norm(C, axis=1) + 1e-12

    for i, j in itertools.combinations(range(len(C)), 2):
        if i not in keep or j not in keep:
            continue
        num = float(np.dot(C[i], C[j]))
        den = float(norms[i] * norms[j])
        sim = num / den if den > 0 else 0.0
        if sim >= sim_thresh:
            keep.discard(j)
            parent[j] = i
    return keep, parent


# --------- BuildTopics client --------- #

class BuildTopics:
    """
    Optimized BERTopic builder with caching and parallel processing.
    
    Key optimizations:
    - Embedding caching (avoid recomputation)
    - Parallel file I/O
    - Multi-core UMAP/HDBSCAN
    - Batch processing
    - Low memory mode
    """

    def __init__(
        self,
        *,
        base_model: str = "all-MiniLM-L6-v2",  # Lightweight, fast model
        chunk_words: int = 150,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        neighbors: int = 10,
        min_df: Any = 5,
        max_df: float = 0.80,
        finetune: bool = False,
        epochs: int = 1,
        pairs_per_doc: int = 4,
        batch_size: int = 32,
        dedup_drop: bool = True,
        artifacts_root: str = "artifacts",
        use_cache: bool = True,  # NEW: Enable caching
    ) -> None:
        self.base_model = base_model
        self.chunk_words = chunk_words
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.neighbors = neighbors
        self.min_df = min_df
        self.max_df = max_df
        self.finetune = finetune
        self.epochs = epochs
        self.pairs_per_doc = pairs_per_doc
        self.batch_size = batch_size
        self.dedup_drop = dedup_drop
        self.artifacts_root = artifacts_root
        self.use_cache = use_cache

    def _get_cache_paths(self, artifacts_root: Path) -> Dict[str, Path]:
        """Get paths for cached artifacts."""
        return {
            "embeddings": artifacts_root / "embeddings.npy",
            "docs": artifacts_root / "docs.pkl",
            "doc2file": artifacts_root / "doc2file.pkl",
            "model": artifacts_root / "bertopic_coverage" / "model.pkl",
        }

    def _load_or_process_docs(
        self,
        input_dir: str,
        cache_paths: Dict[str, Path],
    ) -> Tuple[List[str], List[str]]:
        """Load docs from cache or process from files."""
        docs_cache = cache_paths["docs"]
        doc2file_cache = cache_paths["doc2file"]
        
        if self.use_cache and docs_cache.exists() and doc2file_cache.exists():
            print("✓ Loading documents from cache...")
            with open(docs_cache, "rb") as f:
                docs = pickle.load(f)
            with open(doc2file_cache, "rb") as f:
                doc2file = pickle.load(f)
            return docs, doc2file
        
        print("Processing documents...")
        file_texts = read_txt_files(input_dir)
        if not file_texts:
            raise ValueError(f"No .txt files found under {input_dir}")

        docs: List[str] = []
        doc2file: List[str] = []
        
        for fname, txt in tqdm(file_texts, desc="Segmenting documents"):
            for chunk in segment_text(txt, max_words=self.chunk_words):
                docs.append(chunk)
                doc2file.append(fname)
        
        # Cache for future runs
        with open(docs_cache, "wb") as f:
            pickle.dump(docs, f)
        with open(doc2file_cache, "wb") as f:
            pickle.dump(doc2file, f)
        
        return docs, doc2file

    def _load_or_compute_embeddings(
        self,
        docs: List[str],
        embedder: SentenceTransformer,
        cache_paths: Dict[str, Path],
    ) -> np.ndarray:
        """Load embeddings from cache or compute new ones."""
        embeddings_cache = cache_paths["embeddings"]
        
        if self.use_cache and embeddings_cache.exists():
            print("✓ Loading embeddings from cache...")
            return np.load(embeddings_cache)
        
        print(f"Computing embeddings for {len(docs)} documents on {DEVICE}...")
        
        try:
            import torch as _torch
            ctx = _torch.no_grad()
        except Exception:
            class _NullCtx:
                def __enter__(self): pass
                def __exit__(self, *a): pass
            ctx = _NullCtx()

        with ctx:
            embeddings = embedder.encode(
                docs,
                batch_size=64,  # Optimized batch size
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        
        # Cache for future runs
        np.save(embeddings_cache, embeddings)
        print(f"✓ Embeddings cached to {embeddings_cache}")
        
        return embeddings

    def run(
        self,
        input_dir: str,
        topic_name: str,
        topic_slug: str,
    ) -> Dict[str, Any]:
        """
        Build topics from .txt files with optimizations.
        
        Returns:
            {
              "topics": [ ... topic dicts ... ],
              "model_dir": <path to BERTopic model>,
              "embeddings_path": <path>,
              "docs_path": <path>
            }
        """
        artifacts_root = Path(self.artifacts_root) / topic_slug
        artifacts_root.mkdir(parents=True, exist_ok=True)
        
        cache_paths = self._get_cache_paths(artifacts_root)

        # 1) Load or process documents (with caching)
        docs, doc2file = self._load_or_process_docs(input_dir, cache_paths)
        print(f"✓ Loaded {len(docs)} document chunks")

        # 2) Embedding model (+ optional fine-tuning)
        model_id = self.base_model
        print(f"Loading embedding model: {model_id} on {DEVICE}")
        embedder = SentenceTransformer(model_id, device=DEVICE)

        if self.finetune:
            print("Fine-tuning embedding model...")
            train_examples = build_pseudo_pairs(
                docs, doc2file, pairs_per_doc=self.pairs_per_doc, seed=42
            )
            if len(train_examples) >= 2:
                train_loader = DataLoader(
                    train_examples,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                train_loss = losses.MultipleNegativesRankingLoss(embedder)
                warmup_steps = max(10, int(0.05 * len(train_loader) * self.epochs))
                embedder.fit(
                    train_objectives=[(train_loader, train_loss)],
                    epochs=self.epochs,
                    warmup_steps=warmup_steps,
                    show_progress_bar=True,
                )
                ft_dir = artifacts_root / "finetuned_model"
                ft_dir.mkdir(exist_ok=True)
                embedder.save(str(ft_dir))
                model_id = str(ft_dir)
                embedder = SentenceTransformer(model_id, device=DEVICE)
                print(f"✓ Model fine-tuned and saved to {ft_dir}")

        # 3) Load or compute embeddings (with caching)
        embeddings = self._load_or_compute_embeddings(docs, embedder, cache_paths)

        # Save docs as JSONL for compatibility
        docs_jsonl_path = artifacts_root / "docs.jsonl"
        if not docs_jsonl_path.exists():
            with open(docs_jsonl_path, "w", encoding="utf-8") as f:
                for d in docs:
                    f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")

        # 4) Optimized BERTopic with multi-core processing
        print("Building topic model (UMAP + HDBSCAN)...")
        
        umap_model = umap.UMAP(
            n_neighbors=int(self.neighbors),
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,  # Memory optimization
            n_jobs=-1,  # Use all CPU cores
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=int(self.min_cluster_size),
            min_samples=int(self.min_samples),
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,
            core_dist_n_jobs=-1,  # Use all CPU cores
        )

        # min_df parsing
        min_df = self.min_df
        if isinstance(min_df, str) and any(ch in min_df for ch in ".eE"):
            min_df = float(min_df)
        else:
            try:
                min_df = int(min_df)
            except Exception:
                min_df = 2

        topic_model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=float(self.max_df),
            ),
            calculate_probabilities=False,  # Faster
            verbose=False,
            low_memory=True,
            language="english",
        )

        topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
        print(f"✓ Discovered {len(set(topics))} initial topics")

        # 5) Reassign outliers and dedup
        print("Refining topics (outlier reassignment + deduplication)...")
        valid_ids = [t for t in set(topics) if t != -1]
        topics = reassign_outliers(embeddings, list(topics), valid_ids, min_sim=0.50)

        keep, parent_map = dedup_topics_by_ctfidf(topic_model, sim_thresh=0.90)

        info_df = topic_model.get_topic_info()
        info_df = info_df[info_df["Topic"] != -1].reset_index(drop=True)

        topic_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, t in enumerate(topics):
            if t != -1:
                topic_to_indices[int(t)].append(idx)

        # 6) Build topic dicts
        print("Building topic metadata...")
        results = []
        for _, row in info_df.iterrows():
            t_id = int(row["Topic"])

            if self.dedup_drop and t_id not in keep:
                continue

            words_weights = topic_model.get_topic(t_id) or []
            top_words = [w for (w, wt) in words_weights][:15]

            rep_snips_raw = topic_model.get_representative_docs(t_id) or []
            rep_snips = [clean_for_display(s) for s in rep_snips_raw[:3]]

            indices = topic_to_indices.get(t_id, [])
            file_counts = Counter(doc2file[i] for i in indices)
            relevant_documents = [fn for fn, _ in file_counts.most_common(10)]

            results.append(
                {
                    "topic_id": t_id,
                    "topic_slug": topic_slug,
                    "topic_name": topic_name,
                    "count": int(row["Count"]),
                    "top_words": top_words,
                    "representative_docs": rep_snips,
                    "relevant_documents": relevant_documents,
                }
            )
        
        # 7) Save model
        model_dir = artifacts_root / "bertopic_coverage"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        topic_model.save(str(model_path))
        print(f"✓ Model saved to {model_path}")

        print(f"\n{'='*60}")
        print(f"✓ Topic modeling complete!")
        print(f"  Final topics: {len(results)}")
        print(f"  Total documents: {len(docs)}")
        print(f"{'='*60}\n")

        return {
            "topics": results,
            "model_dir": str(model_dir),
            "embeddings_path": str(cache_paths["embeddings"]),
            "docs_path": str(docs_jsonl_path),
        }
