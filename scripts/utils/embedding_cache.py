"""Embedding cache for avoiding recomputation."""

import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""

    def __init__(self, cache_dir: Path, model_name: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        self.cache_file = self.cache_dir / f"embeddings_{model_hash}.pkl"
        
        self.cache: Dict[str, np.ndarray] = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"  ✓ Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"  ⚠ Could not load cache: {e}")

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        return self.cache.get(text)

    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding for text."""
        self.cache[text] = embedding

    def save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"  ✓ Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            print(f"  ⚠ Could not save cache: {e}")


def get_embeddings_with_cache(
    texts: List[str],
    model,
    cache: Optional[EmbeddingCache] = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """Get embeddings with caching."""
    from tqdm import tqdm
    
    if cache is None:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
    
    embeddings = []
    missing_texts = []
    missing_indices = []
    
    for i, text in enumerate(texts):
        emb = cache.get(text)
        if emb is not None:
            embeddings.append(emb)
        else:
            missing_texts.append(text)
            missing_indices.append(i)
    
    if missing_texts:
        cache_hits = len(texts) - len(missing_texts)
        print(f"  Cache: {cache_hits}/{len(texts)} hits, computing {len(missing_texts)} new embeddings")
        
        new_embs = model.encode(
            missing_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        
        for text, emb in zip(missing_texts, new_embs):
            cache.set(text, emb)
        
        for idx, emb in zip(missing_indices, new_embs):
            embeddings.insert(idx, emb)
        
        cache.save()
    else:
        print(f"  ✓ All {len(texts)} embeddings loaded from cache")
    
    return np.asarray(embeddings)
