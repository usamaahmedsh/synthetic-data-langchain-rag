"""Enhanced BM25 utilities with context windows."""

from typing import List, Tuple, Dict, Any
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 2]


def extract_context_window(
    doc_text: str,
    query_tokens: List[str],
    window_size: int = 100,
) -> str:
    """
    Extract relevant context window around query tokens.
    
    Args:
        doc_text: Full document text
        query_tokens: Tokenized query
        window_size: Words before/after match
        
    Returns:
        Relevant context string
    """
    doc_tokens = doc_text.lower().split()
    
    # Find positions of query tokens in document
    matches = []
    for i, token in enumerate(doc_tokens):
        if any(qt in token for qt in query_tokens):
            matches.append(i)
    
    if not matches:
        # No matches, return beginning
        return ' '.join(doc_tokens[:window_size * 2])
    
    # Get window around best match (first match)
    center = matches[0]
    start = max(0, center - window_size)
    end = min(len(doc_tokens), center + window_size)
    
    context = ' '.join(doc_tokens[start:end])
    return context


def build_bm25_with_passages(
    docs_dir: Path,
    passage_size: int = 200,
    overlap: int = 50,
) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Build BM25 index over passages instead of full documents.
    Better for long documents.
    
    Args:
        docs_dir: Directory with text files
        passage_size: Words per passage
        overlap: Overlapping words between passages
        
    Returns:
        Tuple of (BM25 index, passage metadata)
    """
    passages = []
    passage_metadata = []
    
    for doc_path in docs_dir.glob("*.txt"):
        try:
            text = doc_path.read_text(encoding='utf-8', errors='ignore')
            words = text.split()
            
            # Split into overlapping passages
            for i in range(0, len(words), passage_size - overlap):
                passage_words = words[i:i + passage_size]
                if len(passage_words) < 50:  # Skip very short passages
                    continue
                
                passage_text = ' '.join(passage_words)
                passages.append(tokenize(passage_text))
                
                passage_metadata.append({
                    "doc_name": doc_path.name,
                    "start_word": i,
                    "end_word": min(i + passage_size, len(words)),
                    "text": passage_text[:500],  # Store snippet
                })
        
        except Exception as e:
            print(f"  ⚠ Error processing {doc_path}: {e}")
            continue
    
    if not passages:
        return None, []
    
    bm25 = BM25Okapi(passages)
    return bm25, passage_metadata


def retrieve_best_passages(
    bm25: BM25Okapi,
    passage_metadata: List[Dict[str, Any]],
    query_tokens: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant passages for query.
    
    Args:
        bm25: BM25 index
        passage_metadata: Passage information
        query_tokens: Tokenized query
        top_k: Number of passages to return
        
    Returns:
        List of passage dicts with scores
    """
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if idx >= len(passage_metadata):
            continue
        
        passage = passage_metadata[idx].copy()
        passage["bm25_score"] = float(scores[idx])
        results.append(passage)
    
    return results


def compute_query_document_relevance(
    query: str,
    bm25: BM25Okapi,
    passage_metadata: List[Dict[str, Any]],
    top_k: int = 5,
) -> float:
    """
    Compute relevance score between query and document corpus.
    
    Args:
        query: Query string
        bm25: BM25 index
        passage_metadata: Passage information
        top_k: Number of top passages to consider
        
    Returns:
        Relevance score (0-1)
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    
    passages = retrieve_best_passages(
        bm25, passage_metadata, query_tokens, top_k
    )
    
    if not passages:
        return 0.0
    
    # Use max score
    max_score = max(p["bm25_score"] for p in passages)
    
    # Normalize (BM25 scores are unbounded)
    # Use sigmoid to map to [0, 1]
    normalized = 1 / (1 + np.exp(-max_score / 10))
    
    return float(normalized)


def jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Jaccard similarity between token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
