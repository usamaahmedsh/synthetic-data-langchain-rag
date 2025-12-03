"""Advanced query filtering and quality checks."""

import re
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class AdvancedQueryFilter:
    """Multi-stage query filtering for quality assurance."""

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 150,
        min_words: int = 2,
        max_words: int = 20,
        max_repetition_ratio: float = 0.5,
        min_entropy: float = 1.5,
    ):
        """
        Args:
            min_length: Minimum character length
            max_length: Maximum character length
            min_words: Minimum word count
            max_words: Maximum word count
            max_repetition_ratio: Max ratio of repeated words
            min_entropy: Minimum word entropy (diversity)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
        self.max_repetition_ratio = max_repetition_ratio
        self.min_entropy = min_entropy

    def calculate_entropy(self, query: str) -> float:
        """Calculate word-level entropy (diversity measure)."""
        words = query.lower().split()
        if len(words) < 2:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            entropy -= p * np.log2(p)
        
        return entropy

    def check_repetition(self, query: str) -> bool:
        """Check if query has too many repeated words."""
        words = query.lower().split()
        if len(words) < 3:
            return True
        
        unique_words = len(set(words))
        repetition_ratio = 1.0 - (unique_words / len(words))
        
        return repetition_ratio <= self.max_repetition_ratio

    def check_linguistic_quality(self, query: str) -> Tuple[bool, str]:
        """Check linguistic quality of query."""
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in query if c in '!?.,;:')
        if punct_count > 3:
            return False, "excessive_punctuation"
        
        # Check for multiple consecutive spaces
        if '  ' in query:
            return False, "formatting_error"
        
        # Check for all caps (likely spam/generated)
        if query.isupper() and len(query) > 10:
            return False, "all_caps"
        
        # Check for excessive special characters
        special_chars = sum(1 for c in query if not c.isalnum() and c not in ' -')
        if special_chars > len(query) * 0.2:
            return False, "too_many_special_chars"
        
        # Check for proper word structure
        words = query.split()
        if any(len(w) > 30 for w in words):  # Suspiciously long words
            return False, "invalid_word_length"
        
        return True, ""

    def check_stopword_ratio(self, query: str) -> bool:
        """Check if query has reasonable stopword ratio."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        }
        
        words = query.lower().split()
        if len(words) < 3:
            return True
        
        stopword_count = sum(1 for w in words if w in stopwords)
        stopword_ratio = stopword_count / len(words)
        
        # Too many stopwords = low quality
        # Too few stopwords might be fine (technical queries)
        return stopword_ratio < 0.7

    def filter_query(
        self,
        query: str,
        topic_keywords: List[str],
    ) -> Tuple[bool, str]:
        """
        Comprehensive quality check for a single query.
        
        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        q = query.strip()
        
        # Basic length checks
        if len(q) < self.min_length:
            return False, "too_short"
        if len(q) > self.max_length:
            return False, "too_long"
        
        words = q.split()
        if len(words) < self.min_words:
            return False, "too_few_words"
        if len(words) > self.max_words:
            return False, "too_many_words"
        
        # Entropy check (diversity)
        entropy = self.calculate_entropy(q)
        if entropy < self.min_entropy:
            return False, "low_entropy"
        
        # Repetition check
        if not self.check_repetition(q):
            return False, "too_repetitive"
        
        # Linguistic quality
        is_valid, reason = self.check_linguistic_quality(q)
        if not is_valid:
            return False, reason
        
        # Stopword ratio
        if not self.check_stopword_ratio(q):
            return False, "too_many_stopwords"
        
        # Topic relevance (at least one keyword present)
        if topic_keywords:
            q_lower = q.lower()
            has_topic_keyword = any(
                kw.lower() in q_lower for kw in topic_keywords
            )
            if not has_topic_keyword:
                return False, "not_relevant"
        
        return True, ""

    def filter_batch(
        self,
        queries: List[str],
        topic_keywords: List[str],
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter a batch of queries."""
        valid_queries = []
        rejection_stats = {}
        
        for query in queries:
            is_valid, reason = self.filter_query(query, topic_keywords)
            
            if is_valid:
                valid_queries.append(query)
            else:
                rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
        
        return valid_queries, rejection_stats


def deduplicate_near_duplicates(
    queries: List[str],
    similarity_threshold: float = 0.85,
) -> List[str]:
    """
    Remove near-duplicate queries using TF-IDF similarity.
    
    Args:
        queries: List of query strings
        similarity_threshold: Cosine similarity threshold
        
    Returns:
        Deduplicated list of queries
    """
    if len(queries) < 2:
        return queries
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            lowercase=True,
        )
        tfidf_matrix = vectorizer.fit_transform(queries)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Mark duplicates
        to_remove = set()
        for i in range(len(queries)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(queries)):
                if j in to_remove:
                    continue
                if similarities[i, j] >= similarity_threshold:
                    to_remove.add(j)  # Keep first occurrence
        
        # Return non-duplicates
        return [q for i, q in enumerate(queries) if i not in to_remove]
    
    except Exception as e:
        print(f"  ⚠ Deduplication failed: {e}")
        return queries
