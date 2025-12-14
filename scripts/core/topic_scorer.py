"""Topic quality scoring and selection."""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class TopicQualityScorer:
    """Score topics by coherence and document coverage."""

    def __init__(self):
        """Initialize scorer."""
        pass

    def compute_topic_coherence(self, topic: Dict[str, Any]) -> float:
        """
        Compute topic coherence based on word co-occurrence.
        Higher coherence = more semantically related words.
        
        Args:
            topic: Topic dictionary
            
        Returns:
            Coherence score (0-1)
        """
        top_words = topic.get("top_words", [])[:10]
        
        if len(top_words) < 3:
            return 0.0
        
        # Simple heuristic: check word overlap in representative docs
        rep_docs = topic.get("representative_docs", [])
        if not rep_docs:
            return 0.5  # Neutral score
        
        # Count how many top words appear together in docs
        co_occurrence_score = 0.0
        for doc in rep_docs[:5]:
            doc_lower = doc.lower()
            present_words = sum(1 for w in top_words if w.lower() in doc_lower)
            co_occurrence_score += present_words / len(top_words)
        
        coherence = co_occurrence_score / min(5, len(rep_docs))
        return min(1.0, coherence)

    def compute_document_coverage(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
    ) -> float:
        """
        Compute how well topic covers document content.
        
        Args:
            topic: Topic dictionary
            docs_dir: Directory with documents
            
        Returns:
            Coverage score (0-1)
        """
        rel_docs = topic.get("relevant_documents", [])
        
        if not rel_docs:
            return 0.0
        
        total_chars = 0
        for doc_name in rel_docs:
            doc_path = docs_dir / doc_name
            if not doc_path.exists():
                continue
            
            try:
                text = doc_path.read_text(encoding='utf-8', errors='ignore')
                total_chars += len(text)
            except Exception:
                continue
        
        # Normalize by max expected size (500k chars = 1.0)
        coverage = min(1.0, total_chars / 500_000)
        return coverage

    def compute_topic_specificity(self, topic: Dict[str, Any]) -> float:
        """
        Compute how specific vs generic a topic is.
        More specific = better for query generation.
        
        Args:
            topic: Topic dictionary
            
        Returns:
            Specificity score (0-1)
        """
        top_words = topic.get("top_words", [])
        
        # Generic words (lower specificity)
        generic_words = {
            'said', 'people', 'time', 'year', 'new', 'first',
            'also', 'many', 'some', 'other', 'such', 'more',
            'been', 'about', 'into', 'than', 'them', 'these',
        }
        
        if not top_words:
            return 0.5
        
        generic_count = sum(1 for w in top_words[:10] if w.lower() in generic_words)
        specificity = 1.0 - (generic_count / 10)
        
        return specificity

    def score_topic(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Compute overall topic quality score.
        
        Args:
            topic: Topic dictionary
            docs_dir: Directory with documents
            weights: Scoring weights (coherence, coverage, specificity)
            
        Returns:
            Overall quality score (0-1)
        """
        if weights is None:
            weights = {
                "coherence": 0.3,
                "coverage": 0.5,
                "specificity": 0.2,
            }
        
        coherence = self.compute_topic_coherence(topic)
        coverage = self.compute_document_coverage(topic, docs_dir)
        specificity = self.compute_topic_specificity(topic)
        
        score = (
            weights["coherence"] * coherence +
            weights["coverage"] * coverage +
            weights["specificity"] * specificity
        )
        
        return score

    def rank_topics(
        self,
        topics: List[Dict[str, Any]],
        docs_dir: Path,
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank topics by quality score.
        
        Args:
            topics: List of topic dicts
            docs_dir: Directory with documents
            top_k: Return only top K topics
            
        Returns:
            Ranked list of topics (best first)
        """
        scored_topics = []
        
        for topic in topics:
            score = self.score_topic(topic, docs_dir)
            topic_with_score = topic.copy()
            topic_with_score["quality_score"] = score
            scored_topics.append(topic_with_score)
        
        # Sort by score descending
        scored_topics.sort(key=lambda t: t["quality_score"], reverse=True)
        
        if top_k:
            return scored_topics[:top_k]
        
        return scored_topics
