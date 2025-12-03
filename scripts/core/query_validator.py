"""Query validation and filtering."""

import re
from typing import List, Dict, Any, Tuple


class QueryValidator:
    """Validate and filter queries before scoring."""

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 150,
        min_words: int = 2,
        max_words: int = 20,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
        
        self.bad_patterns = [
            r"as an ai",
            r"i cannot",
            r"i don't",
            r"i'm sorry",
            r"sorry,",
            r"query:",
            r"example:",
            r"search for",
            r"here is",
            r"here are",
            r"\[.*\]",
            r"\*\*.*\*\*",
        ]
        self.bad_patterns_compiled = [re.compile(p, re.IGNORECASE) for p in self.bad_patterns]

    def validate_query(
        self,
        query: str,
        topic_keywords: List[str],
        require_topic_relevance: bool = True,
    ) -> Tuple[bool, str]:
        """Validate a single query."""
        q = query.strip()
        
        if len(q) < self.min_length:
            return False, "too_short"
        
        if len(q) > self.max_length:
            return False, "too_long"
        
        words = q.split()
        if len(words) < self.min_words:
            return False, "too_few_words"
        
        if len(words) > self.max_words:
            return False, "too_many_words"
        
        for pattern in self.bad_patterns_compiled:
            if pattern.search(q):
                return False, "generation_artifact"
        
        if q.endswith(("...", "…")):
            return False, "incomplete"
        
        question_only = re.match(r'^(what|how|why|when|where|who|which|is|are|do|does)\s*$', q, re.IGNORECASE)
        if question_only:
            return False, "incomplete_question"
        
        if require_topic_relevance and topic_keywords:
            q_lower = q.lower()
            has_topic_word = any(kw.lower() in q_lower for kw in topic_keywords)
            if not has_topic_word:
                return False, "not_relevant_to_topic"
        
        return True, ""

    def filter_queries(
        self,
        rows: List[Dict[str, Any]],
        topics_by_id: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Filter a list of query rows."""
        valid_rows = []
        rejection_stats = {}
        
        for row in rows:
            query = row.get("query", "")
            topic_id = row.get("topic_id")
            
            topic_keywords = []
            if topic_id and topic_id in topics_by_id:
                topic = topics_by_id[topic_id]
                topic_keywords = topic.get("top_words", [])[:10]
            
            is_valid, reason = self.validate_query(query, topic_keywords)
            
            if is_valid:
                valid_rows.append(row)
            else:
                rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
        
        return valid_rows, rejection_stats
