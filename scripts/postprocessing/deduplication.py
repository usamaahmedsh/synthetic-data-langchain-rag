# topics_deduper.py

"""Topic and query deduplication logic."""

from typing import List, Dict, Any, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


TOP_WORDS_OVERLAP_THRESHOLD = 0.6    # 60% overlap on top_words
DOCS_OVERLAP_THRESHOLD = 0.5         # 50% overlap on relevant_documents


def jaccard_overlap(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def decide_keep_drop(t1: Dict[str, Any], t2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (keep_topic, drop_topic) based on higher 'count' (ties by larger topic_id)."""
    c1 = int(t1.get("count", 0))
    c2 = int(t2.get("count", 0))
    if c1 > c2:
        return t1, t2
    if c2 > c1:
        return t2, t1
    # tie-breaker: keep higher topic_id (arbitrary but deterministic)
    if int(t1.get("topic_id", -1)) >= int(t2.get("topic_id", -1)):
        return t1, t2
    else:
        return t2, t1


class BuildTopicsDeduper:
    """
    Deduplicate topics based on overlap in top_words and relevant_documents.

    Pipeline usage:

        from topics_builder import BuildTopics
        from topics_deduper import BuildTopicsDeduper

        topics_res = BuildTopics().run(input_dir=..., topic_name=..., topic_slug=...)
        topics = topics_res["topics"]

        deduper = BuildTopicsDeduper()
        topics_deduped = deduper.run(topics)
    """

    def __init__(
        self,
        *,
        top_words_overlap_threshold: float = TOP_WORDS_OVERLAP_THRESHOLD,
        docs_overlap_threshold: float = DOCS_OVERLAP_THRESHOLD,
    ) -> None:
        self.top_words_overlap_threshold = top_words_overlap_threshold
        self.docs_overlap_threshold = docs_overlap_threshold

    def run(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a new list of topic dicts after dropping near-duplicates."""
        by_id = {int(t["topic_id"]): t for t in topics}
        ids = sorted(by_id.keys())

        to_drop: Set[int] = set()

        for i, id1 in enumerate(ids):
            if id1 in to_drop:
                continue
            t1 = by_id[id1]
            words1 = t1.get("top_words", []) or []
            docs1 = t1.get("relevant_documents", []) or []

            for id2 in ids[i + 1:]:
                if id2 in to_drop:
                    continue
                t2 = by_id[id2]
                words2 = t2.get("top_words", []) or []
                docs2 = t2.get("relevant_documents", []) or []

                # Compute overlaps
                words_overlap = jaccard_overlap(words1, words2)
                docs_overlap = jaccard_overlap(docs1, docs2)

                # If either condition is met, drop the weaker topic
                if (
                    words_overlap >= self.top_words_overlap_threshold
                    or docs_overlap >= self.docs_overlap_threshold
                ):
                    keep, drop = decide_keep_drop(t1, t2)
                    drop_id = int(drop["topic_id"])
                    to_drop.add(drop_id)

        deduped_topics = [t for t in topics if int(t["topic_id"]) not in to_drop]
        print(f"Deduped {len(topics)} → {len(deduped_topics)} topics.")
        return deduped_topics

class QueryDeduplicator:
    """Deduplicate queries across all topics and categories."""

    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold

    def deduplicate(
        self,
        rows: List[Dict[str, Any]],
        keep_highest_score: bool = True,
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate queries globally."""
        if not rows:
            return []

        print(f"  Deduplicating {len(rows)} queries...")
        queries = [str(r.get("query", "")) for r in rows]
        
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
            tfidf_matrix = vectorizer.fit_transform(queries)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            return rows[:1] if rows else []

        to_remove = set()
        n = len(rows)
        
        for i in range(n):
            if i in to_remove:
                continue
            
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                
                if sim_matrix[i, j] >= self.similarity_threshold:
                    if keep_highest_score:
                        score_i = rows[i].get("total_score", 0)
                        score_j = rows[j].get("total_score", 0)
                        
                        if score_i >= score_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
                    else:
                        to_remove.add(j)
        
        deduplicated = [rows[i] for i in range(n) if i not in to_remove]
        duplicates_removed = len(rows) - len(deduplicated)
        print(f"  ✓ Removed {duplicates_removed} duplicates")
        
        return deduplicated