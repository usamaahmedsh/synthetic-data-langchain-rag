"""Utilities for loading and caching topic data."""

import json
from pathlib import Path
from typing import Dict, Set, Any


def load_topics(topics_json_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load topics from topics.json and return dict keyed by topic_id.
    
    Args:
        topics_json_path: Path to topics.json file
        
    Returns:
        Dictionary mapping topic_id -> topic dict
    """
    topics_by_id = {}
    
    if not topics_json_path.exists():
        print(f"Warning: {topics_json_path} does not exist")
        return topics_by_id
    
    with open(topics_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Handle both list and dict formats
        topics_list = data.get('topics', data) if isinstance(data, dict) else data
    
    for topic in topics_list:
        tid = topic.get('topic_id')
        if tid is not None:
            topics_by_id[int(tid)] = topic
    
    return topics_by_id


def precompute_topic_cache(
    topic_ids: Set[int],
    topics_by_id: Dict[int, Dict],
    docs_dir: Path
) -> Dict[int, Dict[str, Any]]:
    """
    Precompute topic metadata cache for faster lookup.
    
    Args:
        topic_ids: Set of topic IDs to cache
        topics_by_id: Dictionary of topic metadata
        docs_dir: Directory containing topic documents
        
    Returns:
        Dictionary mapping topic_id -> cached metadata
    """
    cache = {}
    
    for tid in topic_ids:
        if tid not in topics_by_id:
            continue
            
        topic = topics_by_id[tid]
        cache[tid] = {
            'topic_name': topic.get('topic_name', ''),
            'topic_slug': topic.get('topic_slug', ''),
            'top_words': topic.get('top_words', []),
            'relevant_documents': topic.get('relevant_documents', []),
            'count': topic.get('count', 0),
        }
    
    return cache
