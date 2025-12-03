"""Shared utility functions."""


from .data_utils import load_topics, precompute_topic_cache
from .text_utils import (
    tokenize,
    clean_text,
    chunk_text,
    normalize_query,
    clean_filename,
    slugify,
)
from .bm25_utils import (
    # REMOVED: build_bm25_for_topic_docs (use build_bm25_with_passages instead)
    build_bm25_with_passages,
    retrieve_best_passages,
    extract_context_window,
    compute_query_document_relevance,
    jaccard,
)
from .output_manager import OutputManager, create_outputs_readme
from .embedding_cache import EmbeddingCache, get_embeddings_with_cache
from .profiler import SimpleProfiler


__all__ = [
    # Data utilities
    "load_topics",
    "precompute_topic_cache",
    # Text utilities
    "tokenize",
    "clean_text",
    "chunk_text",
    "normalize_query",
    "clean_filename",
    "slugify",
    # BM25 utilities
    "build_bm25_with_passages",
    "retrieve_best_passages",
    "extract_context_window",
    "compute_query_document_relevance",
    "jaccard",
    # Output management
    "OutputManager",
    "create_outputs_readme",
    # Embedding cache
    "EmbeddingCache",
    "get_embeddings_with_cache",
    # Profiler
    "SimpleProfiler",
]
