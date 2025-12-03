"""Text processing utilities shared across modules."""

import re
from typing import List


# Regex patterns
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9_-]")


def tokenize(text: str) -> List[str]:
    """
    Simple word tokenization (alphanumeric + underscore).
    
    Args:
        text: Input text
        
    Returns:
        List of lowercase tokens
    """
    return WORD_RE.findall(text.lower())


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    return WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, max_words: int = 150, overlap: int = 0) -> List[str]:
    """
    Split text into chunks by word count with optional overlap.
    
    Args:
        text: Input text
        max_words: Maximum words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]
    
    step = max(1, max_words - overlap)
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), step)
    ]


def normalize_query(q: str) -> str:
    """
    Normalize query for deduplication.
    
    Args:
        q: Query string
        
    Returns:
        Normalized lowercase query with normalized whitespace
    """
    q = q.strip().strip("-").rstrip(".?!")
    q = WHITESPACE_RE.sub(" ", q)
    return q.lower()


def clean_filename(name: str, max_length: int = 150) -> str:
    """
    Clean a string to be safe for use as a filename.
    
    Args:
        name: Input string
        max_length: Maximum length of filename
        
    Returns:
        Safe filename string
    """
    cleaned = NON_ALNUM_RE.sub("_", name)
    return cleaned[:max_length]


def slugify(s: str) -> str:
    """
    Convert string to URL-safe slug.
    
    Args:
        s: Input string
        
    Returns:
        Lowercase slug with underscores
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip().lower())
    return cleaned.strip("_")
