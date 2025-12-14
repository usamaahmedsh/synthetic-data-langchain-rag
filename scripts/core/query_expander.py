"""Query expansion and variation generation."""

from typing import List, Set
import re


class QueryExpander:
    """Expand queries with variations and synonyms."""

    # Common synonyms for search intents
    INTENT_SYNONYMS = {
        "what is": ["what are", "define", "meaning of", "explain"],
        "how to": ["how do i", "how can i", "ways to", "guide to"],
        "best": ["top", "greatest", "finest", "recommended"],
        "vs": ["versus", "compared to", "difference between"],
        "buy": ["purchase", "get", "order", "shop for"],
        "free": ["no cost", "gratis", "complimentary"],
    }

    def __init__(self):
        """Initialize expander."""
        pass

    def expand_with_synonyms(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate query variations using synonym substitution.
        
        Args:
            query: Original query
            max_variations: Max variations to generate
            
        Returns:
            List of query variations (including original)
        """
        variations = [query]
        query_lower = query.lower()
        
        # Try to substitute intent phrases
        for original, synonyms in self.INTENT_SYNONYMS.items():
            if original in query_lower:
                for syn in synonyms[:max_variations - 1]:
                    # Case-preserving replacement
                    if query_lower.startswith(original):
                        # Beginning of query
                        variation = syn.capitalize() + query[len(original):]
                    else:
                        # Middle of query
                        variation = query_lower.replace(original, syn)
                    
                    variations.append(variation)
                    
                    if len(variations) >= max_variations + 1:
                        break
                break
        
        return variations[:max_variations + 1]

    def expand_with_morphology(self, query: str) -> List[str]:
        """
        Generate morphological variations (plural, tense, etc).
        
        Args:
            query: Original query
            
        Returns:
            List of variations
        """
        variations = [query]
        words = query.split()
        
        # Simple plural/singular transformations
        for i, word in enumerate(words):
            if word.endswith('s') and len(word) > 3:
                # Try singular
                singular = word[:-1]
                new_query = ' '.join(words[:i] + [singular] + words[i+1:])
                variations.append(new_query)
            elif not word.endswith('s') and len(word) > 3:
                # Try plural
                plural = word + 's'
                new_query = ' '.join(words[:i] + [plural] + words[i+1:])
                variations.append(new_query)
        
        return list(set(variations))  # Remove duplicates

    def expand_with_context(
        self,
        query: str,
        topic_name: str,
        max_variations: int = 2,
    ) -> List[str]:
        """
        Add topic context to short queries.
        
        Args:
            query: Original query
            topic_name: Main topic
            max_variations: Max variations
            
        Returns:
            List of expanded queries
        """
        variations = [query]
        
        # If query is very short, add topic context
        if len(query.split()) <= 3:
            # Add topic at end
            variations.append(f"{query} {topic_name}")
            
            # Add topic at beginning
            variations.append(f"{topic_name} {query}")
        
        return variations[:max_variations + 1]
