"""Query generation using LLM with parallel processing."""

import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BuildQueries:
    """
    Generate search queries for topics using LLM.
    Supports parallel processing for efficiency.
    """

    # Default taxonomy categories
    DEFAULT_TAXONOMY = [
        "informational",
        "exploratory",
        "navigational",
        "comparative",
        "transactional",
        "commercial_investigation",
    ]

    # Category descriptions for prompt
    CATEGORY_DESCRIPTIONS = {
        "informational": "Seeking factual information, definitions, or explanations",
        "exploratory": "Broad research or discovery of new information",
        "navigational": "Finding a specific website, page, or resource",
        "comparative": "Comparing two or more entities, products, or concepts",
        "transactional": "Intent to take action (purchase, download, sign up, etc.)",
        "commercial_investigation": "Pre-purchase research or product evaluation",
    }

    def __init__(
        self,
        hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        llama_cpp_url: str = "http://127.0.0.1:8080",
        max_parallel_topics: int = 8,
        max_parallel_categories: int = 6,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 80,
        timeout: float = 30.0,
    ):
        """
        Initialize query generator.
        
        Args:
            hf_model: Hugging Face model ID (for reference/logging)
            llama_cpp_url: URL of llama.cpp server
            max_parallel_topics: Max topics to process in parallel
            max_parallel_categories: Max categories per topic in parallel
            temperature: LLM temperature
            top_p: LLM top_p
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds
        """
        self.hf_model = hf_model
        self.llama_cpp_url = llama_cpp_url.rstrip("/")
        self.max_parallel_topics = max_parallel_topics
        self.max_parallel_categories = max_parallel_categories
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

def _build_prompt(
    self,
    topic_name: str,
    category: str,
    context: str,
    num_queries: int,
) -> str:
    """Build enhanced prompt with examples and constraints."""
    
    category_desc = self.CATEGORY_DESCRIPTIONS.get(category, "")
    
    # Category-specific examples
    category_examples = {
        "informational": [
            "what is {topic}",
            "{topic} definition and meaning",
            "explain {topic} in simple terms",
        ],
        "exploratory": [
            "{topic} overview and history",
            "everything about {topic}",
            "comprehensive guide to {topic}",
        ],
        "navigational": [
            "{topic} official website",
            "{topic} wikipedia page",
            "where to find {topic}",
        ],
        "comparative": [
            "{topic} vs alternatives",
            "difference between {topic} and",
            "compare {topic} with",
        ],
        "transactional": [
            "buy {topic}",
            "download {topic}",
            "sign up for {topic}",
        ],
        "commercial_investigation": [
            "{topic} reviews",
            "best {topic} options",
            "{topic} pros and cons",
        ],
    }
    
    examples = category_examples.get(category, [])
    example_text = "\n".join([f"  - {ex.format(topic=topic_name)}" for ex in examples[:3]])

    # Enhanced prompt with negative examples
    prompt = f"""You are a search query expert. Generate realistic search queries that users would type into Google.

Topic: {topic_name}
Category: {category} ({category_desc})

Context from documents:
{context}

Task: Generate exactly {num_queries} diverse, natural search queries about "{topic_name}" in the "{category}" category.

Examples of GOOD queries for this category:
{example_text}

Quality Guidelines:
✓ Natural language (how real people search)
✓ Varied lengths (2-15 words)
✓ Different aspects of the topic
✓ Use synonyms and related terms
✓ Include both simple and complex queries

DO NOT:
✗ Include numbering, bullets, or formatting
✗ Add explanations or metadata
✗ Generate duplicate or near-duplicate queries
✗ Use phrases like "search for" or "query about"
✗ Generate incomplete or truncated queries
✗ Use overly formal or academic language

Output format: One query per line, nothing else.

Queries:"""

    return prompt

    async def _call_llm_async(
        self,
        prompt: str,
        client: httpx.AsyncClient,
    ) -> str:
        """
        Call llama.cpp server asynchronously.
        
        Args:
            prompt: Prompt to send
            client: httpx AsyncClient
            
        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n_predict": self.max_tokens,
            "stop": ["\n\n", "###", "---"],
        }

        try:
            response = await client.post(
                f"{self.llama_cpp_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "").strip()
        except Exception as e:
            print(f"  ⚠ LLM call failed: {e}")
            return ""

    def _extract_queries(self, text: str, num_expected: int) -> List[str]:
        """
        Extract queries from LLM output.
        
        Args:
            text: Raw LLM output
            num_expected: Expected number of queries
            
        Returns:
            List of extracted queries
        """
        lines = text.split("\n")
        queries = []

        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove common prefixes
            line = line.lstrip("-*•123456789. ")
            
            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            elif line.startswith("'") and line.endswith("'"):
                line = line[1:-1]
            
            line = line.strip()
            
            # Skip if too short or too long
            if len(line) < 3 or len(line) > 200:
                continue
            
            # Skip if it looks like metadata
            if any(x in line.lower() for x in ["query:", "example:", "search:", "note:"]):
                continue
            
            queries.append(line)
            
            # Stop if we have enough
            if len(queries) >= num_expected * 2:  # Allow some extras for filtering
                break

        return queries[:num_expected * 2]  # Return up to 2x expected (for quality filtering)

    def _get_context_for_topic(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
        max_chars: int = 2000,
    ) -> str:
        """
        Get context text for a topic from its documents.
        
        Args:
            topic: Topic dictionary
            docs_dir: Directory containing documents
            max_chars: Maximum characters to extract
            
        Returns:
            Context string
        """
        relevant_docs = topic.get("relevant_documents", [])
        
        context_parts = []
        total_chars = 0

        for doc_name in relevant_docs[:5]:  # Use top 5 relevant docs
            doc_path = docs_dir / doc_name
            if not doc_path.exists():
                continue

            try:
                text = doc_path.read_text(encoding="utf-8", errors="ignore")
                # Take first portion of document
                chunk = text[:500]
                context_parts.append(chunk)
                total_chars += len(chunk)

                if total_chars >= max_chars:
                    break
            except Exception:
                continue

        if not context_parts:
            # Fallback to topic name and top words
            top_words = topic.get("top_words", [])[:10]
            return f"Topic keywords: {', '.join(top_words)}"

        return "\n\n".join(context_parts)[:max_chars]

    async def generate_queries_for_category_async(
        self,
        topic: Dict[str, Any],
        category: str,
        docs_dir: Path,
        num_queries: int,
        client: httpx.AsyncClient,
    ) -> List[Dict[str, Any]]:
        """
        Generate queries for one topic and one category.
        
        Args:
            topic: Topic dictionary
            category: Query category
            docs_dir: Directory with documents
            num_queries: Number of queries to generate
            client: httpx AsyncClient
            
        Returns:
            List of query dictionaries
        """
        topic_id = topic.get("topic_id")
        topic_name = topic.get("topic_name", "Unknown")
        
        # Get context
        context = self._get_context_for_topic(topic, docs_dir)
        
        # Build prompt
        prompt = self._build_prompt(
            topic_name=topic_name,
            category=category,
            context=context,
            num_queries=num_queries,
        )
        
        # Call LLM
        response_text = await self._call_llm_async(prompt, client)
        
        # Extract queries
        queries = self._extract_queries(response_text, num_queries)
        
        # Build result rows
        rows = []
        for query in queries:
            rows.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "query_type": category,
                "query": query,
            })
        
        return rows

    async def run_for_topic_async(
        self,
        topic: Dict[str, Any],
        docs_dir: Path,
        num_queries_per_type: int,
        client: httpx.AsyncClient,
    ) -> List[Dict[str, Any]]:
        """
        Generate queries for one topic across all categories.
        
        Args:
            topic: Topic dictionary
            docs_dir: Directory with documents
            num_queries_per_type: Queries per category
            client: httpx AsyncClient
            
        Returns:
            List of all query dictionaries for this topic
        """
        semaphore = asyncio.Semaphore(self.max_parallel_categories)
        
        async def process_category(category: str) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self.generate_queries_for_category_async(
                    topic=topic,
                    category=category,
                    docs_dir=docs_dir,
                    num_queries=num_queries_per_type,
                    client=client,
                )
        
        # Process all categories in parallel
        results = await asyncio.gather(
            *[process_category(cat) for cat in self.DEFAULT_TAXONOMY],
            return_exceptions=True,
        )
        
        # Flatten results
        all_rows = []
        for result in results:
            if isinstance(result, Exception):
                continue
            all_rows.extend(result)
        
        return all_rows

    async def run_for_topics_async(
        self,
        topics: List[Dict[str, Any]],
        docs_dir: Path,
        num_queries_per_type: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate queries for multiple topics in parallel.
        
        Args:
            topics: List of topic dictionaries
            docs_dir: Directory with documents
            num_queries_per_type: Queries per category per topic
            
        Returns:
            List of all query dictionaries
        """
        print(f"  Processing {len(topics)} topics in parallel (max {self.max_parallel_topics} concurrent)...")
        
        semaphore = asyncio.Semaphore(self.max_parallel_topics)
        
        # Create shared HTTP client
        async with httpx.AsyncClient() as client:
            
            async def process_one_topic(topic: Dict[str, Any]) -> List[Dict[str, Any]]:
                async with semaphore:
                    return await self.run_for_topic_async(
                        topic=topic,
                        docs_dir=docs_dir,
                        num_queries_per_type=num_queries_per_type,
                        client=client,
                    )
            
            # Process with progress tracking
            tasks = [process_one_topic(t) for t in topics]
            results = []
            
            # Process with tqdm progress bar
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating queries"):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    print(f"\n  ⚠ Error processing topic: {e}")
                    results.append([])
        
        # Flatten all results
        all_rows = []
        errors = 0
        for result in results:
            if isinstance(result, Exception):
                errors += 1
                continue
            all_rows.extend(result)
        
        if errors > 0:
            print(f"  ⚠ {errors} topics had errors")
        
        print(f"  ✓ Generated {len(all_rows)} queries total")
        
        return all_rows


# Synchronous wrapper for compatibility
def run_for_topics_sync(
    topics: List[Dict[str, Any]],
    docs_dir: Path,
    num_queries_per_type: int,
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_parallel_topics: int = 5,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for query generation.
    
    Args:
        topics: List of topic dictionaries
        docs_dir: Directory with documents
        num_queries_per_type: Queries per category per topic
        hf_model: Model ID
        max_parallel_topics: Max parallel topics
        
    Returns:
        List of query dictionaries
    """
    builder = BuildQueries(
        hf_model=hf_model,
        max_parallel_topics=max_parallel_topics,
    )
    
    return asyncio.run(
        builder.run_for_topics_async(
            topics=topics,
            docs_dir=docs_dir,
            num_queries_per_type=num_queries_per_type,
        )
    )
