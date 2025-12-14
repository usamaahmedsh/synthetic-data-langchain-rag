import random
from typing import List, Dict, Any, Set
from collections import defaultdict


def normalize_query(q: str) -> str:
    """Normalize for deduplication."""
    import re
    q = q.strip().strip("-•\"' ").rstrip(".?!")
    q = re.sub(r"\s+", " ", q)
    return q.lower()

def ensure_semantic_diversity(
    queries: List[Dict[str, Any]],
    target_count: int,
    model,
    diversity_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Select queries that maximize semantic diversity.
    
    Uses MMR (Maximal Marginal Relevance) approach.
    
    Args:
        queries: List of query dicts with scores
        target_count: Target number of queries
        model: SentenceTransformer model for embeddings
        diversity_weight: Weight for diversity (0-1), higher = more diverse
        
    Returns:
        Diverse subset of queries
    """
    if len(queries) <= target_count:
        return queries
    
    # Get embeddings
    texts = [q["query"] for q in queries]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    # Greedy selection with MMR
    selected_indices = []
    remaining_indices = list(range(len(queries)))
    
    # Start with highest scored query
    scores = [q.get("total_score", 0) for q in queries]
    first_idx = int(np.argmax(scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select most diverse + high quality
    while len(selected_indices) < target_count and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            # Quality score
            quality = queries[idx].get("total_score", 0)
            
            # Diversity score (min similarity to selected)
            similarities = cosine_similarity(
                embeddings[idx:idx+1],
                embeddings[selected_indices]
            )[0]
            max_sim = similarities.max()
            diversity = 1 - max_sim
            
            # Combined score (MMR)
            mmr_score = (
                (1 - diversity_weight) * quality +
                diversity_weight * diversity
            )
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return [queries[i] for i in selected_indices]



class RejectionSampler:
    """
    Rejection sampling / filtering to select final queries.
    
    Now uses dynamic percentile-based thresholding:
    - Starts with top percentile queries
    - Gradually lowers threshold until global target is met
    - Ensures diversity across topics and categories

    Additionally supports lexical + semantic diversity gates if rows contain:
      - lex_div_score: float in [0,1]
      - sem_div_score: float in [0,1]
    """

    def __init__(
        self,
        min_per_topic_per_type: int = 1,
        enforce_dedup: bool = True,
        random_seed: int = 42,
        initial_percentile: float = 0.90,  # Start with top 10%
        percentile_step: float = 0.05,     # Lower by 5% each iteration
        min_percentile: float = 0.50,      # Don't go below 50th percentile
        lexical_min_div: float = 0.30,     # NEW: min allowed lex_div_score
        semantic_min_div: float = 0.15,    # NEW: min allowed sem_div_score (≈ 1 - 0.85 cos)
    ) -> None:
        self.min_per_topic_per_type = min_per_topic_per_type
        self.enforce_dedup = enforce_dedup
        self.random_seed = random_seed
        self.initial_percentile = initial_percentile
        self.percentile_step = percentile_step
        self.min_percentile = min_percentile
        self.lexical_min_div = lexical_min_div
        self.semantic_min_div = semantic_min_div
        self.rng = random.Random(random_seed)

    def _compute_percentile_threshold(
        self, 
        scores: List[float], 
        percentile: float
    ) -> float:
        """Compute score threshold at given percentile."""
        if not scores:
            return 0.0
        sorted_scores = sorted(scores)
        idx = int(len(sorted_scores) * percentile)
        idx = max(0, min(idx, len(sorted_scores) - 1))
        return sorted_scores[idx]

    def run(
        self,
        scored_rows: List[Dict[str, Any]],
        target_per_type: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Select queries using dynamic percentile-based thresholding + diversity.
        
        Args:
            scored_rows: All scored queries
            target_per_type: Dict mapping query_type to desired count
        
        Returns:
            List of selected queries meeting the target
        """
        global_target = sum(target_per_type.values())
        
        print(f"\n{'='*60}")
        print("Rejection Sampling (Dynamic Percentile-Based)")
        print(f"{'='*60}")
        print(f"Total scored queries: {len(scored_rows)}")
        print(f"Global target: {global_target}")
        print(f"Per-type targets: {target_per_type}")
        print()

        # Filter out queries with errors or missing scores
        valid_rows = [
            r for r in scored_rows
            if r.get("total_score") is not None
            and r.get("error") is None
            and isinstance(r.get("total_score"), (int, float))
        ]
        
        print(f"Valid scored queries: {len(valid_rows)}")
        
        if not valid_rows:
            print("[WARN] No valid scored queries available!")
            return []

        # Collect all scores for percentile calculation
        all_scores = [r["total_score"] for r in valid_rows]
        
        if not all_scores:
            print("[WARN] No scores available!")
            return []

        # Group by type
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in valid_rows:
            qtype = r.get("query_type", "unknown")
            by_type[qtype].append(r)

        # Sort each type by score (descending)
        for qtype in by_type:
            by_type[qtype].sort(key=lambda x: x["total_score"], reverse=True)

        # Dynamic percentile-based selection
        selected: List[Dict[str, Any]] = []
        seen_normalized: Set[str] = set()
        current_percentile = self.initial_percentile

        print(f"Starting percentile: {current_percentile:.1%}")
        print()

        while len(selected) < global_target and current_percentile >= self.min_percentile:
            # Compute threshold for current percentile
            threshold = self._compute_percentile_threshold(all_scores, current_percentile)
            
            print(f"Trying percentile {current_percentile:.1%} (threshold: {threshold:.3f})...")

            # Try to select queries above threshold
            iteration_selected = 0
            type_counts = defaultdict(int)

            for qtype, target_count in target_per_type.items():
                if qtype not in by_type:
                    continue

                # How many more do we need for this type?
                current_count = sum(1 for r in selected if r.get("query_type") == qtype)
                needed = target_count - current_count

                if needed <= 0:
                    continue

                # Get candidates above threshold
                candidates = [
                    r for r in by_type[qtype]
                    if r["total_score"] >= threshold
                ]

                # Select up to 'needed' queries
                for r in candidates:
                    if current_count >= target_count:
                        break

                    # Check if already selected (exact match)
                    if any(s.get("query") == r.get("query") for s in selected):
                        continue

                    # Deduplication check
                    if self.enforce_dedup:
                        norm = normalize_query(r.get("query", ""))
                        if norm in seen_normalized:
                            continue

                    # NEW: diversity gates (if scores present)
                    lex_div = r.get("lex_div_score")
                    sem_div = r.get("sem_div_score")

                    if isinstance(lex_div, (int, float)) and lex_div < self.lexical_min_div:
                        continue

                    if isinstance(sem_div, (int, float)) and sem_div < self.semantic_min_div:
                        continue

                    # Accept
                    if self.enforce_dedup:
                        seen_normalized.add(norm)

                    selected.append(r)
                    iteration_selected += 1
                    type_counts[qtype] += 1
                    current_count += 1

            print(f"  Selected {iteration_selected} queries (total: {len(selected)}/{global_target})")
            if iteration_selected > 0:
                for qtype, count in sorted(type_counts.items()):
                    print(f"    {qtype}: +{count}")

            # If we hit target, stop
            if len(selected) >= global_target:
                break

            # Lower percentile for next iteration
            current_percentile -= self.percentile_step
            print()

        # Fill remaining slots if needed (from any remaining high-scoring queries)
        if len(selected) < global_target:
            print(f"\nFilling remaining {global_target - len(selected)} slots from all queries...")
            
            # Get all remaining queries not yet selected
            remaining = [
                r for r in valid_rows
                if not any(s.get("query") == r.get("query") for s in selected)
            ]
            remaining.sort(key=lambda x: x["total_score"], reverse=True)

            for r in remaining:
                if len(selected) >= global_target:
                    break

                if self.enforce_dedup:
                    norm = normalize_query(r.get("query", ""))
                    if norm in seen_normalized:
                        continue

                lex_div = r.get("lex_div_score")
                sem_div = r.get("sem_div_score")

                if isinstance(lex_div, (int, float)) and lex_div < self.lexical_min_div:
                    continue
                if isinstance(sem_div, (int, float)) and sem_div < self.semantic_min_div:
                    continue

                if self.enforce_dedup:
                    seen_normalized.add(norm)

                selected.append(r)

        print(f"\n{'='*60}")
        print(f"✓ Final selection: {len(selected)} queries")
        print(f"{'='*60}")

        # Show final distribution
        final_counts = defaultdict(int)
        for r in selected:
            final_counts[r.get("query_type", "unknown")] += 1

        print("\nFinal distribution by category:")
        for qtype in sorted(final_counts.keys()):
            target = target_per_type.get(qtype, 0)
            actual = final_counts[qtype]
            print(f"  {qtype:30s}: {actual:3d} / {target:3d} target")

        # Show score statistics
        if selected:
            scores = [r["total_score"] for r in selected]
            print(f"\nScore statistics:")
            print(f"  Min:  {min(scores):.3f}")
            print(f"  Max:  {max(scores):.3f}")
            print(f"  Mean: {sum(scores)/len(scores):.3f}")

        return selected
