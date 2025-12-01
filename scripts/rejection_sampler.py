# rejection_sampler.py

import random
from typing import List, Dict, Any, Set
from collections import defaultdict


def normalize_query(q: str) -> str:
    """Normalize for deduplication."""
    import re
    q = q.strip().strip("-•\"' ").rstrip(".?!")
    q = re.sub(r"\s+", " ", q)
    return q.lower()


class RejectionSampler:
    """
    Rejection sampling / filtering to select final queries.
    
    Now uses dynamic percentile-based thresholding:
    - Starts with top percentile queries
    - Gradually lowers threshold until global target is met
    - Ensures diversity across topics and categories
    """

    def __init__(
        self,
        min_per_topic_per_type: int = 1,
        enforce_dedup: bool = True,
        random_seed: int = 42,
        initial_percentile: float = 0.90,  # Start with top 10%
        percentile_step: float = 0.05,     # Lower by 5% each iteration
        min_percentile: float = 0.50,      # Don't go below 50th percentile
    ) -> None:
        self.min_per_topic_per_type = min_per_topic_per_type
        self.enforce_dedup = enforce_dedup
        self.random_seed = random_seed
        self.initial_percentile = initial_percentile
        self.percentile_step = percentile_step
        self.min_percentile = min_percentile
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
        Select queries using dynamic percentile-based thresholding.
        
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

                    # Check if already selected
                    if any(s.get("query") == r.get("query") for s in selected):
                        continue

                    # Deduplication check
                    if self.enforce_dedup:
                        norm = normalize_query(r.get("query", ""))
                        if norm in seen_normalized:
                            continue
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

                # Dedup check
                if self.enforce_dedup:
                    norm = normalize_query(r.get("query", ""))
                    if norm in seen_normalized:
                        continue
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
