"""Simple profiler for timing operations."""

import time
from contextlib import contextmanager
from typing import Dict


class SimpleProfiler:
    """Track execution time of operations."""

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for timing an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.timings[name] = self.timings.get(name, 0) + duration

    def print_report(self):
        """Print timing report."""
        print("\n" + "=" * 60)
        print("Performance Profile")
        print("=" * 60)
        
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        total_time = sum(self.timings.values())
        
        for name, duration in sorted_timings:
            pct = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{name:40s}: {duration:6.1f}s ({pct:5.1f}%)")
        
        print(f"{'TOTAL':40s}: {total_time:6.1f}s")
