"""Pipeline metrics collection and reporting."""

from typing import Dict, Any, List
from datetime import datetime
import numpy as np


class PipelineMetrics:
    """Track and report pipeline metrics."""

    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            "run_timestamp": self.start_time.isoformat(),
            "stages": {},
            "timing": {},
        }

    def record_stage(self, stage_name: str, data: Dict[str, Any]):
        """Record metrics for a pipeline stage."""
        self.metrics["stages"][stage_name] = data

    def record_timing(self, stage_name: str, duration_seconds: float):
        """Record timing for a stage."""
        self.metrics["timing"][stage_name] = duration_seconds

    def get_quality_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Get statistical distribution of scores."""
        if not scores:
            return {}
        
        scores_arr = np.array(scores)
        return {
            "count": len(scores),
            "min": float(scores_arr.min()),
            "max": float(scores_arr.max()),
            "mean": float(scores_arr.mean()),
            "median": float(np.median(scores_arr)),
            "std": float(scores_arr.std()),
            "percentiles": {
                "p25": float(np.percentile(scores_arr, 25)),
                "p50": float(np.percentile(scores_arr, 50)),
                "p75": float(np.percentile(scores_arr, 75)),
                "p90": float(np.percentile(scores_arr, 90)),
                "p95": float(np.percentile(scores_arr, 95)),
            },
        }

    def print_summary(self):
        """Print a formatted summary of metrics."""
        print("\n" + "=" * 60)
        print("Pipeline Metrics Summary")
        print("=" * 60)
        
        for stage, data in self.metrics["stages"].items():
            print(f"\n{stage}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        print("\nTiming:")
        total_time = 0
        for stage, duration in self.metrics["timing"].items():
            print(f"  {stage}: {duration:.1f}s")
            total_time += duration
        print(f"  TOTAL: {total_time:.1f}s")

    def to_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        end_time = datetime.now()
        self.metrics["end_timestamp"] = end_time.isoformat()
        self.metrics["total_duration_seconds"] = (end_time - self.start_time).total_seconds()
        return self.metrics
