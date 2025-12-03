"""Checkpoint management for pipeline resumption."""

import pickle
from pathlib import Path
from typing import Any, Optional


class PipelineCheckpoint:
    """Save and load pipeline state for resumption."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_step(self, step_name: str, data: Any):
        """Save data for a pipeline step."""
        path = self.checkpoint_dir / f"{step_name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Checkpoint saved: {step_name}")

    def load_step(self, step_name: str) -> Optional[Any]:
        """Load data for a pipeline step."""
        path = self.checkpoint_dir / f"{step_name}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def has_step(self, step_name: str) -> bool:
        """Check if checkpoint exists for a step."""
        return (self.checkpoint_dir / f"{step_name}.pkl").exists()

    def clear(self):
        """Clear all checkpoints."""
        for f in self.checkpoint_dir.glob("*.pkl"):
            f.unlink()
        print("  ✓ Checkpoints cleared")
