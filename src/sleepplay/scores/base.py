from typing import Protocol

import numpy as np


class ScoreAlgorithm(Protocol):
    def score(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
        """Return a motion score for two adjacent sampled frames."""
        ...
