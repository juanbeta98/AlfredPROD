from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class OptimizationAlgorithm(ABC):
    """
    Base interface for all optimization algorithms.

    Contract:
    - Input: canonical pandas DataFrame (parser + validator + solver prep)
    - Output: DataFrame (same rows as input with additional columns) + metrics dict
    - Optional: algorithms may also return a third dict of artifacts (e.g., moves_df)
    """

    name: str = "BASE"

    def __init__(self, params: Dict[str, Any] | None = None):
        self.params: Dict[str, Any] = params or {}

    @abstractmethod
    def solve(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Execute the algorithm.

        Returns:
            results_df : DataFrame of processed labors.
            metrics    : algorithm KPIs and timing info.
            artifacts  : auxiliary outputs (e.g. moves_df, distance_method).
        """
        raise NotImplementedError
