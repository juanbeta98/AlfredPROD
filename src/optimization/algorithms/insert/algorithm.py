import pandas as pd
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from src.optimization.algorithms.base import OptimizationAlgorithm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertAlgoConfig:
    """
    Configuration for INSERT algorithm.
    Keep this minimal and expand as you migrate experiment knobs.
    """
    optimization_obj: Optional[str] = None
    distance_method: Optional[str] = None
    n_processes: Optional[int] = None


class InsertAlgorithm(OptimizationAlgorithm):
    """
    INSERT algorithm.

    Expected behavior:
    - Input: parsed/validated DataFrame in the pipeline schema
    - Output: DataFrame with additional columns (same rows as input, plus decisions)
    - Metrics: algorithm KPIs and timing info
    """

    name = "INSERT"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        params = params or {}
        self.config = InsertAlgoConfig(
            optimization_obj=params.get("optimization_obj"),
            distance_method=params.get("distance_method"),
            n_processes=params.get("n_processes"),
        )

    def solve(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute INSERT algorithm.

        Returns:
            results_df: DataFrame (same base as df, plus added columns)
            metrics: algorithm metrics
        """
        t0 = perf_counter()

        self._validate_preconditions(df)

        # TODO: implement algorithm-specific preparation and execution.
        raise NotImplementedError

    def _validate_preconditions(self, df: pd.DataFrame) -> None:
        required_cols = {"city", "schedule_date", "labor_id", "service_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"INSERT algorithm missing required columns: {sorted(missing)}")
