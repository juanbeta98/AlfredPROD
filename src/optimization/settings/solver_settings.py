from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .model_params import ModelParams
from .master_data import MasterDataParams


@dataclass(frozen=True)
class OptimizationSettings:
    algorithm: str = "OFFLINE"

    # shared hyperparameters
    alpha: float = 0.3
    # max_iterations: int = 1000
    max_iterations: dict[str, int] = field(
        default_factory=lambda: {
            "149": 1000,
            "1": 600,
            "1004": 600,
            "126": 150,
            "150": 150,
            "844": 150,
            "830": 150,
        }
    )

    # execution controls
    n_processes: Optional[int] = None

    # shared model parameters (timings, speeds, etc.)
    model_params: ModelParams = field(default_factory=ModelParams)
    # shared master data locations (csv/parquet)
    master_data: MasterDataParams = field(default_factory=MasterDataParams)

    # per-algorithm overrides (optional)
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def for_algorithm(self, name: str) -> Dict[str, Any]:
        """
        Returns settings for algorithm `name`, applying overrides if present.
        """
        base = {
            "alpha": self.alpha,
            "max_iterations": self.max_iterations,
            "n_processes": self.n_processes,
            "model_params": self.model_params,
            "master_data": self.master_data,
        }
        algo_over = self.overrides.get(name.upper(), {})
        return {**base, **algo_over}
