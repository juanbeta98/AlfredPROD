from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Problem:
    """
    Problem definition that preserves the current algorithm contract:
    algorithms consume a pandas DataFrame in the parser-produced schema.

    This class adds:
    - run metadata (ids, timestamps)
    - request context (department, request_id)
    - precomputed artifacts (distance matrices, mappings, etc.)
    """
    df: pd.DataFrame

    # Traceability / observability
    run_id: str
    solver_version: str
    generated_at: datetime

    # Optional API context
    request_id: Optional[str] = None
    department: Optional[int] = None

    # Derived artifacts for algorithms (optional)
    context: Dict[str, Any] = field(default_factory=dict)

    # Build diagnostics (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def row_count(self) -> int:
        return len(self.df)

    def service_count(self) -> Optional[int]:
        if "service_id" in self.df.columns:
            return int(self.df["service_id"].nunique(dropna=True))
        return None

    def labor_count(self) -> Optional[int]:
        if "labor_id" in self.df.columns:
            return int(self.df["labor_id"].nunique(dropna=True))
        return None
