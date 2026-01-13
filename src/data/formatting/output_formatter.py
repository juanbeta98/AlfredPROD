from typing import Any, Dict
from datetime import datetime

import pandas as pd


class OutputFormatter:
    """
    Schema-agnostic output formatter.
    Wraps optimization results without enforcing field semantics.
    """

    @staticmethod
    def format(
        results: Any,
        metadata: Dict[str, Any] | None = None,
        request_id: str | None = None,
        status: str = "completed",
    ) -> Dict[str, Any]:
        """
        Formats optimization output into a generic JSON structure.

        Args:
            results: Optimization output (DataFrame, list, dict)
            metadata: Optional metadata from input
            request_id: Optional request identifier
            status: Execution status

        Returns:
            JSON-serializable dict
        """
        if isinstance(results, pd.DataFrame):
            results_payload = results.to_dict(orient="records")
        else:
            results_payload = results

        return {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "results": results_payload,
            "metadata": metadata,
        }
