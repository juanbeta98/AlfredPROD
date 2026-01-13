import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class InputParser:
    """
    Schema-agnostic input parser.
    Converts arbitrary JSON payloads into tabular structures
    without assuming a fixed contract.
    """

    @staticmethod
    def parse(json_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Attempts to extract a list of records from a JSON payload
        and returns a DataFrame plus metadata.

        Args:
            json_data: Raw JSON input from API or file

        Returns:
            df: Pandas DataFrame with flattened records
            metadata: Remaining non-tabular JSON data
        """
        if not isinstance(json_data, dict):
            raise ValueError("Input data must be a JSON object")

        records = InputParser._find_record_list(json_data)

        if records is None:
            logger.warning("No list of records found in JSON payload")
            return pd.DataFrame(), {"raw": json_data}

        logger.info(f"Extracted {len(records)} records from JSON payload")

        df = pd.json_normalize(records)

        metadata = {
            k: v for k, v in json_data.items()
            if v is not records
        }

        return df, metadata

    @staticmethod
    def _find_record_list(payload: Dict[str, Any]) -> List[Dict[str, Any]] | None:
        """
        Heuristically find the first list of dicts in the JSON payload.
        """
        for key, value in payload.items():
            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, dict) for item in value)
            ):
                logger.debug(f"Using '{key}' as record list")
                return value

        return None
