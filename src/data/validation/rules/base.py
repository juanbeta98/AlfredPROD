from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd


class ValidationRule(ABC):
    """
    Abstract base class for all validation rules.

    A rule evaluates a single row and decides whether it passes.
    """

    #: Human-readable rule name (used in reports)
    name: str = "unnamed_rule"

    #: Whether this rule is blocking (fail = record invalid)
    #: Non-blocking rules may be used for warnings in the future
    blocking: bool = True

    def __init__(self, **config: Any):
        """
        Optional configuration for rule behavior.

        Args:
            **config: Rule-specific configuration parameters
        """
        self.config = config

    @abstractmethod
    def validate(self, row: pd.Series) -> Tuple[bool, str | None]:
        """
        Validate a single record.

        Args:
            row: Pandas Series representing a single record

        Returns:
            passed: True if record passes this rule
            reason: Failure reason if not passed, else None
        """
        raise NotImplementedError
    
    def validate_df(self, df: pd.DataFrame):
        """
        Dataset-level validation.

        Args:
            df: Pandas DataFrame representing the dataset

        Returns:
            errors: A list of (row_index, error_message).
        """
        return []

    def metadata(self) -> Dict[str, Any]:
        """
        Optional metadata about the rule instance.
        Useful for auditing and debugging.
        """
        return {
            "name": self.name,
            "blocking": self.blocking,
            "config": self.config,
        }
