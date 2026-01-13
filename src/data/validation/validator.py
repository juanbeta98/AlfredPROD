import logging
from collections import defaultdict
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd

from .rules.base import ValidationRule

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Applies a set of validation rules to tabular input data and separates
    valid and invalid records while producing an audit trail.
    """

    def __init__(self, rules: Iterable[ValidationRule]):
        """
        Args:
            rules: Iterable of ValidationRule instances
        """
        self.rules: List[ValidationRule] = list(rules)

        if not self.rules:
            logger.warning("InputValidator initialized with no validation rules")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def validate(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Validate a DataFrame against configured rules.

        Args:
            df: Input DataFrame

        Returns:
            valid_df: Rows that passed all rules
            invalid_df: Rows that failed one or more rules, with reasons
            report: Aggregated validation metrics
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("InputValidator expects a pandas DataFrame")

        if df.empty:
            logger.warning("Empty DataFrame received for validation")
            return df.copy(), self._empty_invalid_df(), self._empty_report()

        logger.info(
            "Starting input validation",
            extra={
                "rows": len(df),
                "rules": [rule.name for rule in self.rules],
            },
        )

        failures_by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for row_index, row in df.iterrows():
            for rule in self.rules:
                passed, reason = rule.validate(row)

                if not passed:
                    failures_by_row[row_index].append(
                        {
                            "rule": rule.name,
                            "reason": reason,
                        }
                    )

        valid_df = df.drop(index=failures_by_row.keys())
        invalid_df = self._build_invalid_df(df, failures_by_row)
        report = self._build_report(df, failures_by_row)

        logger.info(
            "Validation completed",
            extra={
                "total": len(df),
                "valid": len(valid_df),
                "invalid": len(invalid_df),
            },
        )

        return valid_df, invalid_df, report

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _build_invalid_df(
        self,
        df: pd.DataFrame,
        failures_by_row: Dict[int, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        Construct a DataFrame containing invalid rows and validation errors.
        """
        if not failures_by_row:
            return self._empty_invalid_df()

        records = []
        for row_index, failures in failures_by_row.items():
            record = df.loc[row_index].to_dict()
            record["_validation_errors"] = failures
            records.append(record)

        return pd.DataFrame(records)

    def _build_report(
        self,
        df: pd.DataFrame,
        failures_by_row: Dict[int, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Build aggregated validation metrics.
        """
        failures_by_rule = defaultdict(int)
        for failures in failures_by_row.values():
            for failure in failures:
                failures_by_rule[failure["rule"]] += 1

        return {
            "total_records": len(df),
            "valid_records": len(df) - len(failures_by_row),
            "invalid_records": len(failures_by_row),
            "failures_by_rule": dict(failures_by_rule),
        }

    @staticmethod
    def _empty_invalid_df() -> pd.DataFrame:
        """
        Return an empty invalid DataFrame with expected structure.
        """
        return pd.DataFrame(columns=["_validation_errors"])

    @staticmethod
    def _empty_report() -> Dict[str, Any]:
        """
        Return an empty validation report.
        """
        return {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "failures_by_rule": {},
        }
