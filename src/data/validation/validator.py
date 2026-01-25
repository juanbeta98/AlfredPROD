import logging
from collections import defaultdict
from typing import Iterable, List, Dict, Any, Tuple, Hashable

import pandas as pd

from .rules.base import ValidationRule

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Applies validation rules to tabular input data and separates
    valid and invalid records while producing an audit trail.
    """

    def __init__(self, rules: Iterable[ValidationRule]):
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

        logger.debug(
            "Starting input validation",
            extra={
                "rows": len(df),
                "rules": [rule.name for rule in self.rules],
            },
        )

        failures_by_row: Dict[Hashable, List[Dict[str, Any]]] = defaultdict(list)


        # --------------------------------------------------
        # 1. Dataset-level validation
        # --------------------------------------------------
        for rule in self.rules:
            df_errors = rule.validate_df(df)

            if not df_errors:
                continue

            logger.warning(
                "Dataset-level validation failed",
                extra={
                    "rule": rule.name,
                    "errors": len(df_errors),
                },
            )

            for row_index, reason in df_errors:
                failures_by_row[row_index].append(
                    {
                        "rule": rule.name,
                        "reason": reason,
                    }
                )

            if rule.blocking:
                logger.error(
                    "Blocking dataset-level rule failed; stopping validation",
                    extra={"rule": rule.name},
                )
                break

        # --------------------------------------------------
        # 2. Row-level validation
        # --------------------------------------------------
        for row_index, row in df.iterrows():
            # Skip rows already invalidated by blocking dataset rules
            if row_index in failures_by_row:
                continue

            for rule in self.rules:
                passed, reason = rule.validate(row)

                if not passed:
                    failures_by_row[row_index].append(
                        {
                            "rule": rule.name,
                            "reason": reason,
                        }
                    )

                    if rule.blocking:
                        break

        # --------------------------------------------------
        # 3. Build outputs
        # --------------------------------------------------
        invalid_indices = list(failures_by_row.keys())
        valid_df = df.drop(index=invalid_indices, errors="ignore")
        invalid_df = self._build_invalid_df(df, failures_by_row)
        report = self._build_report(df, failures_by_row)

        logger.info(
            "validation_completed total=%s valid_rows=%s invalid_rows=%s",
            len(df),
            len(valid_df),
            len(invalid_df),
        )

        return valid_df, invalid_df, report

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _build_invalid_df(
        self,
        df: pd.DataFrame,
        failures_by_row: Dict[Hashable, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """
        Construct a DataFrame containing invalid rows and validation errors.
        """
        if not failures_by_row:
            return self._empty_invalid_df()

        # 1) Create a Series mapping index -> list of errors
        errors_series = pd.Series(failures_by_row, name="_validation_errors")

        # 2) Select invalid rows by index (type-safe via pd.Index)
        invalid_index = pd.Index(errors_series.index)

        # 3) Extract invalid rows and attach errors (aligns by index)
        invalid_df = df.loc[invalid_index].copy()
        invalid_df["_validation_errors"] = errors_series

        # 4) Return as a normal DataFrame (index preserved)
        return invalid_df.reset_index(drop=False)


    def _build_report(
        self,
        df: pd.DataFrame,
        failures_by_row: Dict[Hashable, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
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
        return pd.DataFrame(columns=["_validation_errors"])

    @staticmethod
    def _empty_report() -> Dict[str, Any]:
        return {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "failures_by_rule": {},
        }
