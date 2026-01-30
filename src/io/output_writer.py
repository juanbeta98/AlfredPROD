import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

def save_local_output(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    request_id: str | None = None,
) -> Path:
    """
    Persist output locally for development/testing.
    """
    path = _resolve_output_path(output_path, request_id=request_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(path, index=False)

    logger.info(
        "local_output_saved file=%s rows=%s",
        path.name,
        len(results_df) if hasattr(results_df, "__len__") else None,
    )

    return path


def save_local_validation_outputs(
    invalid_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
) -> None:
    """
    Persist local validation artifacts (invalid rows + report).
    """
    path = _normalize_output_dir(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    invalid_path = path / "invalid_rows.csv"
    report_path = path / "validation_report.json"

    invalid_df.to_csv(invalid_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(
        "validation_outputs_saved invalid_rows=%s report=%s",
        invalid_path.name,
        report_path.name,
    )


def save_local_solution_validation_outputs(
    issues_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
) -> None:
    """
    Persist solution validation artifacts (issues + report).
    """
    path = _normalize_output_dir(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    issues_path = path / "solution_validation_issues.csv"
    report_path = path / "solution_validation_report.json"

    issues_df.to_csv(issues_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(
        "solution_validation_outputs_saved issues=%s report=%s",
        issues_path.name,
        report_path.name,
    )


def _normalize_output_dir(output_dir: str | Path) -> Path:
    """
    Normalize output directory path, handling file-like inputs.
    """
    path = Path(output_dir)
    if path.suffix.lower() in {".json", ".csv"}:
        return path.parent
    return path


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    request_id: str | None,
) -> Path:
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".csv":
            return path
        output_dir = _normalize_output_dir(path)
    else:
        output_dir = _normalize_output_dir(Config.LOCAL_OUTPUT_DIR)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"output_{request_id}_{timestamp}.csv" if request_id else f"output_{timestamp}.csv"
    return output_dir / suffix
