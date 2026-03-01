import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import Config
from src.io.artifact_naming import build_run_subdir, write_run_manifest

logger = logging.getLogger(__name__)


def _ensure_run_dir(base: Path, run_id: str | None) -> Path:
    """Create the run subdir and write its run.json manifest (idempotent)."""
    path = base / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)
    write_run_manifest(path, run_id or "local")
    return path


def _subdir(run_dir: Path, name: str) -> Path:
    """Create and return a named subfolder inside the run directory."""
    path = run_dir / name
    path.mkdir(exist_ok=True)
    return path


def save_local_output(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist the main assignment output CSV."""
    path = _resolve_output_path(output_path, run_id=run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    logger.info(
        "local_output_saved artifact=output_csv rows=%s",
        len(results_df) if hasattr(results_df, "__len__") else None,
    )
    return path


def save_local_output_payload(
    payload: Any,
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Persist the formatted output payload as JSON."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    out_dir = _subdir(run_dir, "output")
    payload_path = out_dir / "output_payload.json"
    with payload_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    logger.info("local_output_saved artifact=output_payload_json")
    return payload_path


def save_local_validation_outputs(
    invalid_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> None:
    """Persist input validation artifacts (invalid rows + report)."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    val_dir = _subdir(run_dir, "validation")
    invalid_df.to_csv(val_dir / "data_validation_invalid_rows.csv", index=False)
    with (val_dir / "data_validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(
        "validation_outputs_saved generated=data_validation_invalid_rows_csv,data_validation_report_json",
    )


def save_local_solution_validation_outputs(
    issues_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> None:
    """Persist solution validation artifacts (issues + report)."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    val_dir = _subdir(run_dir, "validation")
    issues_df.to_csv(val_dir / "solution_validation_issues.csv", index=False)
    with (val_dir / "solution_validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(
        "solution_validation_outputs_saved generated=issues_csv,validation_report_json",
    )


def save_local_solution_evaluation_report(
    evaluation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Persist solution evaluation report as JSON."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    diag_dir = _subdir(run_dir, "diagnostics")
    report_path = diag_dir / "solution_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info("solution_evaluation_output_saved artifact=solution_evaluation_report_json")
    return report_path


def save_local_preassigned_reconstruction_reports(
    preassigned_df: pd.DataFrame,
    metrics: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Dict[str, Path]:
    """Persist preassigned reconstruction diagnostics in CSV and JSON formats."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    diag_dir = _subdir(run_dir, "diagnostics")
    csv_path = diag_dir / "preassigned_reconstruction_report.csv"
    json_path = diag_dir / "preassigned_reconstruction_report.json"

    report_columns = [
        "service_id", "labor_id", "labor_sequence", "labor_category",
        "assigned_driver", "actual_status", "schedule_date",
        "payload_labor_schedule_date", "actual_start", "actual_end",
        "preassigned_failed", "preassigned_failure_reason",
        "preassigned_fallback_applied", "preassigned_reconstruction_note",
        "preassigned_action", "is_infeasible", "is_warning",
        "infeasibility_cause_code", "infeasibility_cause_detail",
        "original_assigned_driver", "reassignment_candidate", "reassignment_priority",
        "preassignment_infeasible_detected", "preassignment_infeasibility_cause_code",
        "preassignment_infeasibility_cause_detail", "preassignment_warning_detected",
        "preassignment_warning_code", "preassignment_warning_detail",
        "warning_code", "warning_detail", "payload_schedule_reference",
        "computed_arrival", "previous_labor_end", "feasible_window_end",
        "driver_available_at", "computed_travel_min", "minutes_late_payload",
        "minutes_after_window", "dist_km",
    ]
    available_columns = [col for col in report_columns if col in preassigned_df.columns]
    report_df = preassigned_df.loc[:, available_columns].copy()
    report_df.to_csv(csv_path, index=False)

    payload = {
        "summary": {
            "rows": int(len(report_df)),
            "infeasible_rows": int(report_df.get("is_infeasible", pd.Series(dtype=bool)).fillna(False).sum())
            if not report_df.empty else 0,
            "warning_rows": int(report_df.get("is_warning", pd.Series(dtype=bool)).fillna(False).sum())
            if not report_df.empty else 0,
        },
        "metrics": metrics or {},
        "rows": report_df.where(pd.notna(report_df), None).to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "preassigned_reconstruction_reports_saved artifacts=preassigned_report_csv,preassigned_report_json rows=%s",
        len(report_df),
    )
    return {"csv": csv_path, "json": json_path}


def save_local_assignment_diagnostics_report(
    results_df: pd.DataFrame,
    diagnostics_metrics: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Dict[str, Path]:
    """Persist assignment diagnostics in CSV and JSON formats."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    diag_dir = _subdir(run_dir, "diagnostics")
    csv_path = diag_dir / "assignment_diagnostics_report.csv"
    json_path = diag_dir / "assignment_diagnostics_report.json"

    report_columns = [
        "service_id", "labor_id", "labor_sequence", "assigned_driver",
        "actual_status", "actual_start", "actual_end", "is_infeasible",
        "infeasibility_cause_code", "infeasibility_cause_detail", "is_warning",
        "warning_code", "warning_detail", "reassignment_candidate",
        "reassignment_priority", "original_assigned_driver",
        "preassignment_infeasible_detected", "preassignment_infeasibility_cause_code",
        "overtime_minutes",
    ]
    available_columns = [col for col in report_columns if col in results_df.columns]
    report_df = results_df.loc[:, available_columns].copy() if available_columns else pd.DataFrame()
    report_df.to_csv(csv_path, index=False)

    payload = {
        "summary": diagnostics_metrics or {},
        "rows": report_df.where(pd.notna(report_df), None).to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "assignment_diagnostics_report_saved artifacts=assignment_diagnostics_csv,assignment_diagnostics_json rows=%s",
        len(report_df),
    )
    return {"csv": csv_path, "json": json_path}


def save_local_warnings_report(
    warnings_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """Persist the high-level run warnings/errors summary as warnings.json."""
    run_dir = _ensure_run_dir(_normalize_output_dir(output_dir), run_id)
    path = run_dir / "warnings.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(warnings_report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("warnings_report_saved artifact=warnings_json")
    return path


def _normalize_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.suffix.lower() in {".json", ".csv"}:
        return path.parent
    return path


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    run_id: str | None,
) -> Path:
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".csv":
            return path
        base = _normalize_output_dir(path)
    else:
        base = _normalize_output_dir(Config.RUNS_DIR)

    run_dir = _ensure_run_dir(base, run_id)
    out_dir = _subdir(run_dir, "output")
    return out_dir / "output.csv"
