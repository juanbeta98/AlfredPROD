import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import Config
from src.io.artifact_naming import build_artifact_stem, build_run_subdir

logger = logging.getLogger(__name__)

def save_local_output(
    results_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    request_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Persist output locally for development/testing.
    """
    path = _resolve_output_path(
        output_path,
        request_id=request_id,
        run_id=run_id,
    )
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
    request_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Persist the formatted output payload as JSON for local debugging/auditing.
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    payload_path = path / (
        f"{build_artifact_stem('output_payload', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )
    with payload_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    logger.info("local_output_saved artifact=output_payload_json")
    return payload_path


def save_local_validation_outputs(
    invalid_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    request_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """
    Persist local validation artifacts (invalid rows + report).
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    invalid_path = path / (
        f"{build_artifact_stem('data_validation_invalid_rows', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.csv"
    )
    report_path = path / (
        f"{build_artifact_stem('data_validation_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )

    invalid_df.to_csv(invalid_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(
        "validation_outputs_saved generated=data_validation_invalid_rows_csv,data_validation_report_json",
    )


def save_local_solution_validation_outputs(
    issues_df: pd.DataFrame,
    validation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    request_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """
    Persist solution validation artifacts (issues + report).
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    issues_path = path / (
        f"{build_artifact_stem('solution_validation_issues', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.csv"
    )
    report_path = path / (
        f"{build_artifact_stem('solution_validation_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )

    issues_df.to_csv(issues_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    logger.info(
        "solution_validation_outputs_saved generated=issues_csv,validation_report_json",
    )


def save_local_solution_evaluation_report(
    evaluation_report: Dict[str, Any],
    *,
    output_dir: str | Path,
    request_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Persist solution evaluation report as JSON.
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    report_path = path / (
        f"{build_artifact_stem('solution_evaluation_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("solution_evaluation_output_saved artifact=solution_evaluation_report_json")
    return report_path


def save_local_preassigned_reconstruction_reports(
    preassigned_df: pd.DataFrame,
    metrics: Dict[str, Any],
    *,
    output_dir: str | Path,
    request_id: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Path]:
    """
    Persist preassigned reconstruction diagnostics in CSV and JSON formats.
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    csv_path = path / (
        f"{build_artifact_stem('preassigned_reconstruction_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.csv"
    )
    json_path = path / (
        f"{build_artifact_stem('preassigned_reconstruction_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )

    report_columns = [
        "service_id",
        "labor_id",
        "labor_sequence",
        "labor_category",
        "assigned_driver",
        "actual_status",
        "schedule_date",
        "payload_labor_schedule_date",
        "actual_start",
        "actual_end",
        "preassigned_failed",
        "preassigned_failure_reason",
        "preassigned_fallback_applied",
        "preassigned_reconstruction_note",
        "preassigned_action",
        "is_infeasible",
        "is_warning",
        "infeasibility_cause_code",
        "infeasibility_cause_detail",
        "original_assigned_driver",
        "reassignment_candidate",
        "reassignment_priority",
        "preassignment_infeasible_detected",
        "preassignment_infeasibility_cause_code",
        "preassignment_infeasibility_cause_detail",
        "preassignment_warning_detected",
        "preassignment_warning_code",
        "preassignment_warning_detail",
        "warning_code",
        "warning_detail",
        "payload_schedule_reference",
        "computed_arrival",
        "previous_labor_end",
        "feasible_window_end",
        "driver_available_at",
        "computed_travel_min",
        "minutes_late_payload",
        "minutes_after_window",
        "dist_km",
    ]
    available_columns = [col for col in report_columns if col in preassigned_df.columns]
    report_df = preassigned_df.loc[:, available_columns].copy()

    report_df.to_csv(csv_path, index=False)

    payload = {
        "summary": {
            "rows": int(len(report_df)),
            "infeasible_rows": int(report_df.get("is_infeasible", pd.Series(dtype=bool)).fillna(False).sum())
            if not report_df.empty
            else 0,
            "warning_rows": int(report_df.get("is_warning", pd.Series(dtype=bool)).fillna(False).sum())
            if not report_df.empty
            else 0,
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
    request_id: str | None = None,
    run_id: str | None = None,
) -> Dict[str, Path]:
    """
    Persist assignment diagnostics in CSV and JSON formats.
    """
    path = _normalize_output_dir(output_dir)
    path = path / build_run_subdir(run_id)
    path.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    csv_path = path / (
        f"{build_artifact_stem('assignment_diagnostics_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.csv"
    )
    json_path = path / (
        f"{build_artifact_stem('assignment_diagnostics_report', run_id=run_id or 'local', created_at=created_at, request_id=request_id)}.json"
    )

    report_columns = [
        "service_id",
        "labor_id",
        "labor_sequence",
        "assigned_driver",
        "actual_status",
        "actual_start",
        "actual_end",
        "is_infeasible",
        "infeasibility_cause_code",
        "infeasibility_cause_detail",
        "is_warning",
        "warning_code",
        "warning_detail",
        "reassignment_candidate",
        "reassignment_priority",
        "original_assigned_driver",
        "preassignment_infeasible_detected",
        "preassignment_infeasibility_cause_code",
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
    run_id: str | None,
) -> Path:
    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".csv":
            return path
        output_dir = _normalize_output_dir(path)
    else:
        output_dir = _normalize_output_dir(Config.LOCAL_OUTPUT_DIR)

    output_dir = output_dir / build_run_subdir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_artifact_stem(
        "output",
        run_id=run_id or "local",
        request_id=request_id,
    )
    return output_dir / f"{stem}.csv"
