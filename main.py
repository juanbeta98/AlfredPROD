import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config
from src.datetime_utils import utc_to_colombia_series
from src.integration.client import ALFREDAPIClient
from src.integration.sender import ResultSender
from src.io.input_loader import load_local_input
from src.io.request_loader import apply_request_filters, build_settings_from_request, load_request
from src.io.output_writer import (
    save_local_assignment_diagnostics_report,
    save_local_output,
    save_local_output_payload,
    save_local_preassigned_reconstruction_reports,
    save_local_validation_outputs,
    save_local_solution_evaluation_report,
    save_local_solution_validation_outputs,
)
from src.data.loading.driver_directory_loader import load_driver_directory_df
from src.data.loading.master_data_loader import MasterData, load_master_data
from src.data.parsing.input_parser import InputParser
from src.data.parsing.driver_directory_parser import DriverDirectoryParser
from src.data.validation.validator import InputValidator
from src.data.validation.rules.generic import RequiredFieldRule, NonEmptyRowRule, UniqueLaborIdRule
from src.data.validation.rules.domain import (
    CreatedBeforeScheduleRule,
    ValidDepartmentsOnly,
    ValidLocationResolutionStatus,
)
from src.data.formatting.output_formatter import OutputFormatter
from src.optimization.solver import OptimizationSolver
from src.optimization.settings.solver_settings import OptimizationSettings
from src.optimization.evaluation.solution_evaluator import evaluate_solution
from src.optimization.validation.solution_validator import validate_solution
from src.logging_utils import set_run_id, setup_logging_context
from src.pipeline_helpers import (
    _build_intermediate_export_dir,
    _configure_fallback_logging,
    _export_intermediate_dataframe,
    _format_log_message,
    log_info,
    log_step,
    report_failure_if_possible,
    set_pipeline_logger,
)

logger = logging.getLogger(__name__)
set_pipeline_logger(logger)

DEFAULT_KEEP_PAYLOAD_ASSIGNMENT = False


def _filter_df_by_department_code(
    df: pd.DataFrame,
    *,
    department: int | str | None,
    dataset_name: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if department is None:
        return df

    department_code = str(department).strip()
    if not department_code:
        return df

    if "department_code" not in df.columns:
        log_info(
            "department_filter_skipped_missing_column",
            dataset=dataset_name,
            expected_column="department_code",
        )
        return df

    before_rows = len(df)
    department_series = df["department_code"].astype("string").str.strip()
    filtered_df = df.loc[department_series.eq(department_code).fillna(False)].copy()

    log_info(
        "department_filter_applied",
        dataset=dataset_name,
        department=department_code,
        rows_in=before_rows,
        rows_out=len(filtered_df),
    )
    return filtered_df


def _parse_bool_flag(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _resolve_keep_payload_assignment(request_payload: Any) -> bool:
    if request_payload is None:
        return DEFAULT_KEEP_PAYLOAD_ASSIGNMENT

    raw = getattr(request_payload, "raw", {}) or {}
    if not isinstance(raw, dict):
        return DEFAULT_KEEP_PAYLOAD_ASSIGNMENT

    algo = raw.get("algorithm") if isinstance(raw.get("algorithm"), dict) else {}
    algo_params = algo.get("params") if isinstance(algo.get("params"), dict) else {}
    preassignment = raw.get("preassignment") if isinstance(raw.get("preassignment"), dict) else {}

    for candidate in (
        raw.get("keep_payload_assignment"),
        raw.get("keep_payload_assingment"),
        preassignment.get("keep_payload_assignment"),
        preassignment.get("keep_payload_assingment"),
        algo_params.get("keep_payload_assignment"),
        algo_params.get("keep_payload_assingment"),
    ):
        if candidate is None:
            continue
        return _parse_bool_flag(candidate, default=DEFAULT_KEEP_PAYLOAD_ASSIGNMENT)

    return DEFAULT_KEEP_PAYLOAD_ASSIGNMENT


def _prepare_service_rows_for_reassignment(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rows = df.copy()
    if "original_assigned_driver" not in rows.columns:
        rows["original_assigned_driver"] = rows.get("assigned_driver")
    else:
        missing_original = rows["original_assigned_driver"].isna()
        rows.loc[missing_original, "original_assigned_driver"] = rows.loc[missing_original, "assigned_driver"]

    rows["preassignment_infeasible_detected"] = rows.get("is_infeasible", False).fillna(False)
    rows["preassignment_infeasibility_cause_code"] = rows.get("infeasibility_cause_code")
    rows["preassignment_infeasibility_cause_detail"] = rows.get("infeasibility_cause_detail")
    rows["preassignment_warning_detected"] = rows.get("is_warning", False).fillna(False)
    rows["preassignment_warning_code"] = rows.get("warning_code")
    rows["preassignment_warning_detail"] = rows.get("warning_detail")

    rows["reassignment_candidate"] = True
    rows["reassignment_priority"] = 1
    rows["assigned_driver"] = pd.NA
    rows["actual_start"] = pd.NaT
    rows["actual_end"] = pd.NaT
    rows["actual_status"] = pd.NA
    rows["is_infeasible"] = False
    rows["infeasibility_cause_code"] = None
    rows["infeasibility_cause_detail"] = None
    rows["is_warning"] = False
    rows["warning_code"] = None
    rows["warning_detail"] = None
    return rows


def _finalize_assignment_diagnostics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()
    default_columns = {
        "is_infeasible": False,
        "infeasibility_cause_code": None,
        "infeasibility_cause_detail": None,
        "reassignment_candidate": False,
    }
    for col, default_value in default_columns.items():
        if col not in df.columns:
            df[col] = default_value

    failed_mask = df.get("actual_status", pd.Series(index=df.index, dtype="object")).astype("string").str.upper().eq("FAILED")
    reassignment_candidate_mask = df["reassignment_candidate"].fillna(False).astype(bool)
    reassignment_failed_mask = failed_mask & reassignment_candidate_mask
    generic_failed_mask = failed_mask & ~reassignment_candidate_mask

    df.loc[failed_mask, "is_infeasible"] = True

    reassignment_code_missing = reassignment_failed_mask & (
        df["infeasibility_cause_code"].isna()
        | df["infeasibility_cause_code"].astype("string").str.strip().eq("")
    )
    df.loc[reassignment_code_missing, "infeasibility_cause_code"] = "reassignment_failed_unassigned"
    df.loc[reassignment_code_missing, "infeasibility_cause_detail"] = (
        "Service was moved to reassignment queue but no fully feasible assignment was found."
    )

    generic_code_missing = generic_failed_mask & (
        df["infeasibility_cause_code"].isna()
        | df["infeasibility_cause_code"].astype("string").str.strip().eq("")
    )
    df.loc[generic_code_missing, "infeasibility_cause_code"] = "assignment_failed_unassigned"
    df.loc[generic_code_missing, "infeasibility_cause_detail"] = (
        "No feasible driver assignment was found for this labor/service."
    )

    reassigned_success_mask = reassignment_candidate_mask & ~failed_mask
    df.loc[reassigned_success_mask, "is_infeasible"] = False
    solver_failure_code_mask = reassigned_success_mask & df["infeasibility_cause_code"].astype("string").isin(
        ["assignment_failed_unassigned", "reassignment_failed_unassigned"]
    )
    df.loc[solver_failure_code_mask, "infeasibility_cause_code"] = None
    df.loc[solver_failure_code_mask, "infeasibility_cause_detail"] = None

    return df


def _build_assignment_diagnostics_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    if results_df is None or results_df.empty:
        return {
            "rows_total": 0,
            "rows_infeasible": 0,
            "rows_warning": 0,
            "rows_unassigned": 0,
            "rows_reassignment_candidate": 0,
            "rows_reassignment_failed": 0,
            "rows_reassignment_success": 0,
            "infeasibility_causes": {},
            "warning_causes": {},
            "actual_status_counts": {},
        }

    df = results_df.copy()
    status_series = (
        df.get("actual_status", pd.Series(index=df.index, dtype="object"))
        .astype("string")
        .str.upper()
    )
    is_failed = status_series.eq("FAILED")
    reassignment_candidate = (
        df.get("reassignment_candidate", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    is_infeasible = (
        df.get("is_infeasible", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    is_warning = (
        df.get("is_warning", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    assigned_driver = df.get("assigned_driver", pd.Series(index=df.index, dtype="object"))
    rows_unassigned = int((assigned_driver.isna() | assigned_driver.astype("string").str.strip().isin(["", "nan", "None"])).sum())

    def _counts_to_dict(series: pd.Series) -> Dict[str, int]:
        if series is None or series.empty:
            return {}
        cleaned = series.dropna().astype("string").str.strip()
        cleaned = cleaned[cleaned.ne("") & cleaned.ne("<NA>") & cleaned.ne("nan")]
        if cleaned.empty:
            return {}
        return {str(k): int(v) for k, v in cleaned.value_counts().to_dict().items()}

    return {
        "rows_total": int(len(df)),
        "rows_infeasible": int(is_infeasible.sum()),
        "rows_warning": int(is_warning.sum()),
        "rows_unassigned": rows_unassigned,
        "rows_reassignment_candidate": int(reassignment_candidate.sum()),
        "rows_reassignment_failed": int((is_failed & reassignment_candidate).sum()),
        "rows_reassignment_success": int((~is_failed & reassignment_candidate).sum()),
        "infeasibility_causes": _counts_to_dict(df.get("infeasibility_cause_code", pd.Series(dtype="object"))),
        "warning_causes": _counts_to_dict(df.get("warning_code", pd.Series(dtype="object"))),
        "actual_status_counts": _counts_to_dict(status_series),
    }


def _stabilize_results_order(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()
    df["__service_id_num"] = pd.to_numeric(df.get("service_id"), errors="coerce")
    df["__labor_sequence_num"] = pd.to_numeric(df.get("labor_sequence"), errors="coerce")
    df["__labor_id_num"] = pd.to_numeric(df.get("labor_id"), errors="coerce")
    df["__schedule_ts"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)
    df["__actual_start_ts"] = pd.to_datetime(df.get("actual_start"), errors="coerce", utc=True)
    df["__created_ts"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    df["__orig_idx"] = range(len(df))
    df = df.sort_values(
        [
            "__service_id_num",
            "__labor_sequence_num",
            "__labor_id_num",
            "__schedule_ts",
            "__actual_start_ts",
            "__created_ts",
            "__orig_idx",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return df.drop(
        columns=[
            "__service_id_num",
            "__labor_sequence_num",
            "__labor_id_num",
            "__schedule_ts",
            "__actual_start_ts",
            "__created_ts",
            "__orig_idx",
        ],
        errors="ignore",
    )


# ======================================================
# Entry point
# ======================================================
def main() -> int:
    # Generate a run correlation id for this process execution
    setup_logging_context()
    run_id = str(uuid.uuid4())[:8]
    set_run_id(run_id)

    # --------------------------------------------------
    # 1. Bootstrap & validation
    # --------------------------------------------------
    try:
        Config.validate()
        Config.configure_logging()

    except Exception:
        _configure_fallback_logging()
        logger.exception("configuration_error")
        return 1

    log_info("pipeline_start", mode_default="API" if Config.USE_API else "LOCAL")
    enabled_local_outputs: list[str] = []
    if Config.WRITE_VALIDATION_REPORTS:
        enabled_local_outputs.append("validation_reports")
    if Config.WRITE_INTERMEDIATE_DATAFRAMES:
        enabled_local_outputs.append("intermediate_dataframes")
    if Config.WRITE_MODEL_SOLUTION:
        enabled_local_outputs.append("model_solution")
    if enabled_local_outputs:
        log_info("local_outputs_enabled", outputs=",".join(enabled_local_outputs))
    else:
        log_info("local_outputs_disabled")

    use_api = Config.USE_API
    run_mode = "api" if use_api else "local"
    artifact_run_id = f"{run_mode}-{run_id}"
    request_id: Optional[str] = None
    request_payload = None
    request_filters = None
    request_settings: Optional[OptimizationSettings] = None
    keep_payload_assignment = DEFAULT_KEEP_PAYLOAD_ASSIGNMENT
    intermediate_export_dir: Optional[Path] = None
    output_dir = Config.LOCAL_OUTPUT_DIR
    department: int | str | None = Config.DEPARTMENT

    # --------------------------------------------------
    # 2. Load request payload (optional)
    # --------------------------------------------------
    request_path = os.getenv("REQUEST_PATH", "request.json")
    try:
        with log_step("load_request", path=request_path):
            request_payload = load_request(request_path)

        request_id = request_payload.request_id or request_id
        request_filters = request_payload.filters
        keep_payload_assignment = _resolve_keep_payload_assignment(request_payload)
        request_settings = build_settings_from_request(request_payload)
        if request_filters and request_filters.department is not None:
            department = request_filters.department

        if request_payload.output.path:
            output_dir = request_payload.output.path

    except FileNotFoundError:
        log_info("request_missing_using_defaults", path=request_path)
    except Exception as exc:
        logger.exception(_format_log_message("request_load_failed", path=request_path))
        report_failure_if_possible(request_id=request_id, error="REQUEST_LOAD_FAILED", details=str(exc), use_api=use_api)
        return 1

    preassignment_mode = "keep_payload_assignments" if keep_payload_assignment else "reassign_infeasible_services"
    log_info(
        "preassignment_mode_resolved",
        keep_payload_assignment=keep_payload_assignment,
        mode=preassignment_mode,
    )

    # set request_id into logging context ASAP
    raw_input: dict[str, Any] = {}
    input_df = None
    metadata: Dict[str, Any] = {}
    validation_report: Dict[str, Any] = {}
    local_input_is_json_payload = False

    # --------------------------------------------------
    # 3. Acquire input
    # --------------------------------------------------
    try:
        if use_api:
            # log_info("execution_mode", mode="API")

            alfred_client = ALFREDAPIClient(
                endpoint_url=Config.SERVICES_ENDPOINT,
                api_token=Config.API_TOKEN,
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.API_MAX_RETRIES,
            )

            start_date = Config.START_DATE
            end_date = Config.END_DATE

            if request_filters:
                if request_filters.start_date is not None:
                    start_date = request_filters.start_date
                if request_filters.end_date is not None:
                    end_date = request_filters.end_date

            with log_step("fetch_optimization_data"):
                raw_input = alfred_client.get_optimization_data(
                    department=department,
                    start_date=start_date,
                    end_date=end_date,
                )

            # If API returns request_id, prefer it
            request_id = raw_input.get("request_id") or request_id
        else:
            # log_info("execution_mode", mode="LOCAL")
            local_path = Config.LOCAL_INPUT_PATH
            local_input_is_json_payload = Path(local_path).suffix.lower() == ".json"

            with log_step("load_local_input", path=local_path):
                raw_input = load_local_input(local_path, write_debug_json=True)

        if Config.WRITE_INTERMEDIATE_DATAFRAMES:
            intermediate_export_dir = _build_intermediate_export_dir(
                run_id=artifact_run_id,
                request_id=request_id,
            )

    except Exception as exc:
        logger.exception("input_acquisition_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_ACQUISITION_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 2

    # --------------------------------------------------
    # 4. Parse input
    # --------------------------------------------------
    try:
        with log_step("parse_input"):
            input_df, metadata = InputParser.parse(raw_input)

        if input_df is None or input_df.empty:
            raise ValueError("parsed_input_empty")

    except Exception as exc:
        logger.exception("input_parsing_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_PARSING_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 3

    if (
        Config.IGNORE_API_PAYLOAD_DRIVER
        and (use_api or local_input_is_json_payload)
        and "assigned_driver" in input_df.columns
    ):
        assigned_count = int(
            (
                input_df["assigned_driver"].notna()
                & input_df["assigned_driver"].astype(str).str.strip().ne("")
            ).sum()
        )
        input_df["assigned_driver"] = pd.NA
        log_info(
            "api_payload_driver_ignored_for_testing",
            cleared_rows=assigned_count,
        )

    if (
        request_filters
        and request_filters.department is not None
        and "department_code" in input_df.columns
    ):
        department_filter = str(request_filters.department).strip()
        if department_filter:
            department_series = input_df["department_code"].astype("string").str.strip()
            missing_department_mask = department_series.isna() | department_series.eq("")
            missing_rows = int(missing_department_mask.sum())
            if missing_rows > 0:
                input_df.loc[missing_department_mask, "department_code"] = department_filter

                if "location_resolution_status" in input_df.columns:
                    status_series = input_df["location_resolution_status"].astype("string").str.strip()
                    unresolved_mask = (
                        status_series.isna()
                        | status_series.eq("")
                        | status_series.isin(["missing", "ambiguous"])
                    )
                    promote_mask = missing_department_mask & unresolved_mask
                    input_df.loc[promote_mask, "location_resolution_status"] = "resolved_department_only"

                log_info(
                    "department_code_inferred_from_request_filter",
                    inferred_rows=missing_rows,
                    department=department_filter,
                )

    if not use_api and request_filters:
        input_df = apply_request_filters(input_df, request_filters)

    # --------------------------------------------------
    # 5. Validate input
    # --------------------------------------------------
    try:
        rules = [
            NonEmptyRowRule(),

            RequiredFieldRule("service_id"),
            RequiredFieldRule("labor_id"),
            RequiredFieldRule("created_at"),
            RequiredFieldRule("schedule_date"),
            RequiredFieldRule("start_address_point"),
            RequiredFieldRule("labor_name"),
            RequiredFieldRule("end_address_point"),

            UniqueLaborIdRule(),
            CreatedBeforeScheduleRule(minimum_delta_hours=2.0),
            ValidDepartmentsOnly(
                field="department_code",
                valid_departments=(
                    "25",
                    "76",
                    "5",
                ),
            ),
            ValidLocationResolutionStatus(
                field="location_resolution_status",
                valid_statuses=("resolved", "resolved_department_only"),
            ),
        ]

        validator = InputValidator(rules=rules)

        with log_step("validate_input"):
            valid_df, invalid_df, validation_report = validator.validate(input_df)

        if Config.WRITE_VALIDATION_REPORTS:
            with log_step("write_validation_reports"):
                save_local_validation_outputs(
                    invalid_df,
                    validation_report,
                    output_dir=output_dir,
                    request_id=request_id,
                    run_id=artifact_run_id,
                )

        if not invalid_df.empty:
            logger.warning(
                _format_log_message(
                    "input_validation_found_issues",
                    invalid_rows=int(len(invalid_df)),
                )
            )

        if valid_df.empty:
            raise ValueError("all_rows_failed_validation")

        input_df = valid_df

    except Exception as exc:
        logger.exception("input_validation_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_VALIDATION_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 4

    # --------------------------------------------------
    # 6. Load driver directory + master data
    # --------------------------------------------------
    settings = request_settings or OptimizationSettings()
    driver_directory_df = pd.DataFrame()
    driver_directory_source = None

    try:
        schedule_date = None
        if request_filters and request_filters.schedule_date:
            schedule_date = request_filters.schedule_date
        elif Config.SCHEDULE_DATE:
            schedule_date = Config.SCHEDULE_DATE
        elif "schedule_date" in input_df.columns and not input_df.empty:
            inferred = utc_to_colombia_series(input_df["schedule_date"], errors="coerce")
            if inferred.notna().any():
                schedule_date = inferred.dropna().iloc[0].date()

        if use_api:
            if schedule_date is None:
                raise ValueError("schedule_date_required_for_driver_directory")

            driver_client = ALFREDAPIClient(
                endpoint_url=Config.ALFREDS_ENDPOINT,
                api_token=Config.API_TOKEN,
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.API_MAX_RETRIES,
            )

            try:
                with log_step(
                    "fetch_driver_directory",
                    # schedule_date=str(schedule_date),
                    # department=department,
                ):
                    raw_drivers = driver_client.get_driver_directory(
                        active=True,
                        schedule_date=schedule_date,
                        department=department,
                    )
            except Exception as exc:
                logger.exception("driver_directory_api_failed")
                raise

            if not raw_drivers:
                logger.error("driver_directory_empty_from_api")
                report_failure_if_possible(
                    request_id=request_id,
                    error="DRIVER_DIRECTORY_EMPTY",
                    details="No drivers returned from API",
                    use_api=use_api,
                )
                return 2

            driver_directory_df = DriverDirectoryParser.parse(raw_drivers)
            driver_directory_source = "API"
            if driver_directory_df.empty:
                logger.error("driver_directory_empty_after_parse source=API")
                report_failure_if_possible(
                    request_id=request_id,
                    error="DRIVER_DIRECTORY_EMPTY",
                    details="No usable drivers after parsing API response",
                    use_api=use_api,
                )
                return 2
        else:
            with log_step(
                "load_local_driver_directory",
                path=Config.LOCAL_DRIVER_DIRECTORY_FILE,
            ):
                driver_directory_df = load_driver_directory_df(
                    Config.LOCAL_DRIVER_DIRECTORY_FILE
                )

            if request_filters and request_filters.department is not None:
                driver_directory_df = _filter_df_by_department_code(
                    driver_directory_df,
                    department=request_filters.department,
                    dataset_name="driver_directory",
                )

            driver_directory_source = "LOCAL_CSV"
            if driver_directory_df.empty:
                raise ValueError("driver_directory_empty_local_csv")

    except Exception as exc:
        logger.exception("driver_directory_load_failed")
        if use_api:
            try:
                with log_step(
                    "load_driver_directory_fallback",
                    path=Config.DRIVER_DIRECTORY_FALLBACK_PATH,
                ):
                    driver_directory_df = load_driver_directory_df(
                        Config.DRIVER_DIRECTORY_FALLBACK_PATH
                    )
                driver_directory_source = "FALLBACK_FILE"
                if driver_directory_df.empty:
                    raise ValueError("driver_directory_empty_fallback")

                log_info(
                    "driver_directory_loaded",
                    source=driver_directory_source,
                    rows=len(driver_directory_df),
                )
            except Exception as fallback_exc:
                logger.exception("driver_directory_fallback_failed")
                report_failure_if_possible(
                    request_id=request_id,
                    error="DRIVER_DIRECTORY_FAILED",
                    details=str(fallback_exc),
                    use_api=use_api,
                )
                return 2
        else:
            report_failure_if_possible(
                request_id=request_id,
                error="DRIVER_DIRECTORY_FAILED",
                details=str(exc),
                use_api=use_api,
            )
            return 2

    _export_intermediate_dataframe(
        driver_directory_df,
        name="driver_directory",
        export_dir=intermediate_export_dir,
        run_id=artifact_run_id,
        request_id=request_id,
    )

    master_data = load_master_data(settings.master_data)
    if driver_directory_df is not None and not driver_directory_df.empty:
        master_data = MasterData(
            directorio_df=driver_directory_df,
            duraciones_df=master_data.duraciones_df,
            dist_dict=master_data.dist_dict,
        )

    # --------------------------------------------------
    # 7. Reconstruct preassigned schedule (if any)
    # --------------------------------------------------
    preassigned_df = pd.DataFrame()
    preassigned_moves_df = pd.DataFrame()
    preassigned_metrics: Dict[str, Any] = {}

    algorithm_name = str(settings.algorithm or "").strip().upper()
    should_reconstruct_preassigned = algorithm_name != "OFFLINE"
    if not should_reconstruct_preassigned:
        log_info(
            "preassigned_reconstruction_skipped",
            reason="algorithm_offline",
            algorithm=algorithm_name,
        )

    if should_reconstruct_preassigned and "assigned_driver" in input_df.columns:
        has_preassigned = (
            input_df["assigned_driver"].notna()
            & input_df["assigned_driver"].astype(str).str.strip().ne("")
        )
        if has_preassigned.any():
            from src.optimization.common.preassigned import reconstruct_preassigned_state

            with log_step("reconstruct_preassigned"):
                preassigned_df, input_df, preassigned_moves_df, preassigned_metrics = reconstruct_preassigned_state(
                    input_df,
                    directorio_df=master_data.directorio_df,
                    duraciones_df=master_data.duraciones_df,
                    dist_method=settings.distance_method,
                    dist_dict=master_data.dist_dict,
                    model_params=settings.model_params,
                )

            preassigned_report_df = preassigned_df.copy()
            preassigned_report_df["preassigned_action"] = "keep_fixed"

            if not keep_payload_assignment and not preassigned_df.empty:
                infeasible_service_ids = preassigned_df.loc[
                    preassigned_df.get("is_infeasible", False).fillna(False),
                    "service_id",
                ].dropna().unique()

                if len(infeasible_service_ids) > 0:
                    reassignment_mask = preassigned_df["service_id"].isin(infeasible_service_ids)
                    candidate_rows = _prepare_service_rows_for_reassignment(preassigned_df.loc[reassignment_mask])
                    preassigned_df = preassigned_df.loc[~reassignment_mask].copy()
                    if not preassigned_moves_df.empty and "service_id" in preassigned_moves_df.columns:
                        fixed_service_ids = preassigned_df["service_id"].dropna().unique()
                        preassigned_moves_df = preassigned_moves_df[
                            preassigned_moves_df["service_id"].isin(fixed_service_ids)
                        ].copy()
                    candidate_rows["preassigned_action"] = "reassign_service"
                    preassigned_report_df.loc[
                        preassigned_report_df["service_id"].isin(infeasible_service_ids),
                        "preassigned_action",
                    ] = "reassign_service"

                    if not candidate_rows.empty:
                        if "reassignment_candidate" not in input_df.columns:
                            input_df["reassignment_candidate"] = False
                        else:
                            input_df["reassignment_candidate"] = input_df["reassignment_candidate"].fillna(False)
                        if "reassignment_priority" not in input_df.columns:
                            input_df["reassignment_priority"] = 0
                        else:
                            input_df["reassignment_priority"] = pd.to_numeric(
                                input_df["reassignment_priority"],
                                errors="coerce",
                            ).fillna(0)
                        input_df = pd.concat([candidate_rows, input_df], axis=0, ignore_index=True)

                    preassigned_metrics["services_marked_for_reassignment"] = int(len(infeasible_service_ids))
                    preassigned_metrics["labors_marked_for_reassignment"] = int(len(candidate_rows))
                else:
                    preassigned_metrics["services_marked_for_reassignment"] = 0
                    preassigned_metrics["labors_marked_for_reassignment"] = 0
            else:
                preassigned_metrics["services_marked_for_reassignment"] = 0
                preassigned_metrics["labors_marked_for_reassignment"] = 0

            if "reassignment_candidate" not in input_df.columns:
                input_df["reassignment_candidate"] = False
            else:
                input_df["reassignment_candidate"] = input_df["reassignment_candidate"].fillna(False)
            if "reassignment_priority" not in input_df.columns:
                input_df["reassignment_priority"] = 0
            else:
                input_df["reassignment_priority"] = pd.to_numeric(
                    input_df["reassignment_priority"],
                    errors="coerce",
                ).fillna(0)

            preassigned_metrics["keep_payload_assignment"] = bool(keep_payload_assignment)
            preassigned_metrics["preassignment_mode"] = preassignment_mode
            metadata["preassigned"] = preassigned_metrics

            if not preassigned_report_df.empty:
                try:
                    with log_step("write_preassigned_reconstruction_reports"):
                        save_local_preassigned_reconstruction_reports(
                            preassigned_report_df,
                            preassigned_metrics,
                            output_dir=output_dir,
                            request_id=request_id,
                            run_id=artifact_run_id,
                        )
                except Exception:
                    logger.exception("preassigned_reconstruction_report_write_failed")

    _export_intermediate_dataframe(
        input_df,
        name="input_df",
        export_dir=intermediate_export_dir,
        run_id=artifact_run_id,
        request_id=request_id,
    )
    _export_intermediate_dataframe(
        preassigned_df,
        name="preassigned_df",
        export_dir=intermediate_export_dir,
        run_id=artifact_run_id,
        request_id=request_id,
    )

    # --------------------------------------------------
    # 8. Run optimization
    # --------------------------------------------------
    algo_artifacts: Dict[str, Any] = {}
    try:
        context: Dict[str, Any] = {}
        if request_payload:
            context = {
                "request": request_payload.raw,
                "filters": request_payload.filters.as_dict(),
            }
        if not preassigned_df.empty:
            context["preassigned"] = {
                "labors_df": preassigned_df,
                "moves_df": preassigned_moves_df,
                "metrics": preassigned_metrics,
            }

        if input_df.empty and not preassigned_df.empty:
            results = preassigned_df
            metrics = {
                "algorithm": "PREASSIGNED_ONLY",
                "preassigned_metrics": preassigned_metrics,
                "input_rows": 0,
                "output_rows": len(results),
            }
            algo_artifacts = {}
        else:
            solver = OptimizationSolver(
                input_df,
                settings=settings,
                context=context,
                master_data_override=master_data,
            )

            with log_step("solve"):
                results, metrics, algo_artifacts = solver.solve()

            if not preassigned_df.empty:
                results = pd.concat([preassigned_df, results], axis=0, ignore_index=True)
        results = _finalize_assignment_diagnostics(results)
        results = _stabilize_results_order(results)
        assignment_diagnostics = _build_assignment_diagnostics_metrics(results)
        metadata["assignment_diagnostics"] = assignment_diagnostics
        if isinstance(metrics, dict):
            metrics["assignment_diagnostics"] = assignment_diagnostics
        log_info(
            "assignment_diagnostics_summary",
            rows_total=assignment_diagnostics.get("rows_total", 0),
            rows_infeasible=assignment_diagnostics.get("rows_infeasible", 0),
            rows_warning=assignment_diagnostics.get("rows_warning", 0),
            rows_reassignment_candidate=assignment_diagnostics.get("rows_reassignment_candidate", 0),
            rows_reassignment_failed=assignment_diagnostics.get("rows_reassignment_failed", 0),
        )
        if Config.WRITE_MODEL_SOLUTION:
            try:
                with log_step("write_assignment_diagnostics_report"):
                    save_local_assignment_diagnostics_report(
                        results,
                        assignment_diagnostics,
                        output_dir=output_dir,
                        request_id=request_id,
                        run_id=artifact_run_id,
                    )
            except Exception:
                logger.exception("assignment_diagnostics_report_write_failed")

        # out_len = len(results) if hasattr(results, "__len__") else None
        # log_info("optimization_completed", rows_out=out_len, metrics=metrics)
    
    except Exception as exc:
        logger.exception("optimization_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="OPTIMIZATION_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 5

    moves_df = algo_artifacts.get("moves_df", pd.DataFrame()) if isinstance(algo_artifacts, dict) else pd.DataFrame()
    dist_dict = algo_artifacts.get("dist_dict") if isinstance(algo_artifacts, dict) else None

    if not preassigned_moves_df.empty:
        if moves_df is None or moves_df.empty:
            moves_df = preassigned_moves_df
        else:
            moves_df = pd.concat([preassigned_moves_df, moves_df], axis=0, ignore_index=True)

    # --------------------------------------------------
    # 9. Validate solution
    # --------------------------------------------------
    try:
        solution_validation_report: Dict[str, Any] = {}
        solution_validation_issues = pd.DataFrame()
        with log_step("validate_solution"):
            solution_validation_report, solution_validation_issues = validate_solution(
                labors_df=results,
                moves_df=moves_df,
                model_params=settings.model_params,
                dist_method=str(algo_artifacts.get("distance_method", settings.distance_method))
                if isinstance(algo_artifacts, dict)
                else settings.distance_method,
                dist_dict=dist_dict,
                strict_time_check=False,
            )

        if Config.WRITE_VALIDATION_REPORTS:
            with log_step("write_solution_validation_outputs"):
                save_local_solution_validation_outputs(
                    solution_validation_issues,
                    solution_validation_report,
                    output_dir=output_dir,
                    request_id=request_id,
                    run_id=artifact_run_id,
                )

        if solution_validation_report.get("blocking_failed"):
            blocking = solution_validation_report.get("blocking_issues", {})
            logger.error(
                _format_log_message(
                    "solution_validation_blocking_failed",
                    issues=len(solution_validation_issues) if hasattr(solution_validation_issues, "__len__") else None,
                    blocking=blocking,
                )
            )
            raise ValueError("solution_validation_blocking_failed")

        if solution_validation_report.get("summary", {}).get("total_issues", 0) > 0:
            logger.warning(
                _format_log_message(
                    "solution_validation_non_blocking_issues",
                    issues=solution_validation_report.get("summary", {}).get("total_issues", 0),
                )
            )

    except Exception as exc:
        logger.exception("solution_validation_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="SOLUTION_VALIDATION_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 8

    # --------------------------------------------------
    # 10. Evaluate solution
    # --------------------------------------------------
    evaluation_report: Dict[str, Any] = {}
    try:
        with log_step("evaluate_solution"):
            results, evaluation_report = evaluate_solution(
                labors_df=results,
                moves_df=moves_df,
                driver_directory_df=driver_directory_df,
                grace_minutes=settings.model_params.tiempo_gracia_min,
                default_shift_end=settings.model_params.workday_end_str,
            )

        if Config.WRITE_MODEL_SOLUTION:
            with log_step("write_solution_evaluation_output"):
                save_local_solution_evaluation_report(
                    evaluation_report,
                    output_dir=output_dir,
                    request_id=request_id,
                    run_id=artifact_run_id,
                )
    except Exception as exc:
        logger.exception("solution_evaluation_failed")
        evaluation_report = {
            "error": "solution_evaluation_failed",
            "details": str(exc),
        }

    # --------------------------------------------------
    # 11. Format output payload
    # --------------------------------------------------
    output_payload: Dict[str, Any] = {}
    need_output_payload = use_api or Config.WRITE_MODEL_SOLUTION
    if need_output_payload:
        try:
            with log_step("format_output"):
                output_payload = OutputFormatter.format(
                    results=results,
                    metadata={
                        "metrics": metrics,
                        "validation": validation_report,
                        "solution_validation": solution_validation_report,
                        "evaluation": evaluation_report,
                        **metadata,
                    },
                    request_id=request_id,
                    status="completed",
                )

        except Exception as exc:
            logger.exception("output_formatting_failed")
            report_failure_if_possible(
                request_id=request_id,
                error="OUTPUT_FORMATTING_FAILED",
                details=str(exc),
                use_api=use_api,
            )
            return 6

    # --------------------------------------------------
    # 12. Deliver output
    # --------------------------------------------------
    try:
        if Config.WRITE_MODEL_SOLUTION:
            with log_step("save_model_solution_artifacts"):
                save_local_output(
                    results,
                    output_path=output_dir,
                    request_id=request_id,
                    run_id=artifact_run_id,
                )
                save_local_output_payload(
                    output_payload,
                    output_dir=output_dir,
                    request_id=request_id,
                    run_id=artifact_run_id,
                )

        if use_api:
            sender = ResultSender(
                base_url=Config.API_ENDPOINT,
                api_key=Config.API_TOKEN,
                auth_scheme="Bearer",
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.API_MAX_RETRIES,
            )
            with log_step("send_results"):
                sender.send_results(output_payload, request_id=request_id)

        log_info("output_delivered")

    except Exception as exc:
        logger.exception("output_delivery_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="OUTPUT_DELIVERY_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 7

    log_info("pipeline_end", status="success")
    return 0

if __name__ == "__main__":
    sys.exit(main())
