import os
import sys
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config
from src.utils.datetime_utils import utc_to_colombia_series
from src.utils.logging_utils import set_run_id, setup_logging_context, add_file_log_handler
from src.integration.client import ALFREDAPIClient
from src.integration.sender import ResultSender
from src.io.input_loader import load_local_input
from src.io.request_loader import (
    apply_request_filters,
    build_settings_from_request,
    load_request,
    DEFAULT_KEEP_PAYLOAD_ASSIGNMENT,
    resolve_keep_payload_assignment,
)
from src.io.output_writer import (
    save_local_assignment_diagnostics_report,
    save_local_output,
    save_local_output_payload,
    save_local_preassigned_reconstruction_reports,
    save_local_validation_outputs,
    save_local_solution_evaluation_report,
    save_local_solution_validation_outputs,
    save_local_warnings_report,
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
from src.optimization.common.preassigned import _prepare_service_rows_for_reassignment
from src.pipeline.helpers import (
    _build_intermediate_export_dir,
    _configure_fallback_logging,
    _export_intermediate_dataframe,
    _format_log_message,
    log_info,
    log_step,
    report_failure_if_possible,
    set_pipeline_logger,
)
from src.pipeline.filters import (
    _filter_df_by_department_code,
    _filter_labors_to_planning_window,
    filter_canceled_services,
)
from src.pipeline.diagnostics import (
    _finalize_assignment_diagnostics,
    _build_assignment_diagnostics_metrics,
    _stabilize_results_order,
)

logger = logging.getLogger(__name__)
set_pipeline_logger(logger)


# ======================================================
# Entry point
# ======================================================
def main() -> int:
    from src.io.artifact_naming import build_run_subdir, write_run_manifest, finalize_run_manifest

    # Generate a run correlation id for this process execution
    setup_logging_context()
    run_id = str(uuid.uuid4())[:8]
    set_run_id(run_id)
    started_at = datetime.now(timezone.utc)

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
    output_dir = Config.RUNS_DIR
    department: int | str | None = Config.DEPARTMENT

    # Create run directory early, write initial manifest, and attach file log
    run_dir = Path(output_dir) / build_run_subdir(artifact_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(run_dir, artifact_run_id, created_at=started_at)
    add_file_log_handler(run_dir / "run.log")

    # --------------------------------------------------
    # 2. Load request payload (optional)
    # --------------------------------------------------
    request_path = os.getenv("REQUEST_PATH", "request.json")
    try:
        with log_step("load_request", path=request_path):
            request_payload = load_request(request_path)

        request_id = request_payload.request_id or request_id
        request_filters = request_payload.filters
        keep_payload_assignment = resolve_keep_payload_assignment(request_payload)
        request_settings = build_settings_from_request(request_payload)
        if request_filters and request_filters.department is not None:
            department = request_filters.department

        if request_payload.output.path:
            output_dir = request_payload.output.path
            run_dir = Path(output_dir) / build_run_subdir(artifact_run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            write_run_manifest(run_dir, artifact_run_id, created_at=started_at)

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
            local_path = Config.LOCAL_INPUT_PATH
            local_input_is_json_payload = Path(local_path).suffix.lower() == ".json"

            with log_step("load_local_input", path=local_path):
                raw_input = load_local_input(local_path, write_debug_json=True, run_id=artifact_run_id)

        # -----------------------------------------------------------
        # SERVICE MASK (testing only) — remove this block and unset
        # SERVICE_MASK_PATH in .env to disable without any side effects.
        # -----------------------------------------------------------
        _mask_path = os.getenv("SERVICE_MASK_PATH")
        if _mask_path:
            from src.io.service_mask import apply_service_mask, load_labor_ids_from_snapshot
            _mask_labor_ids = load_labor_ids_from_snapshot(_mask_path)
            raw_input = apply_service_mask(raw_input, _mask_labor_ids)
        # -----------------------------------------------------------

        if Config.WRITE_INTERMEDIATE_DATAFRAMES:
            intermediate_export_dir = _build_intermediate_export_dir(
                run_id=artifact_run_id,
                request_id=request_id,
                run_base_dir=run_dir,
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

    if request_filters:
        input_df = _filter_labors_to_planning_window(input_df, request_filters)

    input_df = filter_canceled_services(input_df)

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
                with log_step("fetch_driver_directory"):
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

    # Capture full input scope before preassigned split (for run.json reporting)
    _input_services_total = int(input_df["service_id"].nunique()) if "service_id" in input_df.columns and not input_df.empty else 0

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
            ts_raw = request_payload.raw.get("timestamp")
            if ts_raw:
                try:
                    context["decision_time"] = pd.Timestamp(ts_raw)
                except Exception:
                    pass
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
                # Use INSERT-updated base labors when available: downstream shifts
                # applied during insertion may have changed actual_start/actual_end
                # of preassigned labors.  Using the originals would create false
                # driver_overlap issues in the validator.
                _updated_base = (
                    algo_artifacts.get("updated_base_labors_df", pd.DataFrame())
                    if isinstance(algo_artifacts, dict) else pd.DataFrame()
                )
                if not _updated_base.empty:
                    _shifted_ids = set(_updated_base["labor_id"].tolist())
                    preassigned_for_concat = pd.concat(
                        [
                            preassigned_df[~preassigned_df["labor_id"].isin(_shifted_ids)],
                            _updated_base,
                        ],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    preassigned_for_concat = preassigned_df
                # Exclude any preassigned labors already covered by the algorithm's output
                # (e.g. BUFFER_REACT re-optimizes reassignable labors and returns them in
                # results, so concatenating preassigned_df would duplicate those labor_ids).
                if "labor_id" in results.columns and "labor_id" in preassigned_for_concat.columns:
                    _algo_labor_ids = set(results["labor_id"].dropna().tolist())
                    preassigned_for_concat = preassigned_for_concat[
                        ~preassigned_for_concat["labor_id"].isin(_algo_labor_ids)
                    ]
                results = pd.concat([preassigned_for_concat, results], axis=0, ignore_index=True)

        results = _finalize_assignment_diagnostics(results)
        results = _stabilize_results_order(results)
        assignment_diagnostics = _build_assignment_diagnostics_metrics(results)
        metadata["assignment_diagnostics"] = assignment_diagnostics

        # Service-level summary for run.json
        if not results.empty and "service_id" in results.columns:
            _services_total = _input_services_total
            if "is_infeasible" in results.columns:
                _svc_all_failed = results.groupby("service_id")["is_infeasible"].all()
                _services_failed = int(_svc_all_failed.sum())
            else:
                _services_failed = 0
            _services_planned = _services_total - _services_failed
        else:
            _services_total = _services_planned = _services_failed = 0

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
                        run_id=artifact_run_id,
                    )
            except Exception:
                logger.exception("assignment_diagnostics_report_write_failed")

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
        # Replace preassigned moves that were updated by the INSERT algorithm
        # (downstream-shifted moves) with their updated versions.
        _updated_base_moves = (
            algo_artifacts.get("updated_base_moves_df", pd.DataFrame())
            if isinstance(algo_artifacts, dict) else pd.DataFrame()
        )
        if not _updated_base_moves.empty and "labor_id" in _updated_base_moves.columns:
            _shifted_move_ids = set(_updated_base_moves["labor_id"].tolist())
            preassigned_moves_for_concat = pd.concat(
                [
                    preassigned_moves_df[~preassigned_moves_df["labor_id"].isin(_shifted_move_ids)],
                    _updated_base_moves,
                ],
                axis=0,
                ignore_index=True,
            )
        else:
            preassigned_moves_for_concat = preassigned_moves_df
        if moves_df is None or moves_df.empty:
            moves_df = preassigned_moves_for_concat
        else:
            moves_df = pd.concat([preassigned_moves_for_concat, moves_df], axis=0, ignore_index=True)

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
                time_method=str(algo_artifacts.get("time_method", "speed_based"))
                if isinstance(algo_artifacts, dict)
                else "speed_based",
                time_dict=algo_artifacts.get("time_dict")
                if isinstance(algo_artifacts, dict)
                else None,
                strict_time_check=False,
            )

        if Config.WRITE_VALIDATION_REPORTS:
            with log_step("write_solution_validation_outputs"):
                save_local_solution_validation_outputs(
                    solution_validation_issues,
                    solution_validation_report,
                    output_dir=output_dir,
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
                    run_id=artifact_run_id,
                )
                save_local_output_payload(
                    output_payload,
                    output_dir=output_dir,
                    run_id=artifact_run_id,
                )

        with log_step("save_warnings_report"):
            _overtime_series = results.get("overtime_minutes", pd.Series(dtype=float)).fillna(0) if results is not None and not results.empty else pd.Series(dtype=float)
            _preassigned_infeasible = int(preassigned_df["is_infeasible"].fillna(False).sum()) if not preassigned_df.empty and "is_infeasible" in preassigned_df.columns else 0
            _nontransport_mask = (
                (input_df["labor_category"] != "VEHICLE_TRANSPORTATION")
                & input_df["shop_id"].notna()
            ) if not input_df.empty and "labor_category" in input_df.columns and "shop_id" in input_df.columns else pd.Series(False, index=input_df.index if not input_df.empty else [])
            _et_fallback_mask = (
                _nontransport_mask & input_df["estimated_time"].isna()
            ) if "estimated_time" in input_df.columns else _nontransport_mask
            _et_fallback_count = int(_et_fallback_mask.sum())
            _et_fallback_ids = input_df.loc[_et_fallback_mask, "labor_id"].astype(str).tolist() if _et_fallback_count > 0 else []
            warnings_report = {
                "drivers": {
                    "defaulted_schedule_count": int(driver_directory_df.get("schedule_defaulted", pd.Series(dtype=bool)).fillna(False).sum()) if not driver_directory_df.empty else 0,
                    "defaulted_schedule_driver_ids": driver_directory_df.loc[driver_directory_df["schedule_defaulted"].fillna(False).astype(bool), "driver_id"].astype(str).tolist() if not driver_directory_df.empty and "schedule_defaulted" in driver_directory_df.columns else [],
                },
                "data_validation": {
                    "invalid_records_removed": int(len(invalid_df)) if invalid_df is not None else 0,
                    "failures_by_rule": validation_report.get("failures_by_rule", {}) if validation_report else {},
                },
                "assignment": {
                    "failed_services_count": _services_failed,
                    "overtime_labors_count": int((_overtime_series > 0).sum()),
                    "overtime_total_minutes": round(float(_overtime_series.sum()), 2),
                },
                "preassigned": {
                    "infeasible_count": _preassigned_infeasible,
                },
                "service_durations": {
                    "estimated_time_fallback_count": _et_fallback_count,
                    "estimated_time_fallback_labor_ids": _et_fallback_ids,
                },
            }
            save_local_warnings_report(warnings_report, output_dir=output_dir, run_id=artifact_run_id)

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
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    try:
        _dept_codes: list[str] = []
        _start_date: Optional[str] = None
        _end_date: Optional[str] = None
        if not results.empty:
            if "department_code" in results.columns:
                _dept_codes = sorted(results["department_code"].dropna().astype(str).unique().tolist())
            if "schedule_date" in results.columns:
                _dates = pd.to_datetime(results["schedule_date"], errors="coerce").dropna()
                if not _dates.empty:
                    _start_date = _dates.min().date().isoformat()
                    _end_date = _dates.max().date().isoformat()
        _instance_config = {
            "department": _dept_codes[0] if len(_dept_codes) == 1 else _dept_codes,
            "start_date": _start_date,
            "end_date": _end_date,
        }
        _n_proc = settings.n_processes
        _algorithm_config = {
            "name": str(settings.algorithm or "default").upper(),
            "max_iterations": {d: settings.max_iterations[d] for d in _dept_codes if d in settings.max_iterations},
            "n_processes": _n_proc,
            "parallel": _n_proc is not None and _n_proc != 1,
        }
        _labors_total = int(len(results)) if not results.empty else 0
        _labors_preassigned = int(len(preassigned_df)) if not preassigned_df.empty else 0
        _labors_processed = _labors_total - _labors_preassigned
        _is_infeasible_col = results.get("is_infeasible", pd.Series(False, index=results.index)).fillna(False).astype(bool) if not results.empty else pd.Series(dtype=bool)
        _labors_assigned = int(results.loc[~_is_infeasible_col, "labor_id"].nunique()) if not results.empty and "labor_id" in results.columns else 0
        _labors_summary = {
            "total": _labors_total,
            "preassigned": _labors_preassigned,
            "processed": _labors_processed,
            "assigned": _labors_assigned,
        }
        finalize_run_manifest(
            run_dir,
            status="success",
            duration_seconds=elapsed,
            solver=str(settings.algorithm or "default").upper(),
            services_total=_services_total,
            services_planned=_services_planned,
            services_failed=_services_failed,
            labors_summary=_labors_summary,
            instance_config=_instance_config,
            algorithm_config=_algorithm_config,
        )
    except Exception:
        logger.exception("run_manifest_finalization_failed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
