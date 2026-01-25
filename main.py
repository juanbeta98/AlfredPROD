import os
import sys
import uuid
import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config
from src.integration.client import ALFREDAPIClient
from src.integration.sender import ResultSender
from src.io.input_loader import load_local_input
from src.io.request_loader import build_settings_from_request, load_request
from src.io.output_writer import save_local_output, save_local_validation_outputs
from src.data.parsing.input_parser import InputParser
from src.data.validation.validator import InputValidator
from src.data.validation.rules.generic import RequiredFieldRule, NonEmptyRowRule, UniqueLaborIdRule
from src.data.validation.rules.domain import CreatedBeforeScheduleRule, ValidCitiesOnly
from src.data.formatting.output_formatter import OutputFormatter
from src.optimization.solver import OptimizationSolver
from src.optimization.settings.solver_settings import OptimizationSettings
from src.logging_utils import ContextFilter, set_run_id, setup_logging_context

logger = logging.getLogger(__name__)

def _format_log_message(event: str, **fields: Any) -> str:
    if not fields:
        return event
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"{event} {details}"

def log_info(event: str, **fields: Any) -> None:
    logger.info(_format_log_message(event, **fields))

@contextmanager
def log_step(step: str, **fields):
    start = time.perf_counter()
    log_info("step_start", step=step, **fields)
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            _format_log_message(
                "step_failed",
                step=step,
                duration_ms=round(elapsed_ms, 2),
                **fields,
            )
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_info(
            "step_end",
            step=step,
            duration_ms=round(elapsed_ms, 2),
            **fields,
        )

def _configure_fallback_logging() -> None:
    """If Config.configure_logging() fails, at least show something useful."""
    setup_logging_context()
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


# ======================================================
# Entry point
# ======================================================
def main() -> int:
    # Generate a run correlation id for this process execution
    setup_logging_context()
    set_run_id(str(uuid.uuid4())[:8])

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

    use_api = Config.USE_API
    request_id: Optional[str] = None
    request_payload = None
    request_filters = None

    # --------------------------------------------------
    # 2. Load request payload (optional)
    # --------------------------------------------------
    request_path = os.getenv("REQUEST_PATH", "request.json")
    try:
        with log_step("load_request", path=request_path):
            request_payload = load_request(request_path)

        request_id = request_payload.request_id or request_id
        request_filters = request_payload.filters
        _ = build_settings_from_request(request_payload)

        if request_payload.input.source == "api":
            use_api = True
        elif request_payload.input.source == "local":
            use_api = False

    except FileNotFoundError:
        log_info("request_missing_using_defaults", path=request_path)
    except Exception as exc:
        logger.exception(_format_log_message("request_load_failed", path=request_path))
        report_failure_if_possible(request_id=request_id, error="REQUEST_LOAD_FAILED", details=str(exc), use_api=use_api)
        return 1

    # set request_id into logging context ASAP
    raw_input: dict[str, Any] = {}
    input_df = None
    metadata: Dict[str, Any] = {}
    validation_report: Dict[str, Any] = {}

    # --------------------------------------------------
    # 3. Acquire input
    # --------------------------------------------------
    try:
        if use_api:
            log_info("execution_mode", mode="API")

            alfred_client = ALFREDAPIClient(
                endpoint_url=Config.API_ENDPOINT,
                api_token=Config.API_TOKEN,
            )

            department = Config.DEPARTMENT
            start_date = Config.START_DATE
            end_date = Config.END_DATE

            if request_filters:
                if request_filters.department is not None:
                    department = request_filters.department
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
            log_info("execution_mode", mode="LOCAL")
            local_path = Config.LOCAL_INPUT_FILE
            if request_payload and request_payload.input.path:
                local_path = request_payload.input.path

            with log_step("load_local_input", path=local_path):
                raw_input = load_local_input(local_path, write_debug_json=True)

        log_info("input_acquired", source="API" if use_api else "LOCAL")

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

        log_info("input_parsed")

    except Exception as exc:
        logger.exception("input_parsing_failed")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_PARSING_FAILED",
            details=str(exc),
            use_api=use_api,
        )
        return 3

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
            ValidCitiesOnly(
                field="city",
                valid_cities=(
                    "CUNDINAMARCA-BOGOTA D.C.",
                    "25", "1", "126", "149", "150", "830", "844", "1004"
                ),
            ),
        ]

        validator = InputValidator(rules=rules)

        with log_step("validate_input"):
            valid_df, invalid_df, validation_report = validator.validate(input_df)

        if Config.WRITE_VALIDATION_REPORTS:
            output_dir = Config.LOCAL_OUTPUT_DIR
            if request_payload and request_payload.output.path:
                output_dir = request_payload.output.path

            with log_step("write_validation_reports", output_dir=output_dir):
                save_local_validation_outputs(invalid_df, validation_report, output_dir=output_dir)

        if not invalid_df.empty:
            logger.warning(
                _format_log_message(
                    "input_validation_partial_fail",
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
    # 6. Reconstruct preassigned schedule (if any)
    # --------------------------------------------------
    preassigned_df = pd.DataFrame()
    preassigned_moves_df = pd.DataFrame()
    preassigned_metrics: Dict[str, Any] = {}

    settings = OptimizationSettings()
    if "assigned_driver" in input_df.columns:
        has_preassigned = (
            input_df["assigned_driver"].notna()
            & input_df["assigned_driver"].astype(str).str.strip().ne("")
        )
        if has_preassigned.any():
            from src.data.master_data_loader import load_master_data
            from src.optimization.common.preassigned import reconstruct_preassigned_state

            master_data = load_master_data(settings.master_data)
            with log_step("reconstruct_preassigned"):
                preassigned_df, input_df, preassigned_moves_df, preassigned_metrics = reconstruct_preassigned_state(
                    input_df,
                    directorio_df=master_data.directorio_df,
                    duraciones_df=master_data.duraciones_df,
                    dist_method="haversine",
                    dist_dict=master_data.dist_dict,
                    model_params=settings.model_params,
                )

            metadata["preassigned"] = preassigned_metrics

    # --------------------------------------------------
    # 7. Run optimization
    # --------------------------------------------------
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
        else:
            solver = OptimizationSolver(input_df, settings=settings, context=context)

            with log_step("solve"):
                results, metrics = solver.solve()

            if not preassigned_df.empty:
                results = pd.concat([preassigned_df, results], axis=0, ignore_index=True)

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

    # --------------------------------------------------
    # 7. Format output (API only)
    # --------------------------------------------------
    output_payload: Dict[str, Any] = {}
    if use_api:
        try:
            with log_step("format_output"):
                output_payload = OutputFormatter.format(
                    results=results,
                    metadata={
                        "metrics": metrics,
                        "validation": validation_report,
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
    # 8. Deliver output
    # --------------------------------------------------
    try:
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
        else:
            with log_step("save_local_output"):
                output_path = Config.LOCAL_OUTPUT_DIR
                if request_payload and request_payload.output.path:
                    output_path = request_payload.output.path
                save_local_output(results, output_path=output_path, request_id=request_id)

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

# ======================================================
# Failure reporting (best-effort)
# ======================================================
def report_failure_if_possible(
    request_id: Optional[str],
    error: str,
    details: str,
    use_api: bool,
) -> None:
    if (not use_api) or (not request_id):
        return

    try:
        sender = ResultSender(
            base_url=Config.API_ENDPOINT,
            api_key=Config.API_TOKEN,
            auth_scheme="Bearer",  # keep consistent with success path unless your API truly differs
            timeout=Config.REQUEST_TIMEOUT,
            max_retries=1,
        )

        payload = {
            "request_id": request_id,
            "status": "failed",
            "error": error,
            "details": details,
        }

        sender.send_results(payload, request_id=request_id)

    except Exception:
        logger.exception("failure_reporting_failed")


if __name__ == "__main__":
    sys.exit(main())
