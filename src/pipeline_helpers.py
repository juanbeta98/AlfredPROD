import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import Config
from src.integration.sender import ResultSender
from src.io.artifact_naming import build_artifact_stem, build_run_subdir
from src.logging_utils import ContextFilter, setup_logging_context

logger = logging.getLogger(__name__)
_module_logger = logging.getLogger(__name__)


def set_pipeline_logger(bound_logger: logging.Logger) -> None:
    global logger
    logger = bound_logger


def _format_log_message(event: str, **fields: Any) -> str:
    if not fields:
        return event
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    return f"{event} {details}"


def log_info(event: str, **fields: Any) -> None:
    logger.info(_format_log_message(event, **fields))


def _build_intermediate_export_dir(
    *,
    run_id: str,
    request_id: Optional[str],
) -> Path:
    export_dir = Path(Config.INTERMEDIATE_EXPORT_BASE_DIR) / build_run_subdir(run_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _export_intermediate_dataframe(
    df: pd.DataFrame,
    *,
    name: str,
    export_dir: Optional[Path],
    run_id: str,
    request_id: Optional[str],
) -> None:
    if export_dir is None:
        return

    file_name = build_artifact_stem(
        name,
        run_id=run_id,
        request_id=request_id,
    )
    path = export_dir / f"{file_name}.csv"
    try:
        df.to_csv(path, index=False)
        _module_logger.info(
            _format_log_message(
                "intermediate_dataframe_exported",
                dataframe=name,
                rows=len(df),
            )
        )
    except Exception:
        _module_logger.exception(
            _format_log_message(
                "intermediate_dataframe_export_failed",
                dataframe=name,
            )
        )


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
            "step_end  ",
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
            auth_scheme="Bearer",
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
