import sys
import logging
from typing import Any, Dict, Tuple

from src.config import Config
from src.api_client.client import ALFREDAPIClient
from src.api_client.sender import ResultSender
from src.data_handlers.input_parser import InputParser
from src.data_handlers.output_formatter import OutputFormatter
from src.optimization.solver import OptimizationSolver


logger = logging.getLogger(__name__)


# ======================================================
# Entry point
# ======================================================
def main() -> int:
    """
    Main orchestration entrypoint.

    Returns:
        int: Process exit code (0 = success, non-zero = failure)
    """

    # --------------------------------------------------
    # 1. Bootstrap & validation
    # --------------------------------------------------
    try:
        Config.validate()
        Config.configure_logging()
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR)
        logger.error("Configuration error", exc_info=exc)
        return 1

    logger.info("Optimization pipeline starting")

    use_api = Config.USE_API
    request_id: str | None = None

    raw_input: Dict[str, Any] | None = None
    input_df = None
    metadata: Dict[str, Any] = {}

    # --------------------------------------------------
    # 2. Acquire input
    # --------------------------------------------------
    try:
        if use_api:
            logger.info("Execution mode: API")

            client = ALFREDAPIClient(
                base_url=Config.API_BASE_URL,
                api_key=Config.API_KEY,
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.API_MAX_RETRIES,
            )

            raw_input = client.get_optimization_data()
            request_id = raw_input.get("request_id")

        else:
            logger.info("Execution mode: LOCAL")
            raw_input = load_local_input()

    except Exception as exc:
        logger.exception("Failed to acquire input data")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_ACQUISITION_FAILED",
            details=str(exc),
        )
        return 2

    # --------------------------------------------------
    # 3. Parse input (schema-agnostic)
    # --------------------------------------------------
    try:
        input_df, metadata = InputParser.parse(raw_input)

        if input_df.empty:
            raise ValueError("Parsed input DataFrame is empty")

        logger.info(
            "Input parsed successfully",
            extra={"rows": len(input_df)},
        )

    except Exception as exc:
        logger.exception("Failed to parse input data")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_PARSING_FAILED",
            details=str(exc),
        )
        return 3

    # --------------------------------------------------
    # 4. Run optimization
    # --------------------------------------------------
    try:
        solver = OptimizationSolver(input_df)
        results, metrics = solver.solve()

        logger.info(
            "Optimization completed",
            extra={
                "rows_out": len(results) if hasattr(results, "__len__") else None,
                "metrics": metrics,
            },
        )

    except Exception as exc:
        logger.exception("Optimization failed")
        report_failure_if_possible(
            request_id=request_id,
            error="OPTIMIZATION_FAILED",
            details=str(exc),
        )
        return 4

    # --------------------------------------------------
    # 5. Format output
    # --------------------------------------------------
    try:
        output_payload = OutputFormatter.format(
            results=results,
            metadata={
                "metrics": metrics,
                **metadata,
            },
            request_id=request_id,
            status="completed",
        )

    except Exception as exc:
        logger.exception("Failed to format output payload")
        report_failure_if_possible(
            request_id=request_id,
            error="OUTPUT_FORMATTING_FAILED",
            details=str(exc),
        )
        return 5

    # --------------------------------------------------
    # 6. Deliver output
    # --------------------------------------------------
    try:
        if use_api:
            sender = ResultSender(
                base_url=Config.API_BASE_URL,
                api_key=Config.API_KEY,
                timeout=Config.REQUEST_TIMEOUT,
                max_retries=Config.API_MAX_RETRIES,
            )
            sender.send_results(output_payload, request_id=request_id)

        else:
            save_local_output(output_payload)

        logger.info("Output delivered successfully")

    except Exception as exc:
        logger.exception("Failed to deliver output")
        report_failure_if_possible(
            request_id=request_id,
            error="OUTPUT_DELIVERY_FAILED",
            details=str(exc),
        )
        return 6

    logger.info("Optimization pipeline finished successfully")
    return 0


# ======================================================
# Failure reporting (best-effort)
# ======================================================

def report_failure_if_possible(
    request_id: str | None,
    error: str,
    details: str,
) -> None:
    """
    Best-effort failure reporting.
    Does not raise.
    """
    if not Config.USE_API or not request_id:
        return

    try:
        sender = ResultSender(
            base_url=Config.API_BASE_URL,
            api_key=Config.API_KEY,
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
        # Never let failure reporting crash the process
        logger.error("Failed to report failure to API", exc_info=True)


# ======================================================
# Local I/O helpers (intentionally abstract)
# ======================================================

def load_local_input() -> Dict[str, Any]:
    """
    Load local input for development/testing.
    """
    raise NotImplementedError("Local input loading not implemented")


def save_local_output(payload: Dict[str, Any]) -> None:
    """
    Persist output locally for development/testing.
    """
    raise NotImplementedError("Local output saving not implemented")


# ======================================================
# Process exit
# ======================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
