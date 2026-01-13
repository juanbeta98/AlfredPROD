import pandas as pd
import sys
import logging
from typing import Any, Dict, Tuple

from src.config import Config
from src.integration.client import ALFREDAPIClient
from src.integration.sender import ResultSender
from src.data.parsing.input_parser import InputParser
from src.data.validation.validator import InputValidator
from src.data.validation.rules.generic import RequiredFieldRule, NonEmptyRowRule
from src.data.validation.rules.domain import CreatedBeforeScheduleRule
from src.data.formatting.output_formatter import OutputFormatter
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

    raw_input: list[dict[str | Any, str | Any]] | None = None
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
            raw_input = load_local_input(Config.LOCAL_DATA_PATH)

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
    # 4. Validate input (business & data rules)
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

            CreatedBeforeScheduleRule(minimum_delta_hours=2.0),
        ]

        validator = InputValidator(rules=rules)
        valid_df, invalid_df, validation_report = validator.validate(input_df)

        logger.info(
            "Input validation completed",
            extra=validation_report,
        )

        # Optional: persist or report invalid rows
        if not invalid_df.empty:
            logger.warning(
                "Some input records failed validation",
                extra={"invalid_rows": len(invalid_df)},
            )
            # save_invalid_records(invalid_df, request_id)

        if valid_df.empty:
            raise ValueError("All input records failed validation")

        input_df = valid_df

    except Exception as exc:
        logger.exception("Input validation failed")
        report_failure_if_possible(
            request_id=request_id,
            error="INPUT_VALIDATION_FAILED",
            details=str(exc),
        )
        return 4

    # --------------------------------------------------
    # 5. Run optimization
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
        return 5

    # --------------------------------------------------
    # 6. Format output
    # --------------------------------------------------
    try:
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
        logger.exception("Failed to format output payload")
        report_failure_if_possible(
            request_id=request_id,
            error="OUTPUT_FORMATTING_FAILED",
            details=str(exc),
        )
        return 6

    # --------------------------------------------------
    # 7. Deliver output
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
        return 7

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

def load_local_input(
    local_path: str,
) -> list[dict[str | Any, str | Any]]:
    """
    Load local input for development/testing.
    """
    # raise NotImplementedError("Local input loading not implemented")
    import csv
    # import json  # optional if you want to output JSON

    # Open the CSV file
    with open(local_path, mode='r', newline='') as file:
        # Use DictReader to read rows as dictionaries
        csv_reader = csv.DictReader(file)
        
        # Convert to a list of dictionaries
        data = [row for row in csv_reader]

    # Print as Python dictionary
    # print(data)

    # Optional: Convert to JSON string
    # json_data = json.dumps(data, indent=4)
    # print(json_data)

    return data


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
