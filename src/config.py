import os
import logging
from dotenv import load_dotenv
from datetime import datetime

from typing import Any, Dict, Optional

# Load .env file early
load_dotenv()


class Config:
    """
    Centralized configuration for optimization integration.
    Values are loaded from environment variables with safe defaults.
    """

    # --------------------------------------------------
    # API Configuration
    # --------------------------------------------------
    API_ENDPOINT: str = os.getenv(
        "API_ENDPOINT",
        '',
    )
    API_TOKEN: str = os.getenv("API_TOKEN", '')

    # --------------------------------------------------
    # Raw environment variables (strings only)
    # --------------------------------------------------
    _DEPARTMENT_RAW = os.getenv("DEPARTMENT")
    _START_DATE_RAW = os.getenv("START_DATE")
    _END_DATE_RAW = os.getenv("END_DATE")

    # --------------------------------------------------
    # Parsed / typed configuration
    # --------------------------------------------------
    DEPARTMENT: Optional[int] = None
    START_DATE: Optional[datetime] = None
    END_DATE: Optional[datetime] = None

    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "3"))

    # --------------------------------------------------
    # Execution Mode
    # --------------------------------------------------
    USE_API: bool = os.getenv("USE_API", "false").lower() == "true"
    REQUEST_ID: str | None = os.getenv("REQUEST_ID")
    WRITE_VALIDATION_REPORTS: bool = (
        os.getenv("WRITE_VALIDATION_REPORTS", "false").lower() == "true"
    )

    # --------------------------------------------------
    # Local Development
    # --------------------------------------------------
    LOCAL_INPUT_DIR: str = os.getenv("LOCAL_INPUT_DIR", "./data/model_input")
    LOCAL_INPUT_FILE: str = os.getenv("LOCAL_INPUT_FILE", "input.csv")
    LOCAL_OUTPUT_DIR: str = os.getenv("LOCAL_OUTPUT_DIR", "./data/model_output")
    LOCAL_OUTPUT_FILE: str = os.getenv("LOCAL_OUTPUT_FILE", "output.csv")
    _LOCAL_DATA_PATH_LEGACY: str | None = os.getenv("LOCAL_DATA_PATH")
    _LOCAL_OUTPUT_PATH_LEGACY: str | None = os.getenv("LOCAL_OUTPUT_PATH")
    LOCAL_INPUT_PATH: str = _LOCAL_DATA_PATH_LEGACY or (
        LOCAL_INPUT_FILE
        if os.path.isabs(LOCAL_INPUT_FILE)
        else os.path.join(LOCAL_INPUT_DIR, LOCAL_INPUT_FILE)
    )
    LOCAL_OUTPUT_PATH: str = _LOCAL_OUTPUT_PATH_LEGACY or (
        LOCAL_OUTPUT_FILE
        if os.path.isabs(LOCAL_OUTPUT_FILE)
        else os.path.join(LOCAL_OUTPUT_DIR, LOCAL_OUTPUT_FILE)
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    @classmethod
    def validate(cls) -> None:
        """
        Validate critical configuration before execution.
        """
        if cls.USE_API:
            if not cls.API_ENDPOINT:
                raise RuntimeError("USE_API=true but API_ENDPOINT is not set")

            if not cls.API_TOKEN:
                raise RuntimeError("USE_API=true but API_TOKEN is not set")
        if cls.REQUEST_TIMEOUT <= 0:
            raise RuntimeError("REQUEST_TIMEOUT must be a positive integer")
        
        cls.DEPARTMENT = cls._parse_int(
            cls._DEPARTMENT_RAW,
            var_name="DEPARTMENT",
        )

        cls.START_DATE = cls._parse_datetime(
            cls._START_DATE_RAW,
            var_name="START_DATE",
        )

        cls.END_DATE = cls._parse_datetime(
            cls._END_DATE_RAW,
            var_name="END_DATE",
        )

        if cls.START_DATE and cls.END_DATE:
            if cls.START_DATE > cls.END_DATE:
                raise ValueError("START_DATE must be before END_DATE")

    # --------------------------------------------------
    # Logging Setup
    # --------------------------------------------------
    
    @classmethod
    def configure_logging(cls) -> None:
        """
        Configure application-wide logging.
        """
        from src.logging_utils import ContextFilter, setup_logging_context

        setup_logging_context()
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(name)s | run_id=%(run_id)s | %(message)s",
        )
        logging.getLogger().addFilter(ContextFilter())

    # --------------------------------------------------
    # Parsers
    # --------------------------------------------------

    @staticmethod
    def _parse_int(
        value: Optional[str],
        *,
        var_name: str,
    ) -> Optional[int]:
        if value is None or value == "":
            return None

        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer for {var_name}: {value!r}"
            ) from exc

    @staticmethod
    def _parse_datetime(
        value: Optional[str],
        *,
        var_name: str,
    ) -> Optional[datetime]:
        if value is None or value == "":
            return None

        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO datetime for {var_name}: {value!r}"
            ) from exc
