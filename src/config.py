import os
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from src.datetime_utils import utc_to_colombia_timestamp

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
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "")
    API_BASE_URL: str = os.getenv("API_BASE_URL", "").rstrip("/")
    SERVICES_ENDPOINT: str = os.getenv("SERVICES_ENDPOINT", "")
    ALFREDS_ENDPOINT: str = os.getenv("ALFREDS_ENDPOINT", "")

    if not SERVICES_ENDPOINT:
        if API_BASE_URL:
            SERVICES_ENDPOINT = f"{API_BASE_URL}/services-algorithm"
        else:
            SERVICES_ENDPOINT = API_ENDPOINT

    if not ALFREDS_ENDPOINT:
        if API_BASE_URL:
            ALFREDS_ENDPOINT = f"{API_BASE_URL}/alfreds"
        elif API_ENDPOINT:
            base_url = API_ENDPOINT.rsplit("/", 1)[0]
            ALFREDS_ENDPOINT = f"{base_url}/alfreds" if base_url else ""

    if not API_ENDPOINT:
        API_ENDPOINT = SERVICES_ENDPOINT

    API_TOKEN: str = os.getenv("API_TOKEN", '')

    # --------------------------------------------------
    # Raw environment variables (strings only)
    # --------------------------------------------------
    _DEPARTMENT_RAW = os.getenv("DEPARTMENT")
    _START_DATE_RAW = os.getenv("START_DATE")
    _END_DATE_RAW = os.getenv("END_DATE")
    _SCHEDULE_DATE_RAW = os.getenv("SCHEDULE_DATE")

    # --------------------------------------------------
    # Parsed / typed configuration
    # --------------------------------------------------
    DEPARTMENT: Optional[int] = None
    START_DATE: Optional[datetime] = None
    END_DATE: Optional[datetime] = None
    SCHEDULE_DATE: Optional[date] = None

    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "3"))

    # --------------------------------------------------
    # Execution Mode
    # --------------------------------------------------
    USE_API: bool = os.getenv("USE_API", "false").lower() == "true"
    IGNORE_API_PAYLOAD_DRIVER: bool = (
        os.getenv("IGNORE_API_PAYLOAD_DRIVER", "false").lower() == "true"
    )
    REQUEST_ID: str | None = os.getenv("REQUEST_ID")
    WRITE_VALIDATION_REPORTS: bool = (
        os.getenv("WRITE_VALIDATION_REPORTS", "false").lower() == "true"
    )
    WRITE_INTERMEDIATE_DATAFRAMES: bool = (
        os.getenv(
            "WRITE_INTERMEDIATE_DATAFRAMES",
            os.getenv("EXPORT_INTERMEDIATE_DATAFRAMES", "false"),
        ).lower()
        == "true"
    )
    # Backward-compatible alias for previous env var naming.
    EXPORT_INTERMEDIATE_DATAFRAMES: bool = WRITE_INTERMEDIATE_DATAFRAMES
    WRITE_MODEL_SOLUTION: bool = (
        os.getenv("WRITE_MODEL_SOLUTION", "false").lower() == "true"
    )
    INTERMEDIATE_EXPORT_BASE_DIR: str = os.getenv(
        "INTERMEDIATE_EXPORT_BASE_DIR",
        "./data/intermediate_exports",
    )
    ARTIFACT_TIMEZONE: str = os.getenv("ARTIFACT_TIMEZONE", "America/Bogota")

    # --------------------------------------------------
    # Local Development
    # --------------------------------------------------
    LOCAL_INPUT_DIR: str = os.getenv("LOCAL_INPUT_DIR", "./data/model_input")
    LOCAL_INPUT_FILE: str = os.getenv("LOCAL_INPUT_FILE", "input.csv")
    LOCAL_DRIVER_DIRECTORY_FILE: str = os.getenv(
        "LOCAL_DRIVER_DIRECTORY_FILE",
        "./data/model_input/driver_directory.csv",
    )
    LOCAL_OUTPUT_DIR: str = os.getenv("LOCAL_OUTPUT_DIR", "./data/model_output")
    LOCAL_OUTPUT_FILE: str = os.getenv("LOCAL_OUTPUT_FILE", "output.csv")
    _LOCAL_DATA_PATH_LEGACY: str | None = os.getenv("LOCAL_DATA_PATH")
    _LOCAL_OUTPUT_PATH_LEGACY: str | None = os.getenv("LOCAL_OUTPUT_PATH")
    _LOCAL_INPUT_FILE_HAS_DIR: bool = bool(os.path.dirname(LOCAL_INPUT_FILE))
    _LOCAL_OUTPUT_FILE_HAS_DIR: bool = bool(os.path.dirname(LOCAL_OUTPUT_FILE))
    LOCAL_INPUT_PATH: str = _LOCAL_DATA_PATH_LEGACY or (
        LOCAL_INPUT_FILE
        if os.path.isabs(LOCAL_INPUT_FILE) or _LOCAL_INPUT_FILE_HAS_DIR
        else os.path.join(LOCAL_INPUT_DIR, LOCAL_INPUT_FILE)
    )
    LOCAL_OUTPUT_PATH: str = _LOCAL_OUTPUT_PATH_LEGACY or (
        LOCAL_OUTPUT_FILE
        if os.path.isabs(LOCAL_OUTPUT_FILE) or _LOCAL_OUTPUT_FILE_HAS_DIR
        else os.path.join(LOCAL_OUTPUT_DIR, LOCAL_OUTPUT_FILE)
    )
    DRIVER_DIRECTORY_FALLBACK_PATH: str = os.getenv(
        "DRIVER_DIRECTORY_FALLBACK_PATH",
        LOCAL_DRIVER_DIRECTORY_FILE,
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
            if not cls.SERVICES_ENDPOINT:
                raise RuntimeError("USE_API=true but SERVICES_ENDPOINT is not set")

            if not cls.ALFREDS_ENDPOINT:
                raise RuntimeError("USE_API=true but ALFREDS_ENDPOINT is not set")

            if not cls.API_TOKEN:
                raise RuntimeError("USE_API=true but API_TOKEN is not set")
        if cls.REQUEST_TIMEOUT <= 0:
            raise RuntimeError("REQUEST_TIMEOUT must be a positive integer")

        if cls.WRITE_INTERMEDIATE_DATAFRAMES and not cls.INTERMEDIATE_EXPORT_BASE_DIR.strip():
            raise RuntimeError(
                "WRITE_INTERMEDIATE_DATAFRAMES=true but INTERMEDIATE_EXPORT_BASE_DIR is empty"
            )

        try:
            ZoneInfo(cls.ARTIFACT_TIMEZONE)
        except ZoneInfoNotFoundError as exc:
            raise RuntimeError(
                f"Invalid ARTIFACT_TIMEZONE: {cls.ARTIFACT_TIMEZONE!r}"
            ) from exc
        
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
        cls.SCHEDULE_DATE = cls._parse_date(
            cls._SCHEDULE_DATE_RAW,
            var_name="SCHEDULE_DATE",
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
            ts = utc_to_colombia_timestamp(value, errors="raise")
            return ts.to_pydatetime()
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO datetime for {var_name}: {value!r}"
            ) from exc

    @staticmethod
    def _parse_date(
        value: Optional[str],
        *,
        var_name: str,
    ) -> Optional[date]:
        if value is None or value == "":
            return None

        try:
            if "T" in value:
                ts = utc_to_colombia_timestamp(value, errors="raise")
                return ts.date()
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO date for {var_name}: {value!r}"
            ) from exc
