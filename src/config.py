import os
import logging
from dotenv import load_dotenv

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
    API_BASE_URL: str = os.getenv(
        "ALFRED_API_URL",
        "https://api-alfred.segurosbolivar.com",
    )
    API_KEY: str | None = os.getenv("ALFRED_API_KEY")

    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "3"))

    # --------------------------------------------------
    # Execution Mode
    # --------------------------------------------------
    USE_API: bool = os.getenv("USE_API", "false").lower() == "true"
    REQUEST_ID: str | None = os.getenv("REQUEST_ID")

    # --------------------------------------------------
    # Local Development
    # --------------------------------------------------
    LOCAL_DATA_PATH: str = os.getenv(
        "LOCAL_DATA_PATH",
        "./data/input.csv",
    )
    LOCAL_OUTPUT_PATH: str = os.getenv(
        "LOCAL_OUTPUT_PATH",
        "./data/output.csv",
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
            if not cls.API_BASE_URL:
                raise RuntimeError("USE_API=true but ALFRED_API_URL is not set")

            if not cls.API_KEY:
                raise RuntimeError("USE_API=true but ALFRED_API_KEY is not set")

        if cls.REQUEST_TIMEOUT <= 0:
            raise RuntimeError("REQUEST_TIMEOUT must be a positive integer")

    # --------------------------------------------------
    # Logging Setup
    # --------------------------------------------------
    @classmethod
    def configure_logging(cls) -> None:
        """
        Configure application-wide logging.
        """
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
