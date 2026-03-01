import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.datetime_utils import utc_to_colombia_timestamp

logger = logging.getLogger(__name__)


class ALFREDAPIClient:
    """
    HTTP client for ALFRED optimization input API.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_token: str,
        *,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        """
        Args:
            endpoint_url: Full API endpoint URL
            api_token: Authentication token
            timeout: Request timeout in seconds
            max_retries: Max retry attempts
            backoff_factor: Retry backoff factor
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_token}",
        }

        self.session = self._build_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def get_optimization_data(
        self,
        *,
        department: Optional[int | None] = None,
        start_date: Optional[datetime | None] = None,
        end_date: Optional[datetime | None] = None,
    ) -> Dict[str, Any]:
        """
        Fetch optimization input data from ALFRED API.

        Args:
            department: Optional department filter
            start_date: Optional start datetime (ISO)
            end_date: Optional end datetime (ISO)

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: On timeout or invalid response
            requests.HTTPError: On non-success HTTP status
        """
        params = self._build_params(
            department=department,
            start_date=start_date,
            end_date=end_date,
        )

        logger.debug("Endpoint: %s", self.endpoint_url)
        logger.debug("Query params: %s", params)

        try:
            response = self.session.get(
                self.endpoint_url,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
            )

            if not response.ok:
                self._log_http_error(response)
                response.raise_for_status()

            payload = self._parse_json_response(response)
            logger.info(
                "fetch_optimization_data_completed department=%s start_date=%s end_date=%s",
                department,
                start_date.isoformat() if start_date is not None else None,
                end_date.isoformat() if end_date is not None else None,
            )
            return payload

        except requests.exceptions.Timeout as exc:
            logger.error("Timeout while calling ALFRED API")
            raise RuntimeError("ALFRED API request timed out") from exc

        except requests.exceptions.RequestException as exc:
            logger.error(
                "Request error while calling ALFRED API",
                exc_info=exc,
            )
            raise

    def get_driver_directory(
        self,
        *,
        active: Optional[bool] = True,
        schedule_date: Optional[date | datetime | str] = None,
        department: Optional[int | None] = None,
    ) -> list[Dict[str, Any]]:
        """
        Fetch driver directory (alfreds) data from ALFRED API.

        Args:
            active: Optional active filter
            schedule_date: Optional schedule date (YYYY-MM-DD)
            department: Optional department filter

        Returns:
            List of driver records

        Raises:
            RuntimeError: On timeout or invalid response
            requests.HTTPError: On non-success HTTP status
        """
        params = self._build_driver_params(
            active=active,
            schedule_date=schedule_date,
            department=department,
        )

        logger.debug("Endpoint: %s", self.endpoint_url)
        logger.debug("Query params: %s", params)

        try:
            response = self.session.get(
                self.endpoint_url,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
            )

            if not response.ok:
                self._log_http_error(response)
                response.raise_for_status()

            payload = self._parse_json_response(response)
            if isinstance(payload, dict) and "results" in payload:
                payload = payload.get("results")

            if not isinstance(payload, list):
                raise RuntimeError("Driver directory response must be a list")

            logger.info(
                "fetch_driver_directory_completed active=%s schedule_date=%s department=%s rows=%s",
                active,
                self._format_schedule_date(schedule_date),
                department,
                len(payload),
            )
            return payload

        except requests.exceptions.Timeout as exc:
            logger.error("Timeout while calling ALFRED driver directory API")
            raise RuntimeError("ALFRED driver directory API request timed out") from exc

        except requests.exceptions.RequestException as exc:
            logger.error(
                "Request error while calling ALFRED driver directory API",
                exc_info=exc,
            )
            raise

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _build_session(
        *,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """
        Build a requests session with retry configuration.
        """
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retries)

        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    @staticmethod
    def _build_params(
        *,
        department: Optional[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> Dict[str, str]:
        """
        Build query parameters dictionary.
        """
        params: Dict[str, str] = {}

        if department is not None:
            params["department"] = str(department)

        if start_date is not None:
            params["start_date"] = start_date.isoformat()

        if end_date is not None:
            params["end_date"] = end_date.isoformat()

        return params

    @staticmethod
    def _build_driver_params(
        *,
        active: Optional[bool],
        schedule_date: Optional[date | datetime | str],
        department: Optional[int],
    ) -> Dict[str, str]:
        params: Dict[str, str] = {}

        if active is not None:
            params["active"] = "true" if active else "false"

        if schedule_date is not None:
            params["schedule_date"] = ALFREDAPIClient._format_schedule_date(schedule_date)

        if department is not None:
            params["department"] = str(department)

        return params

    @staticmethod
    def _parse_json_response(response: requests.Response) -> Any:
        """
        Safely parse JSON response.
        """
        try:
            return response.json()
        except ValueError as exc:
            logger.error(
                "Failed to decode JSON response",
                extra={
                    "status_code": response.status_code,
                    "response_text": response.text[:500],
                },
            )
            raise RuntimeError("Invalid JSON response from API") from exc

    @staticmethod
    def _format_schedule_date(value: Optional[date | datetime | str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            ts = utc_to_colombia_timestamp(value, errors="coerce")
            return value.date().isoformat() if pd.isna(ts) else ts.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            try:
                if "T" in value:
                    ts = utc_to_colombia_timestamp(value, errors="coerce")
                    if not pd.isna(ts):
                        return ts.date().isoformat()
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
                return date.fromisoformat(value).isoformat()
            except ValueError:
                return value
        return str(value)

    @staticmethod
    def _log_http_error(response: requests.Response) -> None:
        """
        Log HTTP error details safely.
        """
        logger.error(
            "HTTP error from ALFRED API",
            extra={
                "status_code": response.status_code,
                "response_body": response.text[:500],
            },
        )
