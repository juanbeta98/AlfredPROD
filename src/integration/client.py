import logging
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    def _parse_json_response(response: requests.Response) -> Dict[str, Any]:
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
