import logging
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ResultSender:
    """
    Client responsible for sending optimization results to ALFRED API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        auth_scheme: str = "Token",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        """
        Args:
            base_url: Base API URL
            api_key: Optional API token
            auth_scheme: Authorization scheme (e.g., "Token", "Bearer")
            timeout: Request timeout in seconds
            max_retries: Automatic retries on failure
            backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if api_key:
            if " " in api_key:
                self.headers["Authorization"] = api_key
            else:
                self.headers["Authorization"] = f"{auth_scheme} {api_key}"

        self.session = self._build_session(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    def send_results(
        self,
        results: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends optimization results to ALFRED API.

        Args:
            results: Final optimization output payload (already formatted)
            request_id: Optional request identifier for traceability

        Returns:
            API response as dict
        """
        endpoint = self.base_url
        payload = self._build_payload(results=results, request_id=request_id)

        logger.debug("Endpoint: %s", endpoint)
        logger.debug("Payload size: %s characters", len(str(payload)))

        try:
            response = self.session.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )

            if not response.ok:
                self._log_http_error(response)
                response.raise_for_status()

            result = self._parse_json_response(response)
            logger.info(
                "results_sent status_code=%s response_keys=%s",
                response.status_code,
                sorted(result.keys()) if isinstance(result, dict) else None,
            )
            
            return result

        except requests.exceptions.Timeout as exc:
            logger.error("Timeout while sending optimization results")
            raise RuntimeError("ALFRED API timeout while sending results") from exc

        except requests.exceptions.RequestException as exc:
            logger.error(
                "Request error while sending optimization results",
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
            allowed_methods=("POST",),
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retries)

        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    # def _build_endpoint(self) -> str:
    #     """
    #     Build full endpoint URL.
    #     """
    #     return f"{self.base_url}/optimization/output"

    @staticmethod
    def _build_payload(
        *,
        results: Dict[str, Any],
        request_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build request payload.
        """
        return {
            "request_id": request_id,
            **results,
        }

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
