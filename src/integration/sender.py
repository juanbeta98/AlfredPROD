import logging
from typing import Any, Optional

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
        results: Any,
        request_id: Optional[str] = None,
    ) -> Any:
        """
        Sends optimization results to ALFRED API.

        Args:
            results: Final optimization output payload (already formatted)
            request_id: Optional request identifier for traceability

        Returns:
            API response body
        """
        endpoint = self.base_url
        payload = self._build_payload(results=results, request_id=request_id)
        payload_status = payload.get("status") if isinstance(payload, dict) else None

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
            logged_payload_status = "success" if response.status_code == 200 else payload_status
            logger.info(
                "results_sent status_code=%s payload_status=%s",
                response.status_code,
                logged_payload_status,
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
        results: Any,
        request_id: Optional[str],
    ) -> Any:
        """
        Build request payload.
        """
        if results is None:
            return {"data": []}

        # Preserve failure reporting payloads.
        if isinstance(results, dict) and str(results.get("status", "")).lower() == "failed":
            payload = dict(results)
            if request_id and not payload.get("request_id"):
                payload["request_id"] = request_id
            return payload

        if isinstance(results, dict):
            if "data" in results:
                payload = dict(results)
                data = payload.get("data")
                if data is None:
                    payload["data"] = []
                elif isinstance(data, dict) and ResultSender._looks_like_service_payload(data):
                    payload["data"] = [data]
                return payload

            if ResultSender._looks_like_service_payload(results):
                return {"data": [results]}

            return results

        if isinstance(results, list):
            return {"data": results}

        return results

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
    def _log_http_error(response: requests.Response) -> None:
        """
        Log HTTP error details safely.
        """
        body_preview = (response.text or "")[:500]
        logger.error(
            "HTTP error from ALFRED API status_code=%s response_body=%s",
            response.status_code,
            body_preview,
        )

    @staticmethod
    def _looks_like_service_payload(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return "service_id" in value and isinstance(value.get("serviceLabors"), list)
