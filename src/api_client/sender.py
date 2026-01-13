import logging
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ResultSender:
    """
    Client responsible for sending optimization results to ALFRED API
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        """
        Args:
            base_url: Base API URL
            api_key: Optional Bearer token
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
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Configure session with retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def send_results(
        self,
        results: Dict,
        request_id: Optional[str] = None,
    ) -> Dict:
        """
        Sends optimization results to ALFRED API.

        Args:
            results: Final optimization output payload (already formatted)
            request_id: Optional request identifier for traceability

        Returns:
            API response as dict
        """
        endpoint = f"{self.base_url}/optimization/output"

        payload = {
            "request_id": request_id,
            **results,
        }

        logger.info("Sending optimization results to ALFRED API")
        logger.debug(f"Endpoint: {endpoint}")
        logger.debug(f"Payload size: {len(str(payload))} characters")

        try:
            response = self.session.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            logger.info(
                "Optimization results successfully sent "
                f"(request_id={request_id})"
            )

            return response.json()

        except requests.exceptions.Timeout as exc:
            logger.error("Timeout while sending optimization results")
            raise RuntimeError("ALFRED API timeout while sending results") from exc

        except requests.exceptions.HTTPError as exc:
            logger.error(
                "HTTP error while sending optimization results "
                f"(status={response.status_code}, body={response.text})"
            )
            raise

        except requests.exceptions.RequestException as exc:
            logger.error(f"Request error while sending results: {exc}")
            raise
