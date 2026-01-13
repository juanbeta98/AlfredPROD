import logging
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ALFREDAPIClient:
    """
    Client to consume ALFRED optimization input endpoints
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
            base_url: Base API URL (e.g. https://api-alfred.segurosbolivar.com)
            api_key: Optional Bearer token
            timeout: Request timeout in seconds
            max_retries: Number of automatic retries on failure
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
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_optimization_data(self, request_id: Optional[str] = None) -> Dict:
        """
        Fetch optimization input data from ALFRED API.

        Args:
            request_id: Optional request identifier for tracking

        Returns:
            Parsed JSON response as dict
        """
        endpoint = f"{self.base_url}/optimization/input"

        params = {}
        if request_id:
            params["request_id"] = request_id

        logger.info("Requesting optimization data from ALFRED API")
        logger.debug("Endpoint: %s", endpoint)
        logger.debug("Params: %s", params)

        try:
            response = self.session.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()

            services_count = len(data.get("services", []))
            logger.info(
                "Received optimization data",
                extra={"services_count": services_count},
            )

            return data

        except requests.exceptions.Timeout as exc:
            logger.error("Timeout while calling optimization input endpoint")
            raise RuntimeError("ALFRED API timeout") from exc

        except requests.exceptions.HTTPError as exc:
            resp = exc.response
            logger.error(
                "HTTP error from ALFRED API",
                extra={
                    "status_code": resp.status_code if resp else None,
                    "response_body": resp.text if resp else None,
                },
            )
            raise

        except requests.exceptions.RequestException as exc:
            logger.error(
                "Request error while calling ALFRED API",
                exc_info=exc,
            )
            raise

