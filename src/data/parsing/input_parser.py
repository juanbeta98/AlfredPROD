import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class InputParser:
    """
    Input parser for ALFRED optimization payloads.

    Transforms API-compliant JSON input into a flat tabular
    structure suitable for validation and optimization.
    """

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    @staticmethod
    def parse(json_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse optimization input payload into a DataFrame.

        Args:
            json_data: Raw JSON payload from API or local source

        Returns:
            df: Flattened DataFrame (service × labor)
            metadata: Non-tabular metadata
        """
        if not isinstance(json_data, dict):
            raise ValueError("Input data must be a JSON object")

        services = json_data.get("data")

        if services is None:
            raise KeyError("Expected key 'data' not found in input payload")

        if not isinstance(services, list):
            raise TypeError("'data' must be a list of services")

        if not services:
            logger.warning("Input payload contains no services")
            return pd.DataFrame(), {"raw": json_data}

        records: List[Dict[str, Any]] = []

        for service in services:
            service_base = InputParser._parse_service_base(service)

            labors = service.get("serviceLabors", [])
            if not isinstance(labors, list):
                logger.warning(
                    "Invalid serviceLabors format",
                    extra={"service_id": service_base.get("service_id")},
                )
                continue

            for labor in labors:
                record = InputParser._parse_labor_record(
                    service_base=service_base,
                    labor=labor,
                )
                records.append(record)

        if not records:
            logger.warning("No labor records extracted from payload")
            return pd.DataFrame(), {"raw": json_data}

        df = pd.json_normalize(records)

        metadata = {
            "service_count": len(services),
        }

        logger.info("input_parsed rows=%s", len(df))

        return df, metadata

    # --------------------------------------------------
    # Parsing helpers
    # --------------------------------------------------

    @staticmethod
    def _parse_service_base(service: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse service-level fields common to all labor records.
        """
        base: Dict[str, Any] = {
            "service_id": service.get("service_id"),
            "state": service.get("state"),
            "created_at": service.get("created_at"),
            'city': service.get('start_address', {}).get('city', ''),  # used in some algorithms
        }

        base.update(
            InputParser._parse_address(
                service.get("start_address"),
                prefix="start_address",
            )
        )

        base.update(
            InputParser._parse_address(
                service.get("end_address"),
                prefix="end_address",
            )
        )

        return base

    @staticmethod
    def _parse_labor_record(
        *,
        service_base: Dict[str, Any],
        labor: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge service-level and labor-level fields into one record.
        """
        record = service_base.copy()

        actual_start = labor.get("actual_start") or labor.get("startDate") or labor.get("start_date")
        actual_end = labor.get("actual_end") or labor.get("endDate") or labor.get("end_date")

        record.update(
            {
                "labor_id": labor.get("id"),
                "labor_type": labor.get("labor_id"),
                "labor_name": labor.get("labor_name"),
                "labor_category": labor.get("labor_category"),
                "schedule_date": labor.get("schedule_date"),
                "labor_sequence": labor.get("labor_sequence", 1),
                "assigned_driver": labor.get("alfred"),
                "actual_start": actual_start,
                "actual_end": actual_end,
            }
        )

        shop = labor.get("shop", {})
        if shop == None:
            record.update({
                "shop_id": None,
                "address_id": None,
                "address_point": None,
            })
        else:
            record["shop_id"] = shop.get("id", None)

            record.update(
                InputParser._parse_address(
                    labor.get("shop_address"),
                    prefix="shop_address",
                )
            )

        return record

    @staticmethod
    def _parse_address(
        address: Dict[str, Any] | None,
        *,
        prefix: str,
    ) -> Dict[str, Any]:
        """
        Parse an address object into flattened fields.

        Expected structure:
        {
            "id": int,
            "point": { "x": lon, "y": lat }
        }
        """
        if not isinstance(address, dict):
            return {
                f"{prefix}_id": None,
                f"{prefix}_point": None,
            }

        point = address.get("point", {})

        lon = point.get("x")
        lat = point.get("y")

        point_wkt = (
            f"POINT ({lon} {lat})"
            if lon is not None and lat is not None
            else None
        )

        return {
            f"{prefix}_id": address.get("id"),
            f"{prefix}_point": point_wkt,
        }
