#!/usr/bin/env python3
"""
check_availability.py — ALFRED time slot availability checker.

Validates whether a new service can be assigned at a customer's desired time
and, if not, returns the list of feasible 30-minute slots for the day.

Supports two request formats (auto-detected):

  API format (address-centric, from customer-facing endpoint):
  ------------------------------------------------------------
  {
      "department_id": 25,
      "department_name": "CUNDINAMARCA",
      "date": "2026-03-09T14:30:00-05:00",
      "start_address": {
          "id": 123456,
          "name": "Calle 123",
          "city": "Bogotá",
          "department": "CUNDINAMARCA",
          "point": {"x": -74.08, "y": 4.71, "srid": 4326}
      },
      "end_address": {
          "id": 123456,
          "name": "Calle 123",
          "city": "Bogotá",
          "department": "CUNDINAMARCA",
          "point": {"x": -74.05, "y": 4.62, "srid": 4326}
      }
  }

  Internal format (explicit labors, for programmatic use):
  ---------------------------------------------------------
  {
      "service_id": "SVC-001",
      "department_code": "25",
      "desired_slot": "2026-03-04T09:00:00-05:00",
      "as_of_time":   "2026-03-04T07:30:00-05:00",
      "labors": [
          {
              "labor_sequence": 1,
              "labor_category": "VEHICLE_TRANSPORTATION",
              "labor_type": "TRASLADO_VEHICULO",
              "map_start_point": "POINT (-74.0817 4.6097)",
              "map_end_point":   "POINT (-74.1234 4.6543)"
          }
      ]
  }

Output JSON format
------------------
{
    "service_id": "SVC-001",
    "desired_slot": {
        "slot_time": "2026-03-04T09:00:00-05:00",
        "feasible": false,
        "reason": "no_driver"
    },
    "scan_performed": true,
    "total_slots_checked": 24,
    "schedule_date": "2026-03-04",
    "feasible_slots": [
        {"slot_time": "2026-03-04T10:00:00-05:00", "feasible": true},
        {"slot_time": "2026-03-04T10:30:00-05:00", "feasible": true}
    ],
    "error": null
}
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from src.config import Config
from src.availability import check_availability
from src.availability.models import AvailabilityResponse, LaborRequest, ServiceRequest
from src.availability.request_parser import parse_api_request

logger = logging.getLogger(__name__)


def main() -> int:
    Config.validate()
    Config.configure_logging()

    parser = argparse.ArgumentParser(
        description="ALFRED time slot availability checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--request",
        required=True,
        metavar="PATH",
        help="Path to the availability request JSON file",
    )
    args = parser.parse_args()

    # Load request file
    try:
        with open(args.request, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load request file %s: %s", args.request, exc)
        return 1

    # Parse into ServiceRequest — auto-detect format by key presence
    try:
        request = _parse_request(data)
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Invalid request format: %s", exc)
        _print_error(data.get("service_id") or data.get("department_id", "unknown"), str(exc))
        return 1

    # Run availability check
    try:
        response = check_availability(request)
    except Exception as exc:
        logger.exception("availability_check_failed")
        _print_error(request.service_id, str(exc))
        return 1

    print(json.dumps(_serialize_response(response), indent=2, ensure_ascii=False))
    return 0


def _parse_request(data: Dict[str, Any]) -> ServiceRequest:
    """
    Auto-detect and parse either the API format (department_id + addresses)
    or the internal format (department_code + labors array).
    """
    if "department_id" in data:
        # API / customer-facing format
        return parse_api_request(data)

    # Internal / programmatic format
    labors = [
        LaborRequest(
            labor_sequence=int(lb["labor_sequence"]),
            labor_category=lb["labor_category"],
            map_start_point=lb["map_start_point"],
            map_end_point=lb["map_end_point"],
            labor_type=lb.get("labor_type"),
            estimated_time=float(lb["estimated_time"]) if lb.get("estimated_time") is not None else None,
            labor_name=lb.get("labor_name"),
        )
        for lb in data["labors"]
    ]
    return ServiceRequest(
        service_id=str(data["service_id"]),
        department_code=str(data["department_code"]),
        labors=labors,
        desired_slot=datetime.fromisoformat(data["desired_slot"]),
        as_of_time=datetime.fromisoformat(data["as_of_time"]),
    )


def _serialize_response(response: AvailabilityResponse) -> Dict[str, Any]:
    return {
        "service_id": response.service_id,
        "desired_slot": {
            "slot_time": _fmt_ts(response.desired_slot_result.slot_time),
            "feasible": response.desired_slot_result.feasible,
            "reason": response.desired_slot_result.reason,
        },
        "scan_performed": response.scan_performed,
        "total_slots_checked": response.total_slots_checked,
        "schedule_date": response.schedule_date_str,
        "feasible_slots": [
            {
                "slot_time": _fmt_ts(s.slot_time),
                "feasible": True,
            }
            for s in response.feasible_slots
        ],
        "error": response.error,
    }


def _print_error(service_id: str, message: str) -> None:
    print(
        json.dumps(
            {
                "service_id": service_id,
                "desired_slot": None,
                "scan_performed": False,
                "total_slots_checked": 0,
                "schedule_date": None,
                "feasible_slots": [],
                "error": message,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _fmt_ts(ts: Optional[Any]) -> Optional[str]:
    if ts is None:
        return None
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


if __name__ == "__main__":
    sys.exit(main())
