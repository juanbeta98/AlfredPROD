"""
src/availability — Time slot availability module.

Public API
----------
check_availability(request) -> AvailabilityResponse
    Top-level entry point.  Loads the live schedule from the ALFRED API
    and checks whether the customer's desired time slot is feasible.
    If not, scans 30-minute intervals from as_of_time to workday_end and
    returns all feasible alternatives.

Typical usage
-------------
    from src.availability import check_availability
    from src.availability.models import LaborRequest, ServiceRequest
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/Bogota")
    request = ServiceRequest(
        service_id="SVC-001",
        department_code="25",
        labors=[
            LaborRequest(
                labor_sequence=1,
                labor_category="VEHICLE_TRANSPORTATION",
                labor_type="TRASLADO_VEHICULO",
                map_start_point="POINT (-74.0817 4.6097)",
                map_end_point="POINT (-74.1234 4.6543)",
            )
        ],
        desired_slot=datetime(2026, 3, 4, 9, 0, tzinfo=tz),
        as_of_time=datetime(2026, 3, 4, 7, 30, tzinfo=tz),
    )
    response = check_availability(request)
"""

from src.availability.models import AvailabilityResponse, ServiceRequest
from src.availability.request_parser import parse_api_request
from src.availability.schedule_loader import load_schedule_state
from src.availability.slot_scanner import scan_availability
from src.optimization.settings.solver_settings import OptimizationSettings


def check_availability(request: ServiceRequest) -> AvailabilityResponse:
    """
    Check time slot availability for a new service request.

    Loads the current live schedule from the ALFRED API, then probes the
    customer's desired time slot.  If infeasible, scans 30-minute intervals
    from request.as_of_time to workday_end and returns all feasible slots.

    Args:
        request: ServiceRequest with labor definitions, desired_slot, and
                 as_of_time.

    Returns:
        AvailabilityResponse with desired_slot_result and feasible_slots.

    Raises:
        ScheduleLoadError: If the live schedule cannot be loaded.
    """
    settings = OptimizationSettings(algorithm="INSERT")
    state = load_schedule_state(
        department_code=request.department_code,
        schedule_date=request.desired_slot.date(),
        settings=settings,
    )
    return scan_availability(request, state)


__all__ = [
    "check_availability",
    "parse_api_request",
    "ServiceRequest",
    "AvailabilityResponse",
]
