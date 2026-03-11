from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.data.loading.master_data_loader import MasterData
from src.optimization.settings.solver_settings import OptimizationSettings


@dataclass(frozen=True)
class LaborRequest:
    """A single labor within a customer service request."""

    labor_sequence: int
    labor_category: str       # "VEHICLE_TRANSPORTATION" or other
    map_start_point: str      # WKT POINT, e.g. "POINT (-74.08 4.71)"
    map_end_point: str        # WKT POINT
    labor_type: Optional[str] = None
    estimated_time: Optional[float] = None   # minutes, for non-VT labors
    labor_name: Optional[str] = None


@dataclass(frozen=True)
class ServiceRequest:
    """
    A full customer service request with one or more labors in sequence.

    Attributes:
        service_id      : Caller-supplied identifier (used for logging/tracing).
        department_code : Department code string matching existing pipeline values
                          ("25", "76", "5", ...).
        labors          : Ordered list of LaborRequest (by labor_sequence).
        desired_slot    : Customer's preferred schedule_date for the first labor
                          (timezone-aware).
        as_of_time      : Current moment — slot scan starts here (timezone-aware).
    """

    service_id: str
    department_code: str
    labors: List[LaborRequest]
    desired_slot: datetime
    as_of_time: datetime


@dataclass(frozen=True)
class TimeSlotResult:
    """Result of a feasibility probe at a single time slot."""

    slot_time: datetime
    feasible: bool
    reason: Optional[str] = None  # "no_driver" | "workday_end" | "not_inserted"


@dataclass
class AvailabilityResponse:
    """
    Full response to a ServiceRequest.

    Attributes:
        service_id           : Echoed from the request.
        desired_slot_result  : Feasibility result for the customer's preferred time.
        feasible_slots       : All feasible 30-min slots from as_of_time to workday_end
                               (populated only if desired slot is infeasible).
        scan_performed       : True if a full interval scan was run.
        total_slots_checked  : Number of slots examined (including desired slot).
        schedule_date_str    : The scheduling day examined ("YYYY-MM-DD").
        error                : Non-None if an unrecoverable error occurred.
    """

    service_id: str
    desired_slot_result: TimeSlotResult
    feasible_slots: List[TimeSlotResult] = field(default_factory=list)
    scan_performed: bool = False
    total_slots_checked: int = 0
    schedule_date_str: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ScheduleState:
    """
    Frozen snapshot of the preassigned schedule for a given city+day.
    Reused across all slot probes; never mutated.
    """

    base_labors_df: pd.DataFrame    # preassigned labors
    base_moves_df: pd.DataFrame     # preassigned movement records
    master_data: MasterData         # directorio_df, duraciones_df, dist_dict
    settings: OptimizationSettings
    day_str: str                    # "YYYY-MM-DD"
    department_code: str
