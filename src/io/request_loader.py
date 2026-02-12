import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.datetime_utils import utc_to_colombia_series, utc_to_colombia_timestamp
from src.location import series_location_key
from src.optimization.settings.solver_settings import OptimizationSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestFilters:
    department: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    schedule_date: Optional[date] = None
    city: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "department": self.department,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "schedule_date": self.schedule_date,
            "city": self.city,
        }


@dataclass(frozen=True)
class RequestAlgorithm:
    name: str = "OFFLINE"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestInput:
    path: Optional[str] = None


@dataclass(frozen=True)
class RequestOutput:
    path: Optional[str] = None


@dataclass(frozen=True)
class RequestPayload:
    request_id: Optional[str]
    input: RequestInput
    filters: RequestFilters
    algorithm: RequestAlgorithm
    output: RequestOutput
    raw: Dict[str, Any] = field(default_factory=dict)


def load_request(path: str | Path) -> RequestPayload:
    """
    Load a request payload from JSON and normalize it for runtime use.
    """
    request_path = Path(path)
    if not request_path.exists():
        raise FileNotFoundError(f"Request file not found: {request_path}")

    logger.debug("Loading request file", extra={"path": str(request_path)})

    with request_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Request payload must be a JSON object")

    return _parse_request_payload(data)


def build_settings_from_request(payload: RequestPayload) -> OptimizationSettings:
    """
    Convert request algorithm config to OptimizationSettings.
    """
    algo_name = payload.algorithm.name.strip().upper()
    algo_params = payload.algorithm.params or {}
    return OptimizationSettings(
        algorithm=algo_name,
        overrides={algo_name: algo_params},
    )


def apply_request_filters(
    df: pd.DataFrame,
    filters: RequestFilters,
) -> pd.DataFrame:
    """
    Apply request filters to a DataFrame (defensive, optional).
    """
    if df.empty:
        return df

    filtered = df
    mask = pd.Series(True, index=filtered.index)

    if filters.city:
        filter_city = str(filters.city).strip()
        location_key = series_location_key(filtered).astype("string").str.strip()
        location_mask = location_key == filter_city
        mask &= location_mask.fillna(False)

    if (filters.start_date or filters.end_date) and "schedule_date" in filtered.columns:
        schedule_dt = utc_to_colombia_series(filtered["schedule_date"], errors="coerce")
        if filters.start_date:
            mask &= schedule_dt >= filters.start_date
        if filters.end_date:
            mask &= schedule_dt <= filters.end_date
    if filters.schedule_date and "schedule_date" in filtered.columns:
        schedule_dt = utc_to_colombia_series(filtered["schedule_date"], errors="coerce")
        mask &= schedule_dt.dt.date == filters.schedule_date

    if filters.department is not None:
        department_filter = str(filters.department).strip()
        if "department_code" in filtered.columns:
            department_values = (
                filtered["department_code"].astype("string").str.strip()
            )
            mask &= department_values.eq(department_filter).fillna(False)
        elif "department_name" in filtered.columns:
            # Backward compatibility for datasets that still carry names.
            department_values = (
                filtered["department_name"].astype("string").str.strip()
            )
            mask &= department_values.eq(department_filter).fillna(False)

    filtered_df = filtered.loc[mask].copy()
    logger.info(
        "request_filters_applied rows_in=%s rows_out=%s",
        len(df),
        len(filtered_df),
    )
    return filtered_df


def _parse_request_payload(data: Dict[str, Any]) -> RequestPayload:
    request_id = data.get("request_id")

    input_data = data.get("input") or {}
    input_spec = RequestInput(
        path=input_data.get("path"),
    )

    filters_data = data.get("filters") or {}
    filters = RequestFilters(
        department=_parse_int(filters_data.get("department"), field="filters.department"),
        start_date=_parse_datetime(filters_data.get("start_date"), field="filters.start_date"),
        end_date=_parse_datetime(filters_data.get("end_date"), field="filters.end_date"),
        schedule_date=_parse_date(filters_data.get("schedule_date"), field="filters.schedule_date"),
        city=_parse_str(filters_data.get("city")),
    )

    algo_data = data.get("algorithm") or {}
    algorithm = RequestAlgorithm(
        name=str(algo_data.get("name", "OFFLINE")).strip().upper(),
        params=algo_data.get("params") or {},
    )

    output_data = data.get("output") or {}
    output_spec = RequestOutput(
        path=output_data.get("path"),
    )

    return RequestPayload(
        request_id=request_id,
        input=input_spec,
        filters=filters,
        algorithm=algorithm,
        output=output_spec,
        raw=data,
    )


def _parse_int(value: Any, *, field: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {field}: {value!r}") from exc


def _parse_datetime(value: Any, *, field: str) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        ts = utc_to_colombia_timestamp(value, errors="coerce")
        return value if pd.isna(ts) else ts.to_pydatetime()
    if not isinstance(value, str):
        raise ValueError(f"Invalid datetime for {field}: {value!r}")
    try:
        ts = utc_to_colombia_timestamp(value, errors="raise")
        return ts.to_pydatetime()
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime for {field}: {value!r}") from exc


def _parse_date(value: Any, *, field: str) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        ts = utc_to_colombia_timestamp(value, errors="coerce")
        return value.date() if pd.isna(ts) else ts.date()
    if not isinstance(value, str):
        raise ValueError(f"Invalid date for {field}: {value!r}")
    try:
        if "T" in value:
            ts = utc_to_colombia_timestamp(value, errors="raise")
            return ts.date()
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date for {field}: {value!r}") from exc


def _parse_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return str(value)
