from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd


class OutputFormatter:
    """
    Formatter for optimization results into API-ready JSON payloads.
    """

    KPI_DECIMALS = 2
    DIAGNOSTIC_ADD_DATA_FIELDS = (
        "is_infeasible",
        "is_warning",
        "infeasibility_cause_code",
        "infeasibility_cause_detail",
        "reassignment_candidate",
        "reassignment_priority",
        "original_assigned_driver",
        "preassignment_infeasible_detected",
        "preassignment_infeasibility_cause_code",
        "preassignment_infeasibility_cause_detail",
        "preassignment_warning_detected",
        "preassignment_warning_code",
        "preassignment_warning_detail",
        "warning_code",
        "warning_detail",
        "payload_schedule_reference",
        "computed_arrival",
        "previous_labor_end",
        "feasible_window_end",
        "driver_available_at",
        "computed_travel_min",
        "minutes_late_payload",
        "minutes_after_window",
    )

    @staticmethod
    def _round_kpi(value: float) -> float:
        return round(float(value), OutputFormatter.KPI_DECIMALS)

    @staticmethod
    def _add_add_data_value(add_data: Dict[str, Any], key: str, value: float | None) -> None:
        if value is None:
            return
        add_data[key] = OutputFormatter._round_kpi(value)

    @staticmethod
    def format(
        results: Any,
        metadata: Dict[str, Any] | None = None,
        request_id: str | None = None,
        status: str = "completed",
    ) -> Dict[str, Any] | Any:
        """
        Format optimization output into the payload expected by the API.

        Args:
            results: Optimization output (DataFrame, list, dict)
            metadata: Unused for now (kept for compatibility)
            request_id: Unused for now (kept for compatibility)
            status: Unused for now (kept for compatibility)

        Returns:
            JSON-serializable payload matching API contract directly.
        """
        _ = metadata
        _ = request_id
        _ = status

        # Preserve already-structured non-completed payloads, e.g. failure reports.
        if isinstance(results, dict) and "status" in results and "data" not in results:
            return results

        if isinstance(results, pd.DataFrame):
            return {"data": OutputFormatter._format_results_df(results)}

        if results is None:
            return {"data": []}

        if isinstance(results, dict) and "data" in results:
            payload = dict(results)
            if isinstance(payload.get("data"), dict):
                payload["data"] = [payload["data"]]
            elif payload.get("data") is None:
                payload["data"] = []
            return payload

        if isinstance(results, list):
            return {"data": results}

        if isinstance(results, dict) and OutputFormatter._looks_like_service_payload(results):
            return {"data": [results]}

        return results

    @staticmethod
    def _format_results_df(df: pd.DataFrame) -> list[Dict[str, Any]]:
        if df.empty:
            return []

        output: list[Dict[str, Any]] = []

        for service_id, group in df.groupby("service_id", sort=False):
            first = group.iloc[0]
            service_schedule_date = OutputFormatter._service_schedule_date(first)

            service: Dict[str, Any] = {
                "service_id": OutputFormatter._clean_int_id(service_id),
                "schedule_date": OutputFormatter._format_dt_utc(service_schedule_date),
                "serviceLabors": [],
            }

            for _, row in group.iterrows():
                labor_id = OutputFormatter._clean_int_id(row.get("labor_id"))
                if labor_id is None:
                    continue

                labor_schedule_date = OutputFormatter._labor_schedule_date(row)
                if labor_schedule_date is None:
                    continue

                labor: Dict[str, Any] = {
                    "id": labor_id,
                    "schedule_date": labor_schedule_date,
                }

                add_data = OutputFormatter._build_add_data(row)
                if add_data is not None:
                    labor["addData"] = add_data

                driver_id = OutputFormatter._alfred_id_for_labor(row)
                if driver_id is not None:
                    labor["alfred"] = {"id": driver_id}

                service["serviceLabors"].append(labor)

            if service["serviceLabors"]:
                output.append(service)

        return output

    @staticmethod
    def _service_schedule_date(row: pd.Series) -> Any:
        # Preserve service-level schedule_date from input payload when available.
        if "service_schedule_date" in row.index and not OutputFormatter._is_missing(row.get("service_schedule_date")):
            return row.get("service_schedule_date")
        if "schedule_date" in row.index:
            return row.get("schedule_date")
        return None

    @staticmethod
    def _labor_schedule_date(row: pd.Series) -> str | None:
        for col in (
            "actual_start",
            "payload_labor_schedule_date",
            "schedule_date",
            "service_schedule_date",
        ):
            if col in row.index and not OutputFormatter._is_missing(row.get(col)):
                value = OutputFormatter._format_dt_utc(row.get(col))
                if value is not None:
                    return value
        return None

    @staticmethod
    def _alfred_id_for_labor(row: pd.Series) -> int | None:
        if OutputFormatter._labor_has_shop(row):
            return None
        return OutputFormatter._clean_int_id(row.get("assigned_driver"))

    @staticmethod
    def _labor_has_shop(row: pd.Series) -> bool:
        if "shop_id" not in row.index:
            return False
        return not OutputFormatter._is_missing(row.get("shop_id"))

    @staticmethod
    def _build_add_data(row: pd.Series) -> Dict[str, Any] | None:
        labor_distance_value = OutputFormatter._first_numeric(
            row,
            ("labor_distance_km", "dist_km", "distance_km", "labor_distance"),
        )
        driver_move_distance_value = OutputFormatter._first_numeric(
            row,
            ("driver_move_distance_km",),
        )
        duration_value = OutputFormatter._first_numeric(
            row,
            ("duration_min", "labor_duration"),
        )
        if duration_value is None:
            duration_value = OutputFormatter._duration_from_actual_times(
                row.get("actual_start"),
                row.get("actual_end"),
            )

        add_data: Dict[str, Any] = {}
        OutputFormatter._add_add_data_value(add_data, "distance_km", labor_distance_value)
        OutputFormatter._add_add_data_value(add_data, "labor_distance_km", labor_distance_value)
        OutputFormatter._add_add_data_value(add_data, "driver_move_distance_km", driver_move_distance_value)
        OutputFormatter._add_add_data_value(add_data, "duration_min", duration_value)
        OutputFormatter._add_diagnostics_add_data(add_data, row)

        return add_data or None

    @staticmethod
    def _add_diagnostics_add_data(add_data: Dict[str, Any], row: pd.Series) -> None:
        for key in OutputFormatter.DIAGNOSTIC_ADD_DATA_FIELDS:
            if key not in row.index:
                continue
            cleaned = OutputFormatter._clean_add_data_scalar(row.get(key))
            if cleaned is None:
                continue
            add_data[key] = cleaned

    @staticmethod
    def _clean_add_data_scalar(value: Any) -> Any | None:
        if OutputFormatter._is_missing(value):
            return None

        if isinstance(value, (np.bool_, bool)):
            return bool(value)

        if isinstance(value, pd.Timestamp):
            return OutputFormatter._format_dt_utc(value)

        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)

        numeric = OutputFormatter._clean_numeric(value)
        if numeric is not None:
            return OutputFormatter._round_kpi(numeric)

        return str(value)

    @staticmethod
    def _first_numeric(row: pd.Series, columns: tuple[str, ...]) -> float | None:
        for col in columns:
            if col not in row.index:
                continue
            cleaned = OutputFormatter._clean_numeric(row.get(col))
            if cleaned is not None:
                return cleaned
        return None

    @staticmethod
    def _duration_from_actual_times(actual_start: Any, actual_end: Any) -> float | None:
        start_ts = pd.to_datetime(actual_start, errors="coerce", utc=True)
        end_ts = pd.to_datetime(actual_end, errors="coerce", utc=True)
        if pd.isna(start_ts) or pd.isna(end_ts):
            return None

        duration_min = (end_ts - start_ts).total_seconds() / 60.0
        if duration_min < 0:
            return None

        return float(duration_min)

    @staticmethod
    def _format_dt_utc(value: Any) -> str | None:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        return None

    @staticmethod
    def _clean_int_id(value: Any) -> int | None:
        if OutputFormatter._is_missing(value):
            return None

        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)

        if isinstance(value, (np.floating, float)):
            if pd.isna(value):
                return None
            return int(float(value))

        txt = str(value).strip()
        if not txt or txt.lower() in {"none", "null", "nan"}:
            return None

        try:
            return int(txt)
        except ValueError:
            pass

        try:
            return int(float(txt))
        except ValueError:
            return None

    @staticmethod
    def _clean_numeric(value: Any) -> float | None:
        if OutputFormatter._is_missing(value):
            return None

        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return float(value)

        if isinstance(value, (np.floating, float)):
            num = float(value)
            if math.isnan(num):
                return None
            return num

        txt = str(value).strip()
        if not txt or txt.lower() in {"none", "null", "nan"}:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if value is pd.NA or value is pd.NaT:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    @staticmethod
    def _looks_like_service_payload(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        labors = value.get("serviceLabors")
        return "service_id" in value and isinstance(labors, list)
