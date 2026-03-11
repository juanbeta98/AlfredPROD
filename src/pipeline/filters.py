from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.utils.datetime_utils import utc_to_colombia_series

logger = logging.getLogger(__name__)


def _filter_df_by_department_code(
    df: pd.DataFrame,
    *,
    department: int | str | None,
    dataset_name: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if department is None:
        return df

    department_code = str(department).strip()
    if not department_code:
        return df

    if "department_code" not in df.columns:
        logger.info(
            "department_filter_skipped_missing_column dataset=%s expected_column=department_code",
            dataset_name,
        )
        return df

    before_rows = len(df)
    department_series = df["department_code"].astype("string").str.strip()
    filtered_df = df.loc[department_series.eq(department_code).fillna(False)].copy()

    logger.info(
        "department_filter_applied dataset=%s department=%s rows_in=%s rows_out=%s",
        dataset_name,
        department_code,
        before_rows,
        len(filtered_df),
    )
    return filtered_df


def _filter_labors_to_planning_window(
    df: pd.DataFrame,
    request_filters: Any,
) -> pd.DataFrame:
    """Drop labor rows outside the request date window and capture prior stop context.

    When a multi-labor service has some labors on a past date (already executed) and
    at least one on the planning day, the past labors are removed. For the first
    retained labor of each such service, ``prior_stop_point`` is set to the
    shop/end location of the last removed labor so that ``_ensure_map_points``
    can correctly derive where the alfred driver should pick up the vehicle.
    """
    if df is None or df.empty or request_filters is None:
        return df

    start_date = getattr(request_filters, "start_date", None)
    end_date = getattr(request_filters, "end_date", None)
    if start_date is None and end_date is None:
        return df

    if "schedule_date" not in df.columns:
        return df

    schedule_col = utc_to_colombia_series(df["schedule_date"], errors="coerce")
    in_window = pd.Series(True, index=df.index)
    if start_date is not None:
        in_window &= schedule_col >= start_date
    if end_date is not None:
        in_window &= schedule_col <= end_date

    if in_window.all():
        return df

    df = df.copy()
    df["prior_stop_point"] = pd.NA

    if "service_id" in df.columns and "labor_sequence" in df.columns:
        for _, group in df.groupby("service_id", sort=False):
            group_sorted = group.sort_values("labor_sequence", kind="stable")
            svc_in_window = in_window.loc[group_sorted.index]

            if svc_in_window.all() or (~svc_in_window).all():
                continue

            dropped = group_sorted.loc[~svc_in_window]
            kept = group_sorted.loc[svc_in_window]
            if kept.empty:
                continue

            last_dropped = dropped.iloc[-1]
            prior_stop = None
            for col in ("shop_address_point", "end_address_point", "start_address_point"):
                if col in last_dropped.index:
                    val = last_dropped.get(col)
                    if pd.notna(val) and str(val).strip():
                        prior_stop = str(val).strip()
                        break

            if prior_stop:
                df.at[kept.index[0], "prior_stop_point"] = prior_stop

    result = df.loc[in_window].copy()
    logger.info(
        "labor_planning_window_filter_applied rows_in=%s rows_out=%s rows_dropped=%s",
        len(df),
        len(result),
        len(df) - len(result),
    )
    return result


def filter_canceled_services(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all labor rows belonging to services with state == 'CANCELED'."""
    if df is None or df.empty or "state" not in df.columns:
        return df

    before_rows = len(df)
    mask = df["state"].astype("string").str.upper().ne("CANCELED")
    result = df.loc[mask].copy()

    dropped = before_rows - len(result)
    if dropped:
        logger.info(
            "canceled_services_filtered rows_in=%s rows_out=%s rows_dropped=%s",
            before_rows,
            len(result),
            dropped,
        )
    return result
