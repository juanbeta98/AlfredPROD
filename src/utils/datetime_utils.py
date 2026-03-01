from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

COLOMBIA_TIMEZONE_NAME = "America/Bogota"
COLOMBIA_TIMEZONE = ZoneInfo(COLOMBIA_TIMEZONE_NAME)


def now_colombia() -> datetime:
    return datetime.now(COLOMBIA_TIMEZONE)


def utc_to_colombia_series(values: pd.Series, *, errors: str = "coerce") -> pd.Series:
    """
    Parse values as UTC datetimes and convert to Colombia timezone.
    """
    parsed = pd.to_datetime(values, errors=errors, utc=True)
    return parsed.dt.tz_convert(COLOMBIA_TIMEZONE_NAME)


def utc_to_colombia_timestamp(value: Any, *, errors: str = "coerce") -> pd.Timestamp | type(pd.NaT):
    """
    Parse a single value as UTC datetime and convert to Colombia timezone.
    """
    parsed = pd.to_datetime(value, errors=errors, utc=True)
    if parsed is pd.NaT:
        return pd.NaT
    if not isinstance(parsed, pd.Timestamp):
        raise TypeError(f"Expected Timestamp from pd.to_datetime, got {type(parsed)!r}")
    return parsed.tz_convert(COLOMBIA_TIMEZONE_NAME)


def normalize_datetime_columns_to_colombia(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    In-place normalization helper for known datetime columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = utc_to_colombia_series(df[col], errors=errors)
    return df
