from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPLICIT_ID_COLUMNS = frozenset(
    {
        "service_id",
        "labor_id",
        "labor_type",
        "start_address_id",
        "end_address_id",
        "shop_address_id",
        "shop_id",
        "city_code",
        "department_code",
        "assigned_driver",
    }
)


def is_id_like_column(column: str) -> bool:
    col = str(column).strip().lower()
    return col in EXPLICIT_ID_COLUMNS or col.endswith("_id") or col.endswith("_code")


def normalize_id_value(value: Any) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, str):
        txt = value.strip()
        return txt or None

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        float_val = float(value)
        if float_val.is_integer():
            return str(int(float_val))
        return format(float_val, ".15g")

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    txt = str(value).strip()
    return txt or None


def normalize_id_columns(
    df: pd.DataFrame,
    *,
    columns: Iterable[str] | None = None,
    include_detected: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    target_columns = {str(col) for col in columns or ()}
    if include_detected:
        target_columns.update(col for col in df.columns if is_id_like_column(col))

    for col in target_columns:
        if col not in df.columns:
            continue
        normalized = df[col].map(normalize_id_value)
        df[col] = pd.Series(normalized, index=df.index, dtype="string")

    return df
