from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.geo.location_resolver import get_location_resolver


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def split_department_city(value: Any) -> tuple[str | None, str | None]:
    text = _clean_text(value)
    if not text:
        return None, None

    left, sep, right = text.partition("-")
    if not sep:
        return None, text

    department_name = _clean_text(left)
    city_name = _clean_text(right)
    return department_name, city_name


def parse_service_location(raw_city: Any) -> dict[str, str | None]:
    resolver = get_location_resolver()

    if isinstance(raw_city, dict):
        city_code = _clean_text(raw_city.get("id"))
        city_payload_name = _clean_text(raw_city.get("name"))
        city_data = raw_city.get("data") or {}
        city_name = None
        department_code = None
        department_name = None
        if isinstance(city_data, Mapping):
            department_code = _clean_text(city_data.get("department_code"))
            department_name = _clean_text(city_data.get("department_name"))
            city_name = _clean_text(city_data.get("city_name"))

        return resolver.resolve(
            city_code=city_code,
            city_name=city_name,
            department_code=department_code,
            department_name=department_name,
            department_city_text=city_payload_name,
        )

    text = _clean_text(raw_city)
    if text is None:
        return resolver.resolve()
    if text.isdigit():
        return resolver.resolve(city_code=text)
    return resolver.resolve(department_city_text=text)


def row_location_key(row: Mapping[str, Any]) -> str | None:
    # Runtime matching/grouping should use department code whenever possible.
    return _clean_text(row.get("department_code")) or _clean_text(row.get("department_name"))


def series_location_key(df: pd.DataFrame) -> pd.Series:
    index = df.index
    department_code = (
        df["department_code"].astype("string").str.strip()
        if "department_code" in df.columns
        else pd.Series(pd.NA, index=index, dtype="string")
    )
    department_name = (
        df["department_name"].astype("string").str.strip()
        if "department_name" in df.columns
        else pd.Series(pd.NA, index=index, dtype="string")
    )
    location_key = department_code.where(
        department_code.notna() & department_code.ne(""),
        department_name,
    )
    return location_key.where(
        location_key.notna() & location_key.ne(""),
        pd.NA,
    )
