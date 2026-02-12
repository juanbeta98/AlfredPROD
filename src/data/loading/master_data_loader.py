from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.optimization.settings.master_data import MasterDataParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MasterData:
    directorio_df: pd.DataFrame
    duraciones_df: pd.DataFrame
    dist_dict: Dict[str, Dict[Tuple[str, str], Any]]


def load_master_data(params: MasterDataParams) -> MasterData:
    params.validate()
    return _load_master_data_cached(
        params.base_dir,
        params.directorio_stem,
        params.duraciones_stem,
        params.dist_dict_stem,
        params.prefer_parquet,
    )


@lru_cache(maxsize=4)
def _load_master_data_cached(
    base_dir: str,
    directorio_stem: str,
    duraciones_stem: str,
    dist_dict_stem: str,
    prefer_parquet: bool,
) -> MasterData:
    base_path = Path(base_dir)
    directorio_df, directorio_path = _load_any(
        base_path=base_path,
        stem=directorio_stem,
        prefer_parquet=prefer_parquet,
    )
    duraciones_df, duraciones_path = _load_any(
        base_path=base_path,
        stem=duraciones_stem,
        prefer_parquet=prefer_parquet,
    )
    dist_dict, dist_path = _load_dist_dict(
        base_path=base_path,
        stem=dist_dict_stem,
        prefer_parquet=prefer_parquet,
    )
    city_to_department, city_label_to_department = _load_city_department_crosswalk(base_path)
    if city_to_department:
        duraciones_df = _remap_duraciones_to_department_code(
            duraciones_df,
            city_to_department=city_to_department,
        )
        dist_dict = _remap_dist_dict_to_department_code(
            dist_dict,
            city_to_department=city_to_department,
        )
        directorio_df = _enrich_directorio_department_code(
            directorio_df,
            city_to_department=city_to_department,
            city_label_to_department=city_label_to_department,
        )

    logger.info(
        "master_data_loaded directorio duraciones dist_dict",
        # directorio_path.name,
        # duraciones_path.name,
        # dist_path.name,
    )

    return MasterData(
        directorio_df=directorio_df,
        duraciones_df=duraciones_df,
        dist_dict=dist_dict,
    )


def _load_any(
    *,
    base_path: Path,
    stem: str,
    prefer_parquet: bool,
) -> tuple[pd.DataFrame, Path]:
    parquet_path = base_path / f"{stem}.parquet"
    csv_path = base_path / f"{stem}.csv"

    chosen_path: Optional[Path] = None
    if prefer_parquet and parquet_path.exists():
        chosen_path = parquet_path
    elif csv_path.exists():
        chosen_path = csv_path
    elif parquet_path.exists():
        chosen_path = parquet_path

    if chosen_path is None:
        raise FileNotFoundError(
            f"Master data file not found for {stem!r} in {base_path}"
        )

    if chosen_path.suffix == ".parquet":
        return pd.read_parquet(chosen_path), chosen_path

    return pd.read_csv(chosen_path), chosen_path


def _load_dist_dict(
    *,
    base_path: Path,
    stem: str,
    prefer_parquet: bool,
) -> tuple[Dict[str, Dict[Tuple[str, str], Any]], Path]:
    parquet_path = base_path / f"{stem}.parquet"
    pickle_path = base_path / f"{stem}.pkl"

    chosen_path: Optional[Path] = None
    if prefer_parquet and parquet_path.exists():
        chosen_path = parquet_path
    elif pickle_path.exists():
        chosen_path = pickle_path
    elif parquet_path.exists():
        chosen_path = parquet_path

    if chosen_path is None:
        raise FileNotFoundError(
            f"Master data dist_dict not found for {stem!r} in {base_path}"
        )

    if chosen_path.suffix == ".parquet":
        df = pd.read_parquet(chosen_path)
        if {"city", "p1", "p2", "distance_km"}.issubset(df.columns):
            dist_dict: Dict[str, Dict[Tuple[str, str], Any]] = {}
            for city, p1, p2, dist in df[["city", "p1", "p2", "distance_km"]].itertuples(index=False):
                city_key = str(city)
                dist_dict.setdefault(city_key, {})[(str(p1), str(p2))] = dist
            return dist_dict, chosen_path
        raise ValueError(
            f"dist_dict parquet must contain columns ['city','p1','p2','distance_km']: {chosen_path}"
        )

    dist_obj = pd.read_pickle(chosen_path)
    if isinstance(dist_obj, dict):
        return dist_obj, chosen_path
    if isinstance(dist_obj, pd.DataFrame):
        if {"city", "p1", "p2", "distance_km"}.issubset(dist_obj.columns):
            dist_dict: Dict[str, Dict[Tuple[str, str], Any]] = {}
            for city, p1, p2, dist in dist_obj[["city", "p1", "p2", "distance_km"]].itertuples(index=False):
                city_key = str(city)
                dist_dict.setdefault(city_key, {})[(str(p1), str(p2))] = dist
            return dist_dict, chosen_path
    raise ValueError(f"Unsupported dist_dict format in {chosen_path}")


def _load_city_department_crosswalk(base_path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    cities_path = base_path / "cities.csv"
    if not cities_path.exists():
        return {}, {}

    df = pd.read_csv(cities_path, dtype=str, keep_default_na=False)
    if df.empty or "city" not in df.columns or "data" not in df.columns:
        return {}, {}

    city_to_department: Dict[str, str] = {}
    city_label_to_department: Dict[str, str] = {}

    for _, row in df.iterrows():
        city_key = str(row.get("city", "")).strip()
        data_raw = row.get("data", "")
        if not city_key or not data_raw:
            continue

        try:
            payload = json.loads(data_raw)
        except json.JSONDecodeError:
            continue

        department_code = payload.get("department_code")
        if department_code is None:
            continue
        department_code_txt = str(department_code).strip()
        if not department_code_txt:
            continue

        city_to_department[city_key] = department_code_txt

        city_name = _clean_text(payload.get("city_name"))
        display_name = _extract_city_from_display_name(row.get("name"))
        for alias in _city_aliases(city_name, display_name):
            city_label_to_department.setdefault(alias, department_code_txt)

    return city_to_department, city_label_to_department


def _remap_duraciones_to_department_code(
    df: pd.DataFrame,
    *,
    city_to_department: Dict[str, str],
) -> pd.DataFrame:
    if df is None or df.empty or "city" not in df.columns:
        return df

    result = df.copy()
    city_series = result["city"].astype("string").str.strip()
    mapped = city_series.map(city_to_department)
    result["city"] = mapped.where(mapped.notna() & mapped.ne(""), city_series)
    return result


def _remap_dist_dict_to_department_code(
    dist_dict: Dict[str, Dict[Tuple[str, str], Any]],
    *,
    city_to_department: Dict[str, str],
) -> Dict[str, Dict[Tuple[str, str], Any]]:
    if not isinstance(dist_dict, dict) or not city_to_department:
        return dist_dict

    remapped: Dict[str, Dict[Tuple[str, str], Any]] = {}
    for city_key, city_dist in dist_dict.items():
        source_key = str(city_key).strip()
        target_key = city_to_department.get(source_key, source_key)
        bucket = remapped.setdefault(target_key, {})
        if isinstance(city_dist, dict):
            bucket.update(city_dist)

    return remapped


def _enrich_directorio_department_code(
    df: pd.DataFrame,
    *,
    city_to_department: Dict[str, str],
    city_label_to_department: Dict[str, str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    result = df.copy()
    department_series = (
        result["department_code"].astype("string").str.strip()
        if "department_code" in result.columns
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    department_series = department_series.where(
        department_series.notna() & department_series.ne(""),
        pd.NA,
    )

    if "city_id" in result.columns:
        city_id_series = result["city_id"].astype("string").str.strip()
        mapped_by_city_id = city_id_series.map(city_to_department)
        department_series = department_series.fillna(mapped_by_city_id)

    if "city" in result.columns:
        city_series = result["city"].astype("string").str.strip()
        mapped_numeric_city = city_series.map(city_to_department)
        mapped_by_label = city_series.map(
            lambda value: city_label_to_department.get(_normalize_city_label(value))
            if pd.notna(value)
            else None
        )
        department_series = department_series.fillna(mapped_numeric_city).fillna(mapped_by_label)

    result["department_code"] = department_series.astype("string")
    return result


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_city_from_display_name(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    left, sep, right = text.partition("-")
    if sep:
        return _clean_text(right) or _clean_text(text)
    return text


def _normalize_city_label(value: Any) -> str:
    text = _clean_text(value)
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.upper().split())
    return normalized


def _city_aliases(*values: Any) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        label = _normalize_city_label(value)
        if not label:
            continue
        aliases.add(label)

        left, sep, _ = label.partition(" Y ")
        if sep and left:
            aliases.add(left.strip())

    return aliases
