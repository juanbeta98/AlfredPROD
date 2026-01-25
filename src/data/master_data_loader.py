from __future__ import annotations

import logging
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
