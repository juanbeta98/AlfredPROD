# distance_utils.py
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from datetime import datetime

import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import traceback

import pickle
import os
import csv


def _build_osrm_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(connect=3, read=3, backoff_factor=0.3, raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

_osrm_session = _build_osrm_session()


def parse_point(s: str) -> tuple[float, float]:
    """
    Extrae latitud y longitud de un texto con formato 'POINT(lon lat)'.

    Parámetros:
        s (str): Texto de coordenadas.

    Retorna:
        tuple: (latitud, longitud) o (None, None) si no es válido.
    """
    if pd.isna(s) or not isinstance(s, str) or not s.strip().startswith("POINT"):
        return float('nan'), float('nan')
    lon, lat = map(float, s.lstrip('POINT').strip(' ()').split())
    return lon, lat


def distance(
    p1: str,
    p2: str,
    method: str,
    dist_dict: dict[Any, Any] | None = None,
    timeout: int = 5,
    **kwargs: Any,
) -> tuple[float, dict[Any, Any] | None]:
    """
    Calcula la distancia entre dos puntos geográficos según el método indicado.

    Parámetros:
        p1, p2 (str): Puntos en formato 'POINT(lon lat)'.
        method (str): 'precalced', 'haversine', 'osrm', 'manhattan'.
        dist_dict (dict): Diccionario precalculado, requerido si method='precalced'.
        timeout (int): Tiempo máximo en segundos para consultas OSRM.

    Retorna:
        float: Distancia en kilómetros, NaN si no es calculable.
    """
    if method == 'precalced':
        if dist_dict is None:
            return float('nan'), dist_dict
        if (p1, p2) in dist_dict:
            return dist_dict.get((p1, p2), float('nan')), dist_dict
        new_distance, _ = distance(p1, p2, 'osrm', timeout=timeout)
        return new_distance, dist_dict

    lon1, lat1 = parse_point(p1)
    lon2, lat2 = parse_point(p2)
    if None in (lat1, lon1, lat2, lon2):
        return float('nan'), dist_dict

    if method == 'haversine':
        φ1, φ2 = map(math.radians, (lat1, lat2))
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return 2 * 6371 * math.atan2(math.sqrt(a), math.sqrt(1-a)), dist_dict

    if method == 'osrm':
        if dist_dict and (p1, p2) in dist_dict:
            return dist_dict[(p1, p2)], dist_dict
        osrm_url = kwargs.get("osrm_url") or os.environ.get("OSRM_URL")
        if not osrm_url:
            raise ValueError("OSRM_URL is not configured for OSRM distance requests")
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        try:
            r = _osrm_session.get(osrm_url + coords + "?overview=false", timeout=timeout)
            r.raise_for_status()
            result_km = r.json()['routes'][0]['distance'] / 1000
            updated = {**(dist_dict or {}), (p1, p2): result_km}
            return result_km, updated
        except requests.exceptions.RequestException as e:
            logger.warning(
                "osrm_fallback_to_haversine coords=%s reason=request_failed error=%s",
                coords, e,
            )
            return distance(p1, p2, method='haversine')
        except Exception as e:
            logger.warning(
                "osrm_fallback_to_haversine coords=%s reason=unexpected_error error=%s traceback=%s",
                coords, e, traceback.format_exc().splitlines()[-1],
            )
            return distance(p1, p2, method='haversine')


    # Manhattan
    KM_PER_DEG_LAT = 111.32

    mean_lat = math.radians((lat1 + lat2) / 2)
    dlat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    dlon = abs(lon1 - lon2) * KM_PER_DEG_LAT * math.cos(mean_lat)

    return dlat + dlon, dist_dict


# OSRM default --max-table-size: maximum unique coordinates per Table API request.
_OSRM_MAX_TABLE_COORDS = 100


def _osrm_table_chunk(
    src_pts: list[str],
    dst_pts: list[str],
    table_url: str,
    timeout: int,
    include_times: bool = False,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Single OSRM Table API call for one chunk of sources × destinations.
    Caller is responsible for ensuring len(unique(src_pts + dst_pts)) <= _OSRM_MAX_TABLE_COORDS.

    Returns:
        (dist_result, time_result) where distances are in km and times in minutes.
        time_result is empty when include_times=False.
    """
    unique: list[str] = list(dict.fromkeys(src_pts + dst_pts))
    if len(unique) < 2:
        return {}, {}
    src_idx = [unique.index(o) for o in src_pts]
    dst_idx = [unique.index(d) for d in dst_pts]

    coord_parts: list[str] = []
    for pt in unique:
        lon, lat = parse_point(pt)
        if math.isnan(lon) or math.isnan(lat):
            raise ValueError(f"invalid point: {pt}")
        coord_parts.append(f"{lon},{lat}")

    # NOTE: OSRM Table API uses semicolons to separate index lists,
    # not commas (commas separate lon,lat within a coordinate).
    annotations = "distance,duration" if include_times else "distance"
    url = (
        table_url
        + ";".join(coord_parts)
        + "?sources=" + ";".join(map(str, src_idx))
        + "&destinations=" + ";".join(map(str, dst_idx))
        + "&annotations=" + annotations
    )
    r = _osrm_session.get(url, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    dist_matrix = payload["distances"]   # list[list[float | None]], metres
    time_matrix = payload.get("durations") if include_times else None  # seconds

    dist_result: dict[tuple[str, str], float] = {}
    time_result: dict[tuple[str, str], float] = {}
    for i, orig in enumerate(src_pts):
        for j, dest in enumerate(dst_pts):
            dval = dist_matrix[i][j]
            if dval is not None:
                dist_result[(orig, dest)] = dval / 1000.0
            if time_matrix is not None:
                tval = time_matrix[i][j]
                if tval is not None:
                    time_result[(orig, dest)] = tval / 60.0
    return dist_result, time_result


def batch_distance_matrix(
    origins: list[str],
    destinations: list[str],
    osrm_route_url: str,
    timeout: int = 60,
    include_times: bool = False,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """
    Batch-compute all origin→destination road distances (and optionally durations) via the OSRM Table API.

    Automatically chunks requests so each stays within OSRM's max-table-size limit
    (default 100 unique coordinates per request). Falls back to empty dicts on any
    error so callers use per-call mode instead.

    Args:
        origins:        POINT strings used as row sources.
        destinations:   POINT strings used as column destinations.
        osrm_route_url: Value of OSRM_URL env var, e.g. "http://localhost:5050/route/v1/driving/"
        timeout:        HTTP timeout in seconds (default 60).
        include_times:  If True, also request duration matrix and return it (minutes).

    Returns:
        (dist_dict, time_dict) — distances in km, times in minutes.
        time_dict is empty when include_times=False.
    """
    if not origins or not destinations:
        return {}, {}

    # OSRM Table API requires ≥2 unique coordinates; a single-point request returns 400.
    _all_unique = list(dict.fromkeys(origins + destinations))
    if len(_all_unique) < 2:
        logger.debug("osrm_table_skipped unique_points=%d — need ≥2 for table API", len(_all_unique))
        return {}, {}

    table_url = osrm_route_url.replace("/route/v1/driving/", "/table/v1/driving/")
    # chunk_size = 50: worst case 50 sources + 50 destinations = 100 unique coords = OSRM limit.
    chunk_size = _OSRM_MAX_TABLE_COORDS // 2

    src_chunks = [origins[i:i + chunk_size] for i in range(0, len(origins), chunk_size)]
    dst_chunks = [destinations[j:j + chunk_size] for j in range(0, len(destinations), chunk_size)]

    dist_result: dict[tuple[str, str], float] = {}
    time_result: dict[tuple[str, str], float] = {}
    try:
        for src_chunk in src_chunks:
            for dst_chunk in dst_chunks:
                d, t = _osrm_table_chunk(src_chunk, dst_chunk, table_url, timeout, include_times)
                dist_result.update(d)
                time_result.update(t)
    except Exception as e:
        logger.warning("osrm_table_failed reason=%s — falling back to per-call mode", e)
        return {}, {}

    return dist_result, time_result


def travel_time_minutes(
    p1: str,
    p2: str,
    speed_kmh: float,
    time_method: str = "speed_based",
    dist_method: str = "osrm",
    dist_dict: dict[Any, Any] | None = None,
    time_dict: dict[Any, Any] | None = None,
    timeout: int = 5,
    **kwargs: Any,
) -> tuple[float, float, dict[Any, Any], dict[Any, Any]]:
    """
    Compute travel time (minutes) and road distance (km) between two POINT strings.

    Modes
    -----
    speed_based (default):
        Fetches road distance via distance() and divides by speed_kmh.
    osrm_times:
        Uses OSRM-reported duration directly. Checks time_dict cache first;
        on cache miss fetches both distance and duration from the /route API;
        falls back to speed_based on OSRM failure.

    Returns
    -------
    (dist_km, time_min, updated_dist_dict, updated_time_dict)
        dist_km and time_min are NaN when the calculation is not possible.
    """
    dist_dict = dist_dict or {}
    time_dict = time_dict or {}

    if time_method == "osrm_times":
        # --- cache hit ---
        if (p1, p2) in time_dict:
            t_min = time_dict[(p1, p2)]
            d_km = dist_dict.get((p1, p2), float("nan"))
            return d_km, t_min, dist_dict, time_dict

        # --- cache miss: call OSRM /route and extract both distance and duration ---
        osrm_url = kwargs.get("osrm_url") or os.environ.get("OSRM_URL")
        lon1, lat1 = parse_point(p1)
        lon2, lat2 = parse_point(p2)
        if math.isnan(lon1) or math.isnan(lat1) or math.isnan(lon2) or math.isnan(lat2):
            return float("nan"), float("nan"), dist_dict, time_dict
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        try:
            r = _osrm_session.get(
                osrm_url + coords + "?overview=false", timeout=timeout
            )
            r.raise_for_status()
            route = r.json()["routes"][0]
            d_km = route["distance"] / 1000.0
            t_min = route["duration"] / 60.0
            updated_dist = {**dist_dict, (p1, p2): d_km}
            updated_time = {**time_dict, (p1, p2): t_min}
            return d_km, t_min, updated_dist, updated_time
        except Exception as e:
            logger.warning(
                "osrm_times_fallback_to_speed_based coords=%s reason=%s",
                coords, e,
            )
            # fall through to speed_based

    # --- speed_based (default or fallback) ---
    d_km, updated_dist = distance(p1, p2, method=dist_method, dist_dict=dist_dict, timeout=timeout, **kwargs)
    t_min = 0.0 if math.isnan(d_km) else d_km / speed_kmh * 60.0
    return d_km, t_min, updated_dist, time_dict
