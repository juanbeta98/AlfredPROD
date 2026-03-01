# distance_utils.py
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

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


def parse_point(s: str) -> Tuple[float, float]:
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
    dist_dict: Optional[Dict[Any, Any]] = None,
    timeout: int = 5,
    **kwargs: Any,
) -> Tuple[float, Optional[Dict[Any, Any]]]:
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


def batch_distance_matrix(
    origins: List[str],
    destinations: List[str],
    osrm_route_url: str,
    timeout: int = 60,
) -> Dict[Tuple[str, str], float]:
    """
    Batch-compute all origin→destination road distances via the OSRM Table API.

    Makes a single HTTP request to /table/v1/ instead of one request per pair,
    eliminating the per-iteration I/O bottleneck when distance_method='osrm'.

    Returns a dict compatible with dist_dict: {(p1_str, p2_str): km}.
    Returns an empty dict on any error so callers fall back to per-call behaviour.

    Args:
        origins:        POINT strings used as row sources (e.g. driver positions, labor starts).
        destinations:   POINT strings used as column destinations (e.g. labor starts, labor ends).
        osrm_route_url: Value of OSRM_URL env var, e.g. "http://localhost:5050/route/v1/driving/"
        timeout:        HTTP timeout in seconds (default 60; matrix requests are larger than single-pair calls).
    """
    if not origins or not destinations:
        return {}

    table_url = osrm_route_url.replace("/route/v1/driving/", "/table/v1/driving/")

    # Build a single deduplicated coordinate list; origins and destinations index into it.
    unique: List[str] = list(dict.fromkeys(origins + destinations))
    src_idx = [unique.index(o) for o in origins]
    dst_idx = [unique.index(d) for d in destinations]

    coord_parts: List[str] = []
    for pt in unique:
        lon, lat = parse_point(pt)
        if math.isnan(lon) or math.isnan(lat):
            logger.warning("batch_distance_matrix invalid point=%s — aborting batch", pt)
            return {}
        coord_parts.append(f"{lon},{lat}")

    # NOTE: OSRM Table API uses semicolons to separate index lists,
    # not commas (commas separate lon,lat within a coordinate).
    url = (
        table_url
        + ";".join(coord_parts)
        + "?sources=" + ";".join(map(str, src_idx))
        + "&destinations=" + ";".join(map(str, dst_idx))
        + "&annotations=distance"
    )

    try:
        r = _osrm_session.get(url, timeout=timeout)
        r.raise_for_status()
        matrix = r.json()["distances"]   # list[list[float | None]], values in metres
    except Exception as e:
        logger.warning("osrm_table_failed reason=%s — falling back to per-call mode", e)
        return {}

    result: Dict[Tuple[str, str], float] = {}
    for i, orig in enumerate(origins):
        for j, dest in enumerate(destinations):
            val = matrix[i][j]
            if val is not None:
                result[(orig, dest)] = val / 1000.0   # metres → km

    return result
