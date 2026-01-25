# distance_utils.py
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from datetime import datetime

import math
import requests

import traceback

import pickle
import os
import csv



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

    lat1, lon1 = parse_point(p1)
    lat2, lon2 = parse_point(p2)
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
            r = requests.get(osrm_url + coords + "?overview=false", timeout=timeout)
            r.raise_for_status()
            return r.json()['routes'][0]['distance'] / 1000, {}
        except requests.exceptions.RequestException as e:
            # Catch all HTTP-related issues (timeout, 500, bad connection, etc.)
            print(f"⚠️ OSRM request failed for coords={coords}")
            print(f"   → Exception: {e}")
            print(f"   → Traceback: {traceback.format_exc().splitlines()[-1]}")
            return distance(p1, p2, method='haversine')
        except Exception as e:
            # Catch any unexpected error (JSON decoding, missing fields, etc.)
            print(f"🚨 Unexpected error while getting OSRM distance for {coords}")
            print(f"   → Exception: {e}")
            print(f"   → Traceback: {traceback.format_exc().splitlines()[-1]}")
            return distance(p1, p2, method='haversine')


    # Manhattan
    KM_PER_DEG_LAT = 111.32
    mean_lat = math.radians((lat1 + lat2) / 2)
    dlat = abs(lat1 - lat2) * KM_PER_DEG_LAT
    dlon = abs(lon1 - lon2) * KM_PER_DEG_LAT * math.cos(mean_lat)
    
    return dlat + dlon, dist_dict
