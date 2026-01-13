import pandas as pd
import numpy as np

from datetime import timedelta
from pathlib import Path
from typing import Optional, Union, Tuple

from src.data.distance_utils import distance
from src.utils.filtering import filter_valid_services
from src.config.experimentation_config import codificacion_ciudades


def generate_labors_raw_df(
    data_path: Union[str, Path],
    tz: str = "America/Bogota",
    strict_datetime_format: Optional[str] = "%Y-%m-%d %H:%M:%S.%f %z",
) -> pd.DataFrame:
    """
    Genera el dataset unificado de labores (`labors_raw_df`) a partir de las tablas 
    crudas de servicios (`service_labors`, `services`, `address`).

    Proceso
    -------
    1. Carga las tablas de entrada (`service_labors.csv`, `services.csv`, `address.csv`) 
       desde la carpeta `data_raw/`.
    2. Realiza el parseo de columnas de fechas usando formato estricto si se especifica, 
       con retroceso a un parseo flexible en caso de error. Convierte todas las fechas 
       a la zona horaria indicada.
    3. Enriquece la tabla `service_labors` con:
        - Información de servicios desde `services`.
        - Información de direcciones (`address_id`, `point`, `name`) desde `address`.
    4. Normaliza la columna `city` a códigos de texto.
    5. Filtra las filas para conservar únicamente las ciudades válidas definidas en 
       `codificacion_ciudades`.

    Parámetros
    ----------
    data_path : str | Path
        Ruta base del proyecto que contiene la subcarpeta `data_raw/`.
    tz : str, opcional
        Zona horaria de destino para conversión de fechas (por defecto = "America/Bogota").
    strict_datetime_format : str | None, opcional
        Si se indica, fuerza un parseo estricto de las fechas con ese formato.
        En caso de fallo, se aplica un parseo flexible.

    Retorna
    -------
    labors_raw_df : pd.DataFrame
        Dataset unificado de labores de servicio, enriquecido con información de 
        servicios y direcciones, filtrado para incluir únicamente ciudades válidas.
    """

    data_path = Path(data_path)
    raw_dir = data_path / "data_raw"

    # Cargar tablas base
    address_df = pd.read_csv(raw_dir / "address.csv", low_memory=False)
    service_labors_df = pd.read_csv(raw_dir / "service_labors.csv", low_memory=False)
    services_df = pd.read_csv(raw_dir / "services.csv", low_memory=False)

    # Función auxiliar para parsear fechas con fallback flexible
    def _parse_dt_cols(df: pd.DataFrame, cols: list, fmt: Optional[str]):
        for col in cols:
            if col not in df.columns:
                continue
            if fmt:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, utc=True, errors="raise")
                    df[col] = df[col].dt.tz_convert(tz)
                    continue
                except Exception:
                    pass
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC").dt.tz_convert(tz)
            else:
                df[col] = df[col].dt.tz_convert(tz)

    # Parsear fechas
    _parse_dt_cols(service_labors_df, ["labor_created_at", "labor_start_date", "labor_end_date"], strict_datetime_format)
    _parse_dt_cols(services_df, ["created_at", "schedule_date"], strict_datetime_format)

    # Tablas de lookup para direcciones
    address_shop = (
        address_df.loc[address_df["shop"].notna(), ["shop", "address_id", "point", "name"]]
        .drop_duplicates(subset=["shop"])
        .set_index("shop")
    )
    address_alfred = (
        address_df.loc[address_df["alfred"].notna(), ["alfred", "address_id", "point", "name"]]
        .drop_duplicates(subset=["alfred"])
        .set_index("alfred")
    )

    # Merge principal
    unified = service_labors_df.merge(
        services_df,
        on="service_id",
        how="left",
        suffixes=("_labor", "_service"),
    )

    # Mapear address_id, point, name
    unified["address_id"] = (
        unified["shop"].map(address_shop["address_id"])
        .combine_first(unified["alfred"].map(address_alfred["address_id"]))
        .astype("Int64")
    )
    unified["address_point"] = (
        unified["shop"].map(address_shop["point"])
        .combine_first(unified["alfred"].map(address_alfred["point"]))
    )
    unified["address_name"] = (
        unified["shop"].map(address_shop["name"])
        .combine_first(unified["alfred"].map(address_alfred["name"]))
    )

    unified["city"] = unified["city"].astype("Int64").astype(str)

    # Filtrar servicios válidos
    valid_cities = list(codificacion_ciudades.keys())
    labors_raw_df = filter_valid_services(unified, city_list=valid_cities)

    return labors_raw_df


def remap_to_base_date(df, date_columns, base_day):
    """
    Ajusta las fechas de un DataFrame para que coincidan con un día base.

    Parámetros:
        df (pd.DataFrame): Datos a ajustar.
        date_columns (list): Nombres de columnas de tipo fecha.
        base_day (date): Día base para el remapeo.

    Retorna:
        pd.DataFrame: DataFrame con fechas ajustadas.
    """
    def remap(dt):
        if pd.isna(dt):
            return dt
        delta = (base_day - dt.date()).days
        return dt + timedelta(days=delta)

    for c in date_columns:
        df[c] = df[c].apply(remap)
    return df


def build_services_map_df(df):
    """
    Crea el mapa de puntos de inicio y fin por labor en cada servicio.

    Parámetros:
        df (pd.DataFrame): Datos filtrados y ordenados.

    Retorna:
        pd.DataFrame: Tabla con columnas ['service_id','labor_id','map_start_point','map_end_point'].
    """
    rows = []
    for svc, grp in df.groupby('service_id', sort=False):
        n = len(grp)
        idxs = grp.index.tolist()
        for i, idx in enumerate(idxs):
            r = grp.loc[idx]
            if n == 1:
                sp, ep = r['start_address_point'], r['end_address_point']
            else:
                if r['labor_category'] == 'VEHICLE_TRANSPORTATION':
                    if i == 0:
                        sp = r['start_address_point']
                        ep = grp.iloc[i+1]['address_point'] if i < n-1 else r['address_point']
                    else:
                        sp = grp.iloc[i-1]['address_point']
                        ep = r['address_point']
                else:
                    sp = ep = r['address_point']
            rows.append({
                'service_id': svc,
                'labor_id': r['labor_id'],
                'map_start_point': sp,
                'map_end_point': ep
            })
    return pd.DataFrame(rows)


def process_group(grp, dist_method, dist_dict, **kwargs):
    """
    Filtra y ordena las labores de un grupo (servicio) siguiendo la lógica de transporte.

    Parámetros:
        grp (pd.DataFrame): Subconjunto de labores de un mismo servicio.
        dist_method (str): Método de cálculo de distancia.
        dist_dict (dict): Diccionario de distancias precalculadas.

    Retorna:
        pd.DataFrame: Grupo filtrado y ordenado.
    """
    if len(grp) == 1:
        return grp if grp.iloc[0]['labor_category'] == 'VEHICLE_TRANSPORTATION' else grp.iloc[0:0]

    A_idx = list(grp.index[grp['labor_category'] == 'VEHICLE_TRANSPORTATION'])
    if not A_idx:
        return grp.iloc[0:0]

    B_idx = [i for i in grp.index[grp['labor_category'] != 'VEHICLE_TRANSPORTATION']
             if pd.notna(grp.at[i, 'shop']) and pd.notna(grp.at[i, 'address_point'])]
    if not B_idx:
        return grp.iloc[0:0]

    inits = [i for i in A_idx if grp.at[i, 'labor_name'] == 'Alfred Initial Transport']
    firstA = inits[0] if inits else A_idx[0]
    A_rem = [i for i in A_idx if i != firstA]

    start_pt = grp['start_address_point'].iloc[0]
    dist_map = {i: distance(start_pt, 
                            grp.at[i, 'address_point'], 
                            method=dist_method, 
                            dist_dict=dist_dict,
                            **kwargs) for i in B_idx}
    B_sorted = sorted(B_idx, key=lambda i: dist_map[i])[:len(A_idx)-1]

    A_rem_sorted = sorted(A_rem, key=lambda i: grp.at[i, 'labor_start_date'])
    needed_A = [firstA] + A_rem_sorted[:len(B_sorted)]

    final = []
    for j, b in enumerate(B_sorted):
        final += [needed_A[j], b]
    final.append(needed_A[-1])

    return grp.loc[final]


def compute_labor_duration_stats(df, city_col="city", labor_col="labor_type", 
                                 shop_col="shop", start_col="labor_start_date", 
                                 end_col="labor_end_date"):
    """
    Compute duration statistics of labors per city and labor type.
    Removes durations > 1 day (1440 minutes).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing labor records.
    city_col : str
        Column name for city.
    labor_col : str
        Column name for labor type.
    shop_col : str
        Column name for shop.
    start_col : str
        Column name for labor start datetime.
    end_col : str
        Column name for labor end datetime.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        [city, labor_type, n_shops, mean_min, median_min, std_min, p75_min, p85_min, p90_min].
    """

    # Ensure datetimes
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    # Compute duration in minutes
    df = df.copy()
    df["duration_min"] = (df[end_col] - df[start_col]).dt.total_seconds() / 60

    # Drop rows with missing durations
    df = df.dropna(subset=["duration_min"])

    # Remove unrealistic durations (> 1 day)
    df = df[df["duration_min"] <= 1440]

    # Group and compute stats
    stats = (
        df.groupby([city_col, labor_col])
        .agg(
            n_shops=(shop_col, "nunique"),
            mean_min=("duration_min", "mean"),
            median_min=("duration_min", "median"),
            std_min=("duration_min", "std"),
            p75_min=("duration_min", lambda x: np.percentile(x, 75)),
            p85_min=("duration_min", lambda x: np.percentile(x, 85)),
            p90_min=("duration_min", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )

    return stats

