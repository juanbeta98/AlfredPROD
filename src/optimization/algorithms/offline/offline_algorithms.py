import logging
import pandas as pd
import numpy as np
import random
import hashlib

import math
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ...common.distance_utils import parse_point, distance
from ...common.utils import compute_workday_end
from ...common.movements import (
    build_driver_movements,
    _location_fields,
    _filter_drivers_by_city,
    _extract_driver_key,
)
from ...settings.model_params import ModelParams
from ...settings.solver_settings import DEFAULT_DISTANCE_METHOD
from ....data.id_normalization import normalize_id_value
from ....data.loading.master_data_loader import MasterData


DistDict = Dict[str, Any] | None
DriverState = Dict[str, Any]
Candidate = Dict[str, Any]


'''
MAIN EXECUTION ALGORITHM
'''
def run_assignment_algorithm(
    model_params: ModelParams,
    master_data: MasterData,
    labors_df: pd.DataFrame,
    day_str: str,
    city_key: str,
    dist_method: str = DEFAULT_DISTANCE_METHOD,
    dist_dict: Optional[DistDict] = None,
    alpha: float = 1.0,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Ejecuta la simulación de asignación de labores de manera cronológica para una ciudad en un día específico.

    Usa la duración estimada (percentil 75) de cada tipo de labor por ciudad. Si no existe,
    usa el promedio del percentil 75 en otras ciudades. Como último recurso, usa `TIMEPO_OTHER`.
    """

    kwargs = dict(kwargs)
    if model_params is None:
        model_params = kwargs.pop("model_params", None)
    else:
        kwargs.pop("model_params", None)
    if master_data is None:
        master_data = kwargs.pop("master_data", None)
    else:
        kwargs.pop("master_data", None)
    if model_params is None:
        model_params = ModelParams()
    model_params.validate()
    iteration_index = kwargs.pop("iter_idx", 0)
    seed_material = f"{model_params.seed}|{day_str}|{city_key}|{iteration_index}"
    run_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(run_seed)
    kwargs["osrm_url"] = model_params.osrm_url

    directorio_df = master_data.directorio_df
    duraciones_df = master_data.duraciones_df

    if labors_df.empty:
        return pd.DataFrame(), pd.DataFrame(), list()

    workday_end_str = model_params.workday_end_str
    tiempo_previo = model_params.tiempo_previo_min
    tiempo_gracia = model_params.tiempo_gracia_min
    tiempo_alistar = model_params.tiempo_alistar_min
    tiempo_other = model_params.tiempo_other_min
    tiempo_finalizacion = model_params.tiempo_finalizacion_min
    alfred_speed = model_params.alfred_speed_kmh
    vehicle_transport_speed = model_params.vehicle_transport_speed_kmh

    df_sorted = prepare_iteration_data(labors_df)
    assigned = [pd.NA] * len(labors_df)
    starts = [pd.NaT] * len(labors_df)
    ends = [pd.NaT] * len(labors_df)
    distances = [0] * len(labors_df)
    overtime_minutes_list: List[Optional[float]] = [None] * len(labors_df)
    service_end_times: Dict[Any, Optional[pd.Timestamp]] = {}

    drivers = init_drivers(df_sorted, directorio_df=directorio_df, city=city_key, **kwargs)

    workday_end_dt = compute_workday_end(
        day_str=day_str,
        workday_end_str=workday_end_str,
        tzinfo="America/Bogota"
    )

    service_postponed: Dict[Any, bool] = {}
    service_block_after_first_afterhours_nontransport: Dict[Any, bool] = {}
    postponed_labors: List[Dict[str, Any]] = []

    # --- Lógica de asignación principal ---
    for _, row in df_sorted.iterrows():
        original_idx = int(row['original_idx'])
        if original_idx < 0 or original_idx >= len(assigned):
            raise IndexError(
                f"Invalid original_idx={original_idx} for labors_df with {len(assigned)} rows."
            )
        service_id = row['service_id']

        if service_end_times.get(service_id) is pd.NaT:
            continue
        prev_end = service_end_times.get(service_id)

        # Decide postponement
        should_postpone, reason, updates = _evaluate_postponement(
            row,
            prev_end,
            workday_end_dt,
            service_postponed,
            service_block_after_first_afterhours_nontransport
        )

        # Apply updates
        if updates.get("postpone_service"):
            service_postponed[service_id] = True

        if updates.get("block_future_nontransport"):
            service_block_after_first_afterhours_nontransport[service_id] = True

        # If postponed → record and skip scheduling
        if should_postpone:
            postponed_labors.append({
                "labor_row": row,
                "service_id": service_id,
                "reason": reason,
                "prev_end": prev_end,
                "workday_end": workday_end_dt
            })
            service_end_times[service_id] = pd.NaT
            continue

        if row['labor_category'] == 'VEHICLE_TRANSPORTATION':
        
            pick, dist_dict = get_driver_wrapper(
                drivers, 
                row, 
                prev_end, 
                tiempo_previo, 
                tiempo_gracia,
                alfred_speed, 
                dist_method, 
                dist_dict,
                alpha, 
                rng=rng,
                **kwargs
            )

            if not pick:
                overtime_pick = _get_overtime_fallback(
                    drivers, row, tiempo_gracia, alfred_speed, dist_method, dist_dict, **kwargs
                )
                if not overtime_pick:
                    service_end_times[service_id] = pd.NaT
                    continue
                logger.debug(
                    "labor_assigned_with_overtime labor_id=%s service_id=%s driver=%s overtime_min=%.1f",
                    row.get('labor_id'), service_id, overtime_pick['drv'], overtime_pick['overtime_minutes'],
                )
                pick = overtime_pick
                overtime_minutes_list[original_idx] = overtime_pick['overtime_minutes']

            drv = pick['drv']
            is_last = not labors_df[
                (labors_df['service_id'] == service_id) &
                (labors_df['labor_sequence'] > row['labor_sequence'])
            ].empty

            astart, aend, dist_km = assign_task_to_driver(
                drivers[drv],
                pick['arrival'],
                prev_end or (row['schedule_date'] - timedelta(minutes=tiempo_previo)),
                row['map_start_point'], 
                row['map_end_point'], 
                is_last, 
                tiempo_alistar, 
                tiempo_finalizacion,
                vehicle_transport_speed, 
                dist_method, 
                dist_dict, 
                **kwargs
            )
            
            assigned[original_idx] = drv
            starts[original_idx] = astart
            ends[original_idx] = aend
            distances[original_idx] = dist_km
            service_end_times[service_id] = aend

        else:
            # --- Labores no transporte: usar estimated_time o p75 de duraciones_df ---
            astart = prev_end or row['schedule_date']

            est_time = row.get("estimated_time")
            if est_time is not None and pd.notna(est_time):
                duration_min = float(est_time)
            else:
                # Buscar primero por ciudad + labor_type
                dur_row = duraciones_df[
                    (duraciones_df["city"] == city_key) &
                    (duraciones_df["labor_type"] == row["labor_type"])
                ]

                if not dur_row.empty:
                    duration_min = float(dur_row["p75_min"].iloc[0])
                else:
                    # Si no existe en esa ciudad, sacar promedio global para ese labor_type
                    labor_rows = duraciones_df[duraciones_df["labor_type"] == row["labor_type"]]
                    if not labor_rows.empty:
                        duration_min = float(labor_rows["p75_min"].mean())
                    else:
                        # Fallback final
                        duration_min = tiempo_other

            aend = astart + timedelta(minutes=duration_min)
            starts[original_idx], ends[original_idx] = astart, aend
            service_end_times[service_id] = aend

    # --- Construcción DataFrame final ---
    df_result = labors_df.copy()
    df_result['assigned_driver'] = assigned
    df_result['actual_start'] = pd.to_datetime(starts)
    df_result['actual_end'] = pd.to_datetime(ends)
    df_result['dist_km'] = distances
    df_result['overtime_minutes'] = overtime_minutes_list

    failed_services = {
        service_id
        for service_id, service_end in service_end_times.items()
        if pd.isna(service_end)
    }
    df_result['actual_status'] = df_result['service_id'].apply(
        lambda service_id: 'FAILED' if service_id in failed_services else 'COMPLETED'
    )
    
    solution_cols = ['assigned_driver', 'actual_start', 'actual_end']
    for col in solution_cols:
        if col in df_result.columns:
            df_result[col] = df_result.apply(lambda x: pd.NA if x['actual_status']=='FAILED' else x[col], axis=1)

    # — Reconstrucción de movimientos y tiempos libres —
    df_moves = build_driver_movements(
        labors_df=df_result, 
        directory_df=directorio_df, 
        day_str=day_str, 
        dist_method=dist_method, 
        dist_dict=dist_dict, 
        ALFRED_SPEED=alfred_speed, 
        city_key=city_key,
        **kwargs)

    return df_result, df_moves, postponed_labors, dist_dict


'''
OPERATORS
'''
def get_candidate_drivers(
    drivers: Dict[str, DriverState],
    row: pd.Series,
    prev_end: Optional[pd.Timestamp],
    tiempo_previo: int,
    tiempo_gracia: int,
    alfred_speed: float,
    dist_method: str,
    dist_dict: Optional[DistDict],
    **kwargs: Any,
) -> Tuple[List[Candidate], DistDict]:
    """
    Genera lista de conductores candidatos para una labor VT.

    Parámetros
    ----------
    drivers : dict
        Diccionario con información de cada conductor (posición, disponibilidad).
    row : pd.Series
        Fila del DataFrame con información de la labor.
    prev_end : Timestamp
        Fin del servicio anterior para este `service_id`.
    TIEMPO_PREVIO : int
        Minutos antes de la hora programada para permitir llegada anticipada.
    TIEMPO_GRACIA : int
        Minutos de tolerancia para llegada tardía.
    ALFRED_SPEED : float
        Velocidad media del vehículo en km/h.
    method : str
        Método de cálculo de distancia.

    Retorna
    -------
    list of dict
        Lista de conductores con hora estimada de llegada.
    """
    kwargs["return_dist_dict"] = True
    dist_dict_local: DistDict = dist_dict or {}
    
    sched = row['schedule_date']
    early = prev_end or (sched - timedelta(minutes=tiempo_previo))
    late  = (prev_end + timedelta(minutes=tiempo_gracia)) if prev_end else (sched + timedelta(minutes=tiempo_gracia))
    
    cands: List[Candidate] = []
    for name, drv in drivers.items():
        av = drv['available']
        if av.time() < drv['work_start']:
            av = pd.Timestamp(datetime.combine(av.date(), drv['work_start']), tz=av.tz)

        # For getting candidate drivers always use the haversine formula, otherwise it get's to heavy
        dkm, dist_dict_local = distance(
            drv["position"],
            row["map_start_point"],
            method=dist_method,
            dist_dict=dist_dict_local,
            **kwargs,
        )
        arr = av + timedelta(minutes=(0 if math.isnan(dkm) else dkm/alfred_speed*60))

        if arr <= late:
            cands.append({'drv': name, 'arrival': arr, 'dist_km': dkm})

    return cands, dist_dict_local


def _get_overtime_fallback(
    drivers: Dict[str, DriverState],
    row: pd.Series,
    tiempo_gracia: int,
    alfred_speed: float,
    dist_method: str,
    dist_dict: Optional[DistDict] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    When no driver can arrive within the normal time window, select the one
    requiring the least overtime (earliest start relative to their shift start).

    The selected driver is assumed to depart before their shift to arrive exactly
    at the grace deadline (`late`). Returns arrival=late so assign_task_to_driver
    places astart=late.

    Returns None only if there are no drivers at all.
    """
    if not drivers:
        return None

    kwargs["return_dist_dict"] = True
    dist_dict_local: DistDict = dist_dict or {}

    late = row['schedule_date'] + timedelta(minutes=tiempo_gracia)

    best: Optional[Dict[str, Any]] = None
    best_overtime: float = float('inf')

    for name, drv in drivers.items():
        # Use available as-is (= shift start from init); no shift floor applied here
        av = drv['available']
        dkm, dist_dict_local = distance(
            drv["position"],
            row["map_start_point"],
            method=dist_method,
            dist_dict=dist_dict_local,
            **kwargs,
        )
        travel_min = 0.0 if math.isnan(dkm) else dkm / alfred_speed * 60
        arrival = av + timedelta(minutes=travel_min)
        overtime = (arrival - late).total_seconds() / 60  # always > 0 in fallback path

        if overtime < best_overtime:
            best_overtime = overtime
            best = {'drv': name, 'arrival': late, 'dist_km': dkm, 'overtime_minutes': overtime}

    return best


def select_from_candidates(
    cands: List[Candidate],
    alpha: float,
    *,
    rng: Optional[random.Random] = None,
) -> Optional[Candidate]:
    """
    Selecciona un candidato de la lista usando el criterio GRASP con RCL,
    pero ahora basado en la *distancia* (dist_km) en lugar del tiempo de llegada.

    Parámetros
    ----------
    cands : list of dict
        Lista de candidatos. Cada candidato debe tener las claves:
            - 'drv': nombre del conductor
            - 'arrival': tiempo estimado de llegada (Timestamp)
            - 'dist_km': distancia desde la posición actual al punto inicial de la labor
    ALPHA : float
        Parámetro de control de aleatoriedad en [0,1]. 
        - 0 → completamente codicioso (elige el de menor distancia)
        - 1 → completamente aleatorio

    Retorna
    -------
    dict o None
        Candidato seleccionado de la RCL o None si no hay candidatos.
    """
    if not cands:
        return None

    chooser = rng if rng is not None else random

    # --- 1. Extraer las distancias de cada candidato ---
    costs = [c['dist_km'] for c in cands]

    # --- 2. Caso trivial: si solo hay un candidato, lo elegimos ---
    if len(costs) == 1:
        return cands[0]

    # --- 3. Calcular los límites para formar la RCL ---
    min_cost, max_cost = min(costs), max(costs)

    # Si todas las distancias son iguales, elegir uno al azar
    if min_cost == max_cost:
        return chooser.choice(cands)

    # --- 4. Umbral de inclusión en la RCL ---
    # Se permiten candidatos con costo <= thr
    thr = min_cost + alpha * (max_cost - min_cost)

    # --- 5. Construir la RCL con base en la distancia ---
    RCL = [c for c, cost in zip(cands, costs) if cost <= thr]

    # --- 6. Selección final (aleatoria dentro de la RCL) ---
    return chooser.choice(RCL)


'''
AUXILIARY METHODS FOR MAIN ALGORITHMS
'''

def init_drivers(
    labors_df: pd.DataFrame,
    directorio_df: pd.DataFrame,
    city: str = "BOGOTA",
    ignore_schedule: bool = False,
    **kwargs: Any,
) -> Dict[str, DriverState]:
    """
    Inicializa la posición y disponibilidad inicial de los conductores para la simulación.

    Parámetros
    ----------
    df_labores : pd.DataFrame
        DataFrame con las labores programadas, debe incluir la columna 'schedule_date'.
    directorio_df : pd.DataFrame
        DataFrame con la información de los conductores, incluyendo:
        - 'city_id'
        - 'driver_id'
        - 'latitud', 'longitud' (coordenadas iniciales)
        - 'start_time' (hora de inicio de jornada, formato 'HH:MM:SS')
    ciudad : str, opcional
        Ciudad para filtrar conductores (por defecto 'BOGOTA').
    ignore_schedule : bool, opcional
        Si True, los conductores estarán disponibles desde medianoche (00:00:00).

    Retorna
    -------
    dict
        Diccionario de conductores, cada clave es el nombre del conductor y el valor es un dict con:
        - 'position' : str → coordenadas iniciales en formato WKT 'POINT (lon lat)'
        - 'available' : pd.Timestamp → hora inicial de disponibilidad con zona horaria
        - 'work_start' : datetime.time → hora de inicio de jornada
    """
    
    if labors_df.empty:
        return {}
    
    # directorio_df = kwargs.get("directorio_df")
    
    # Zona horaria a partir de las labores
    tz = labors_df['schedule_date'].dt.tz
    if tz is None:
        raise ValueError("La columna 'schedule_date' no tiene zona horaria asignada.")
    
    # Fecha mínima de las labores
    first_date = labors_df['schedule_date'].dt.date.min()
    if pd.isna(first_date):
        raise ValueError("No se pudo determinar la fecha mínima en 'schedule_date'.")
    
    # Filtrar conductores por ciudad
    df_ciudad = _filter_drivers_by_city(directorio_df, city)
    
    conductores: Dict[str, DriverState] = {}
    for _, conductor in df_ciudad.iterrows():
        if pd.isna(conductor['latitud']) or pd.isna(conductor['longitud']):
            continue

        driver_key = _extract_driver_key(conductor)
        if driver_key is None:
            continue
        
        if ignore_schedule:
            hora_inicio = time(0, 0, 0)  # medianoche
        else:
            try:
                hora_inicio = datetime.strptime(conductor['start_time'], '%H:%M:%S').time()
            except ValueError:
                raise ValueError(f"Formato invalido de hora para el conductor {driver_key}")
        
        disponibilidad = datetime.combine(first_date, hora_inicio)
        
        conductores[driver_key] = {
            "position": f"POINT ({conductor['longitud']} {conductor['latitud']})",
            "available": pd.Timestamp(disponibilidad).tz_localize(tz),
            "work_start": hora_inicio,
        }
    
    return conductores


def prepare_iteration_data(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    # Keep a stable positional index for write-back into assignment arrays.
    # The incoming DataFrame index may be sparse/non-contiguous after filtering.
    df = df_cleaned.copy().reset_index(drop=True)
    df["original_idx"] = range(len(df))

    # Reassignment candidates (formerly preassigned, infeasible services that were
    # broken and fed back) must be processed before regular unassigned labors so they
    # retain their implicit customer-commitment priority.  The flag is set in main.py
    # via _prepare_service_rows_for_reassignment(); regular rows default to 0.
    if "reassignment_priority" in df.columns:
        df["reassignment_priority"] = pd.to_numeric(
            df["reassignment_priority"], errors="coerce"
        ).fillna(0)
    else:
        df["reassignment_priority"] = 0

    df = df.sort_values(
        ["reassignment_priority", "schedule_date", "service_id", "labor_sequence"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return df


def assign_task_to_driver(
    driver_data: DriverState,
    arrival: pd.Timestamp,
    early: pd.Timestamp,
    start_point: str,
    end_point: str,
    is_last_in_service: bool,
    tiempo_alistar: int,
    tiempo_finalizacion: int,
    vehicle_speed: float,
    method: str,
    dist_dict: Optional[DistDict] = None,
    **kwargs: Any,
) -> Tuple[pd.Timestamp, pd.Timestamp, float]:
    """
    Asigna una tarea a un conductor y actualiza su disponibilidad y posición.

    Retorna
    -------
    (inicio, fin) : tuple of pd.Timestamp
    """
    astart = max(arrival, early)
    dist_km, _ = distance(start_point, end_point, method=method, dist_dict=dist_dict, **kwargs)
    dur = tiempo_alistar + (0 if math.isnan(dist_km) else dist_km/vehicle_speed*60) \
          + (tiempo_finalizacion if not is_last_in_service else 0)
    aend = astart + timedelta(minutes=dur)

    driver_data["available"] = aend
    driver_data["position"] = end_point

    return astart, aend, dist_km


def get_driver_wrapper(
    drivers: Dict[str, DriverState],
    row: pd.Series,
    prev_end: Optional[pd.Timestamp],
    tiempo_previo: int,
    tiempo_gracia: int,
    alfred_speed: float,
    dist_method: str,
    dist_dict: Optional[DistDict],
    alpha: float,
    rng: Optional[random.Random] = None,
    **kwargs: Any,
) -> Tuple[Optional[Candidate], DistDict]:
    """
    Wrapper para seleccionar un conductor usando el algoritmo
    (RCL con get_candidate_drivers + select_from_candidates).

    Parámetros
    ----------
    drivers : dict
        Diccionario de conductores inicializados.
    row : pd.Series
        Fila de la labor, debe incluir 'alfred' si use_real=True.
    prev_end : Timestamp
        Hora de finalización del servicio previo.
    TIEMPO_PREVIO : int
        Minutos antes de la hora programada para permitir llegada anticipada.
    TIEMPO_GRACIA : int
        Minutos de tolerancia para llegada tardía.
    ALFRED_SPEED : float
        Velocidad media de movimiento en km/h.
    method : str
        Método de cálculo de distancias.
    dist_dict : dict
        Diccionario de distancias precalculadas.
    ALPHA : float
        Parámetro GRASP para seleccionar en modo algoritmo.
    Retorna
    -------
    dict o None
        Candidato elegido de la RCL o None
    """

    candidate_dist_method = dist_method
    cands, updated_dist_dict = get_candidate_drivers(
        drivers=drivers, 
        row=row, 
        prev_end=prev_end,
        tiempo_previo=tiempo_previo, 
        tiempo_gracia=tiempo_gracia,
        alfred_speed=alfred_speed, 
        dist_method=candidate_dist_method, 
        dist_dict=dist_dict,
        **kwargs
    )
    return select_from_candidates(cands, alpha, rng=rng), updated_dist_dict


''' POSTPONING LABORS'''
def _evaluate_postponement(
    row: pd.Series,
    prev_end: Optional[pd.Timestamp],
    workday_end_dt: pd.Timestamp,
    service_postponed: Dict[Any, bool],
    service_block_after_first_afterhours_nontransport: Dict[Any, bool],
) -> Tuple[bool, Optional[str], Dict[str, bool]]:
    """
    Determines whether the given labor should be postponed based on:
      - category (transport vs non-transport)
      - prev_end
      - workday_end_dt
      - per-service postponement states

    Returns:
        should_postpone (bool)
        reason (str or None)
        updates (dict) → caller applies these changes to service dictionaries
    """

    service_id = row["service_id"]
    cat = row["labor_category"]
    updates = {}

    # If previously postponed, continue postponing
    if service_postponed.get(service_id, False):
        return True, "service_already_postponed", updates

    # === TRANSPORT LABOR ===
    if cat == "VEHICLE_TRANSPORTATION":

        # Start candidate = prev_end or schedule_date - TIEMPO_PREVIO (computed outside)
        if pd.notna(prev_end):
            astart_candidate = prev_end
        else:
            astart_candidate = row["schedule_date"]

        # Transport cannot start after workday end
        if astart_candidate > workday_end_dt:
            updates["postpone_service"] = True
            return True, "transport_start_after_workday", updates

        return False, None, updates

    # === NON-TRANSPORT LABOR ===
    else:
        astart = prev_end or row["schedule_date"]

        # Case A: This non-transport starts after the workday boundary as FIRST labor of the service
        if pd.isna(prev_end) and astart > workday_end_dt:
            updates["postpone_service"] = True
            return True, "first_nontransport_after_workday", updates

        # Case B: This labor is after-hours because prev_end overflowed
        if pd.notna(prev_end) and prev_end > workday_end_dt:

            # first allowed after-hours non-transport
            if not service_block_after_first_afterhours_nontransport.get(service_id, False):
                updates["block_future_nontransport"] = True
                return False, None, updates

            # subsequent after-hours non-transport → postpone
            updates["postpone_service"] = True
            return True, "subsequent_afterhours_nontransport", updates

        # Regular case → allowed
        return False, None, updates





def compute_avg_times(df: pd.DataFrame) -> dict:
    """
    Recalcula los tiempos promedio por tipo de labor a partir de las fechas de inicio y fin.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'labor_name', 'labor_start_date', 'labor_end_date'.

    Retorna
    -------
    dict
        Diccionario {labor_name: promedio_duracion_en_minutos}.
    """
    df_temp = (
        df.dropna(subset=['labor_start_date','labor_end_date'])
          .assign(duration_td=lambda d: d['labor_end_date'] - d['labor_start_date'])
    )
    df_temp = df_temp[df_temp['duration_td'] <= pd.Timedelta(days=1)]
    df_temp['duration_min'] = df_temp['duration_td'].dt.total_seconds() / 60
    avg_times_map = df_temp.groupby('labor_name')['duration_min'].mean().to_dict()
    return avg_times_map
