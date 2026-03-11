import logging
import multiprocessing as mp
import os
import pandas as pd
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from src.optimization.algorithms.base import OptimizationAlgorithm
from src.optimization.algorithms.insert.insert_algorithms import (
    get_drivers,
    run_insertion_worker,
    select_best_result,
)
from src.optimization.common.distance_utils import batch_distance_matrix
from src.optimization.common.movements import _filter_drivers_by_city
from src.optimization.common.utils import compute_workday_end
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-process shared state
# ---------------------------------------------------------------------------
# Large DataFrames (base_labors_df, directorio_df, etc.) are constant across
# all iterations of a city run. Passing them via pool.imap would pickle them
# once per task. Instead we push them into each worker once via the pool
# initializer and only send the tiny per-iteration seed over IPC.

_WORKER_SHARED: Dict[str, Any] = {}


def _init_worker(shared_args: Dict[str, Any]) -> None:
    global _WORKER_SHARED
    _WORKER_SHARED = shared_args



def _run_single_iter_shared(slim_args: Dict[str, Any]) -> Dict[str, Any]:
    args = {**_WORKER_SHARED, **slim_args}
    return run_insertion_worker(
        base_labors_df=args["base_labors_df"],
        base_moves_df=args["base_moves_df"],
        new_labors_df=args["new_labors_df"],
        seed=args["seed"],
        city=args["city"],
        fecha=args["fecha"],
        directorio_df=args["directorio_df"],
        drivers=args["drivers"],
        dist_dict=args["dist_dict"],
        distance_method=args["distance_method"],
        time_method=args["time_method"],
        time_dict=args["time_dict"],
        alfred_speed=args["alfred_speed"],
        vehicle_transport_speed=args["vehicle_transport_speed"],
        tiempo_alistar=args["tiempo_alistar"],
        tiempo_finalizacion=args["tiempo_finalizacion"],
        tiempo_gracia=args["tiempo_gracia"],
        early_buffer=args["early_buffer"],
        workday_end_dt=args["workday_end_dt"],
        duraciones_df=args.get("duraciones_df"),
    )


def _resolve_n_processes(value: Any) -> Optional[int]:
    """
    Normalise the n_processes param from request.json.

    Accepts:
      None / null / "None" / 0  → None  (sequential)
      -1 / "-1"                 → os.cpu_count()  (all cores)
      positive int or str       → int(value)
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("none", "null", ""):
            return None
        try:
            value = int(value)
        except ValueError:
            return None
    value = int(value)
    if value <= 0:
        return os.cpu_count() if value == -1 else None
    return value


def _city_dist_slice(dist_dict_all: Any, city_key: str) -> Dict[Any, Any]:
    if not isinstance(dist_dict_all, dict):
        return {}
    city_key_str = str(city_key)
    for key, value in dist_dict_all.items():
        if str(key) == city_key_str:
            return value if isinstance(value, dict) else {}
    return {}


@dataclass(frozen=True)
class InsertAlgoConfig:
    """Configuration for the INSERT algorithm."""
    optimization_obj: Optional[str] = None
    distance_method: str = DEFAULT_DISTANCE_METHOD
    time_method: str = "speed_based"   # "speed_based" | "osrm_times"
    n_processes: Optional[int] = None
    precompute_distances: bool = True  # use OSRM Table API to batch-compute all distances before iterations
    log_progress: bool = False  # show tqdm bar per city; set via request.json algorithm.params.log_progress


class InsertAlgorithm(OptimizationAlgorithm):
    """
    INSERT algorithm.

    Inserts new (unassigned) labors into an already-built preassigned schedule.

    - 1 new service per city  → greedy insertion (1 iteration)
    - >1 new services per city → randomised permutation search with n_iterations
      from solver_settings.max_iterations

    Input: parsed/validated DataFrame in the pipeline schema.
    Output: DataFrame of newly processed labors + moves artifact.
    The caller (main.py) concatenates these with the preassigned outputs.
    """

    name = "INSERT"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        params = params or {}
        self.config = InsertAlgoConfig(
            optimization_obj=params.get("optimization_obj"),
            distance_method=params.get("distance_method") or DEFAULT_DISTANCE_METHOD,
            time_method=params.get("time_method", "speed_based"),
            n_processes=_resolve_n_processes(params.get("n_processes")),
            precompute_distances=bool(params.get("precompute_distances", True)),
            log_progress=bool(params.get("log_progress", False)),
        )

    def _run_iterations(
        self,
        n_iter: int,
        seed_base: int,
        city_key: str,
        shared: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run n_iter worker calls, in parallel or sequentially, with optional tqdm."""
        slim_args = [{"seed": seed_base + i} for i in range(n_iter)]

        if self.config.n_processes and self.config.n_processes > 1:
            with mp.Pool(
                processes=self.config.n_processes,
                initializer=_init_worker,
                initargs=(shared,),
            ) as pool:
                if self.config.log_progress:
                    from tqdm import tqdm
                    results = list(tqdm(
                        pool.imap_unordered(_run_single_iter_shared, slim_args, chunksize=1),
                        total=n_iter,
                        desc=f"INSERT city={city_key}",
                        unit="iter",
                    ))
                else:
                    results = pool.map(_run_single_iter_shared, slim_args)
            return results

        # Sequential
        if self.config.log_progress:
            from tqdm import tqdm
            iterable = tqdm(slim_args, desc=f"INSERT city={city_key}", unit="iter")
        else:
            iterable = slim_args

        results = []
        for slim in iterable:
            results.append(run_insertion_worker(
                base_labors_df=shared["base_labors_df"],
                base_moves_df=shared["base_moves_df"],
                new_labors_df=shared["new_labors_df"],
                seed=slim["seed"],
                city=shared["city"],
                fecha=shared["fecha"],
                directorio_df=shared["directorio_df"],
                drivers=shared["drivers"],
                dist_dict=shared["dist_dict"],
                distance_method=shared["distance_method"],
                time_method=shared["time_method"],
                time_dict=shared["time_dict"],
                alfred_speed=shared["alfred_speed"],
                vehicle_transport_speed=shared["vehicle_transport_speed"],
                tiempo_alistar=shared["tiempo_alistar"],
                tiempo_finalizacion=shared["tiempo_finalizacion"],
                tiempo_gracia=shared["tiempo_gracia"],
                early_buffer=shared["early_buffer"],
                workday_end_dt=shared["workday_end_dt"],
                duraciones_df=shared.get("duraciones_df"),
            ))
        return results

    def solve(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Execute INSERT algorithm.

        Returns:
            results_df : DataFrame of newly processed labors (inserted + failed).
            metrics    : algorithm KPIs.
            artifacts  : dict with key "moves_df" containing new labor move triplets.
        """
        t0 = perf_counter()
        self._validate_preconditions(df)

        # ---- Extract parameters ----
        master_data = self.params.get("master_data")
        model_params = self.params.get("model_params")
        preassigned = self.params.get("preassigned", {})

        dist_dict_all = master_data.dist_dict if master_data else {}
        directorio_df = master_data.directorio_df if master_data else pd.DataFrame()
        duraciones_df = master_data.duraciones_df if master_data else pd.DataFrame()

        max_iterations: Dict[str, int] = self.params.get("max_iterations", {})
        seed_base: int = model_params.seed if model_params and hasattr(model_params, "seed") else 42
        dist_method: str = self.config.distance_method

        base_labors_df: pd.DataFrame = preassigned.get("labors_df", pd.DataFrame())
        base_moves_df: pd.DataFrame = preassigned.get("moves_df", pd.DataFrame())

        # ---- Identify new (unassigned) labors ----
        new_labors_df = df[df["assigned_driver"].isna()].copy()
        if new_labors_df.empty:
            logger.info("INSERT: no unassigned labors found — returning empty result.")
            return (
                pd.DataFrame(columns=df.columns),
                {"algorithm": self.name, "elapsed_seconds": 0.0},
                {"moves_df": pd.DataFrame(), "distance_method": dist_method},
            )

        result_labors_parts: List[pd.DataFrame] = []
        result_moves_parts: List[pd.DataFrame] = []
        updated_base_parts: List[pd.DataFrame] = []
        updated_base_moves_parts: List[pd.DataFrame] = []
        merged_time_dict: Dict[Any, Any] = {}

        date_series = new_labors_df["schedule_date"].dt.date

        for (city_key, day), city_new in new_labors_df.groupby(
            [new_labors_df["department_code"], date_series]
        ):
            city_key = str(city_key)
            day_str = str(day)
            n_services = city_new["service_id"].nunique()
            n_iter = int(max_iterations.get(city_key, 1)) if n_services > 1 else 1

            dist_dict = _city_dist_slice(dist_dict_all, city_key)

            # City+day slice of preassigned schedule
            def _city_day_slice(base: pd.DataFrame) -> pd.DataFrame:
                if base.empty:
                    return base.iloc[:0].copy()  # preserve column schema even when 0 rows
                mask = (base["department_code"] == city_key) & (
                    base["schedule_date"].dt.date == day
                )
                return base[mask].copy()

            city_base_labors = _city_day_slice(base_labors_df)
            city_base_moves = _city_day_slice(base_moves_df)

            # --- OSRM batch pre-computation ---
            # Compute the full driver→labor distance matrix in one Table API call so
            # all iterations start with a warm dist_dict and make zero OSRM HTTP calls.
            # When time_method="osrm_times", also fetch the duration matrix.
            time_dict: Dict[Any, Any] = {}
            if self.config.distance_method == "osrm" and self.config.precompute_distances:
                _osrm_url = os.environ.get("OSRM_URL", "")
                if _osrm_url:
                    _city_directorio = _filter_drivers_by_city(directorio_df, city_key)
                    _driver_positions = [
                        f"POINT ({row.longitud} {row.latitud})"
                        for _, row in _city_directorio.iterrows()
                    ]
                    _all_labors = pd.concat(
                        [f for f in [city_new, city_base_labors] if not f.empty],
                        ignore_index=True,
                    )
                    _labor_starts = _all_labors["map_start_point"].dropna().unique().tolist()
                    _labor_ends   = _all_labors["map_end_point"].dropna().unique().tolist()
                    _all_points = list(dict.fromkeys(
                        _driver_positions + _labor_starts + _labor_ends
                    ))
                    logger.debug(
                        "osrm_precompute_points city=%s drivers=%d labor_starts=%d labor_ends=%d unique_total=%d",
                        city_key, len(_driver_positions), len(_labor_starts), len(_labor_ends), len(_all_points),
                    )
                    _precomp_dist, _precomp_time = batch_distance_matrix(
                        _all_points, _all_points, _osrm_url,
                        include_times=(self.config.time_method == "osrm_times"),
                    )
                    if _precomp_dist:
                        dist_dict = {**_precomp_dist, **dist_dict}   # existing entries take priority
                        time_dict = _precomp_time
                        merged_time_dict.update(time_dict)
                        logger.info(
                            "osrm_precompute city=%s unique_points=%d pairs=%d time_pairs=%d",
                            city_key, len(_all_points), len(_precomp_dist), len(_precomp_time),
                        )
                    else:
                        logger.warning(
                            "osrm_precompute_failed city=%s — iterations will use per-call fallback",
                            city_key,
                        )

            drivers = get_drivers(city_base_labors, directorio_df, city_key)

            logger.info(
                "INSERT city=%s day=%s new_services=%d n_iter=%d n_drivers=%d",
                city_key, day_str, n_services, n_iter, len(drivers),
            )

            workday_end_dt = compute_workday_end(
                day_str=day_str,
                workday_end_str=(
                    model_params.workday_end_str
                    if model_params and hasattr(model_params, "workday_end_str")
                    else "22:00:00"
                ),
                tzinfo=city_new["schedule_date"].dt.tz,
            )

            alfred_speed: float = model_params.alfred_speed_kmh if model_params else 20.0
            vehicle_transport_speed: float = (
                model_params.vehicle_transport_speed_kmh if model_params else 30.0
            )
            tiempo_alistar: float = model_params.tiempo_alistar_min if model_params else 10.0
            tiempo_finalizacion: float = (
                model_params.tiempo_finalizacion_min if model_params else 5.0
            )
            tiempo_gracia: float = model_params.tiempo_gracia_min if model_params else 30.0
            early_buffer: float = model_params.tiempo_previo_min if model_params else 30.0
            dur_df: Optional[pd.DataFrame] = duraciones_df if not duraciones_df.empty else None

            shared = {
                "base_labors_df": city_base_labors,
                "base_moves_df": city_base_moves,
                "new_labors_df": city_new,
                "city": city_key,
                "fecha": day_str,
                "directorio_df": directorio_df,
                "drivers": drivers,
                "dist_dict": dist_dict,
                "distance_method": dist_method,
                "time_method": self.config.time_method,
                "time_dict": time_dict,
                "alfred_speed": alfred_speed,
                "vehicle_transport_speed": vehicle_transport_speed,
                "tiempo_alistar": tiempo_alistar,
                "tiempo_finalizacion": tiempo_finalizacion,
                "tiempo_gracia": tiempo_gracia,
                "early_buffer": early_buffer,
                "workday_end_dt": workday_end_dt,
                "duraciones_df": dur_df,
            }

            iteration_results = self._run_iterations(
                n_iter=n_iter,
                seed_base=seed_base,
                city_key=city_key,
                shared=shared,
            )

            best = select_best_result(iteration_results)
            new_labor_ids = set(city_new["labor_id"].tolist())

            if best is not None:
                # Return only the newly processed labors and their move triplets
                inserted_labors = best["results"][
                    best["results"]["labor_id"].isin(new_labor_ids)
                ]
                inserted_moves = best["moves"][
                    best["moves"]["labor_id"].isin(new_labor_ids)
                ]
                result_labors_parts.append(inserted_labors)
                result_moves_parts.append(inserted_moves)

                # Capture updated base labors — downstream shifts applied by INSERT
                # may have changed actual_start/actual_end of preassigned labors.
                # Return them so main.py can use the updated versions instead of
                # the originals, preventing false driver_overlap validation errors.
                if not city_base_labors.empty:
                    base_labor_ids = set(city_base_labors["labor_id"].tolist())
                    updated_base = best["results"][
                        best["results"]["labor_id"].isin(base_labor_ids)
                    ]
                    if not updated_base.empty:
                        updated_base_parts.append(updated_base)
                    updated_base_moves = best["moves"][
                        best["moves"]["labor_id"].isin(base_labor_ids)
                    ]
                    if not updated_base_moves.empty:
                        updated_base_moves_parts.append(updated_base_moves)
                logger.info(
                    "INSERT city=%s day=%s inserted=%d/%d dist=%.1f best_seed=%d",
                    city_key, day_str,
                    best["num_inserted"], len(city_new),
                    best["dist"], best["seed"],
                )
            else:
                # All iterations failed — pass through unassigned labors
                result_labors_parts.append(city_new)
                logger.warning(
                    "INSERT city=%s day=%s all iterations failed for %d labors",
                    city_key, day_str, len(city_new),
                )

        results_df = (
            pd.concat(result_labors_parts, ignore_index=True)
            if result_labors_parts
            else pd.DataFrame(columns=df.columns)
        )
        moves_df = (
            pd.concat(result_moves_parts, ignore_index=True)
            if result_moves_parts
            else pd.DataFrame()
        )

        metrics = {
            "algorithm": self.name,
            "elapsed_seconds": round(perf_counter() - t0, 3),
            "new_labors_total": len(new_labors_df),
            "new_labors_inserted": int(
                results_df["assigned_driver"].notna().sum()
            ) if not results_df.empty else 0,
        }
        updated_base_df = (
            pd.concat(updated_base_parts, ignore_index=True)
            if updated_base_parts
            else pd.DataFrame()
        )
        updated_base_moves_df = (
            pd.concat(updated_base_moves_parts, ignore_index=True)
            if updated_base_moves_parts
            else pd.DataFrame()
        )
        artifacts = {
            "moves_df": moves_df,
            "distance_method": dist_method,
            "time_method": self.config.time_method,
            "time_dict": merged_time_dict,
            "updated_base_labors_df": updated_base_df,
            "updated_base_moves_df": updated_base_moves_df,
        }

        return results_df, metrics, artifacts

    def _validate_preconditions(self, df: pd.DataFrame) -> None:
        required_cols = {
            "department_code",
            "schedule_date",
            "labor_id",
            "service_id",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"INSERT algorithm missing required columns: {sorted(missing)}"
            )
