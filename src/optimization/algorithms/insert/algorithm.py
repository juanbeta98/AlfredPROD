import logging
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
from src.optimization.common.utils import compute_workday_end
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

logger = logging.getLogger(__name__)


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
    n_processes: Optional[int] = None


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
            n_processes=params.get("n_processes"),
        )

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
                    return pd.DataFrame()
                mask = (base["department_code"] == city_key) & (
                    base["schedule_date"].dt.date == day
                )
                return base[mask].copy()

            city_base_labors = _city_day_slice(base_labors_df)
            city_base_moves = _city_day_slice(base_moves_df)

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

            iteration_results = []
            for i in range(n_iter):
                res = run_insertion_worker(
                    base_labors_df=city_base_labors,
                    base_moves_df=city_base_moves,
                    new_labors_df=city_new,
                    seed=seed_base + i,
                    city=city_key,
                    fecha=day_str,
                    directorio_df=directorio_df,
                    drivers=drivers,
                    dist_dict=dist_dict,
                    distance_method=dist_method,
                    alfred_speed=alfred_speed,
                    vehicle_transport_speed=vehicle_transport_speed,
                    tiempo_alistar=tiempo_alistar,
                    tiempo_finalizacion=tiempo_finalizacion,
                    tiempo_gracia=tiempo_gracia,
                    early_buffer=early_buffer,
                    workday_end_dt=workday_end_dt,
                    duraciones_df=dur_df,
                )
                iteration_results.append(res)

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
        artifacts = {
            "moves_df": moves_df,
            "distance_method": dist_method,
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
