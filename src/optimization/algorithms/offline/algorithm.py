import logging
import multiprocessing as mp
import pandas as pd
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from src.location import series_location_key
from src.optimization.algorithms.base import OptimizationAlgorithm
from src.optimization.algorithms.offline.offline_algorithms import run_assignment_algorithm
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

logger = logging.getLogger(__name__)


def _city_dist_slice(dist_dict_all: Any, city_key: Any) -> Dict[Any, Any]:
    if not isinstance(dist_dict_all, dict):
        return {}
    if city_key in dist_dict_all:
        value = dist_dict_all.get(city_key, {})
        return value if isinstance(value, dict) else {}
    city_key_txt = str(city_key)
    for key, value in dist_dict_all.items():
        if str(key) == city_key_txt:
            return value if isinstance(value, dict) else {}
    return {}


@dataclass(frozen=True)
class OfflineAlgoConfig:
    """
    Configuration for OFFLINE baseline algorithm.
    Keep this minimal and expand as you migrate experiment knobs.
    """
    distance_method: str = DEFAULT_DISTANCE_METHOD
    n_processes: Optional[int] = None
    max_iterations_by_city: Optional[Dict[Any, int]] = None  # city_key -> max_iter


class OfflineAlgorithm(OptimizationAlgorithm):
    """
    OFFLINE baseline algorithm.

    Expected behavior:
    - Input: parsed/validated DataFrame in the pipeline schema
    - Output: DataFrame with additional columns (same rows as input, plus decisions)
    - Metrics: algorithm KPIs and timing info
    """

    name = "OFFLINE"

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        params = params or {}
        max_iterations = params.get("max_iterations_by_city", params.get("max_iterations"))
        self.config = OfflineAlgoConfig(
            distance_method=params.get("distance_method") or DEFAULT_DISTANCE_METHOD,
            n_processes=params.get("n_processes"),
            max_iterations_by_city=max_iterations,
        )

    def solve(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Execute offline baseline algorithm.

        Returns:
            results_df: DataFrame (same base as df, plus added columns)
            metrics: algorithm metrics
            artifacts: auxiliary outputs (e.g., moves_df)
        """
        t0 = perf_counter()

        self._validate_preconditions(df)

        # ---- Prepare algorithm-specific structures from df ----
        prepared = self._prepare_inputs(df)
        master_data = self.params.get("master_data")
        directorio_df = master_data.directorio_df if master_data else pd.DataFrame()
        duraciones_df = master_data.duraciones_df if master_data else pd.DataFrame()
        dist_dict_all = prepared.get("dist_dict", {}) if isinstance(prepared, dict) else {}
        if not dist_dict_all and master_data:
            dist_dict_all = master_data.dist_dict
        alpha = prepared.get("alpha", self.params.get("alpha", 1)) if isinstance(prepared, dict) else self.params.get("alpha", 1)
        city_keys = self._city_keys(df)
        cities = city_keys.dropna().drop_duplicates().tolist()

        run_results: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        postponed_labors: List[Any] = []

        for city_key in cities:
            df_city = df[city_keys == city_key]
            if df_city.empty:
                continue

            dist_dict = _city_dist_slice(dist_dict_all, city_key)
        
            max_iter = self._get_max_iter(city_key)
            iter_args = [
                {
                    "city_key": city_key,
                    "iter_idx": i,
                    "labors_df": df_city,
                    "dist_dict": dist_dict,
                    "directorio_df": directorio_df,
                    "duraciones_df": duraciones_df,
                    "distance_method": self.config.distance_method,
                    "alpha": alpha,
                    "model_params": self.params.get("model_params"),
                    "master_data": master_data,
                }
                for i in range(1, max_iter + 1)
            ]

            results = self._run_iterations_parallel(iter_args)

            df_results = pd.DataFrame(results)
            best_idx = self._select_best_iteration(df_results)
            inc_state = df_results.iloc[best_idx]

            postponed_labors.extend(inc_state.get("postponed_labors", []))
            run_results.append((inc_state["results"], inc_state["moves"]))

        results_df, moves_df = self._concat_run_results(run_results)

        metrics = {
            "algorithm": self.name,
            "elapsed_seconds": perf_counter() - t0,
            "cities_processed": len(cities),
            "postponed_labors_count": len(postponed_labors),
            "moves_rows": len(moves_df) if hasattr(moves_df, "__len__") else None,
        }

        artifacts = {
            "moves_df": moves_df,
            "distance_method": self.config.distance_method,
        }

        return results_df, metrics, artifacts

    # --------------------------------------------------
    # Internal methods (wrap your existing functions here)
    # --------------------------------------------------

    def _validate_preconditions(self, df: pd.DataFrame) -> None:
        required_cols = {
            "department_code",
            "schedule_date",
            "labor_id",
            "service_id",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"OFFLINE algorithm missing required columns: {sorted(missing)}")

    def _city_keys(self, df: pd.DataFrame) -> pd.Series:
        keys = series_location_key(df)
        keys = keys.astype("string").str.strip()
        return keys.where(keys.notna() & keys.ne(""), pd.NA)

    def _prepare_inputs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Adapter layer: replace this progressively with true DF-driven prep.

        For now, you can call your existing prep_algorithm_inputs() here,
        or derive the same artifacts from df.
        """
        # TODO: Implement / adapt from your existing prep_algorithm_inputs
        # raise NotImplementedError

        prepared = {
            "dist_dict": {},  # Placeholder
            "duraciones_df": pd.DataFrame(),  # Placeholder
            "alpha": self.params.get("alpha", 3.0),  # Placeholder
        }

        return prepared

    def _filter_labors_by_date(self, labors_df: pd.DataFrame, fecha: Any) -> pd.DataFrame:
        # TODO: wrap filter_labors_by_date(...)
        raise NotImplementedError

    def _filter_history(self, directorio_hist_df: pd.DataFrame, *, city: Any, fecha: Any) -> pd.DataFrame:
        # TODO: wrap flexible_filter(...)
        raise NotImplementedError

    def _get_max_iter(self, city: Any) -> int:
        if self.config.max_iterations_by_city and city in self.config.max_iterations_by_city:
            return self.config.max_iterations_by_city[city]
        # fallback
        return 1

    def _run_iterations_parallel(self, iter_args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Production note:
        - multiprocessing + pandas can be expensive.
        - keep this optional via config.
        """
        if not iter_args:
            return []

        if self.config.n_processes and self.config.n_processes > 1:
            with mp.Pool(processes=self.config.n_processes) as pool:
                return pool.map(_run_single_iteration, iter_args)

        return [_run_single_iteration(args) for args in iter_args]

    def _select_best_iteration(self, df_results: pd.DataFrame) -> int:
        if df_results is None or df_results.empty:
            raise ValueError("No iteration results to select from")

        df = df_results.copy()
        if "success" in df.columns:
            df = df[df["success"]]
        if df.empty:
            return int(df_results.index[0])

        def _score_moves(moves_df: Any) -> Tuple[int, float]:
            if not isinstance(moves_df, pd.DataFrame) or moves_df.empty:
                return 0, float("inf")

            labor_count = (
                int(moves_df["labor_id"].nunique())
                if "labor_id" in moves_df.columns
                else 0
            )
            if "driver_distance" in moves_df.columns:
                total_dist = float(moves_df["driver_distance"].sum())
            elif "distance_km" in moves_df.columns:
                total_dist = float(moves_df["distance_km"].sum())
            else:
                total_dist = float("inf")

            return labor_count, total_dist

        scores = df["moves"].apply(_score_moves)
        df = df.assign(
            labor_count=[s[0] for s in scores],
            total_distance=[s[1] for s in scores],
        )

        max_labors = df["labor_count"].max()
        best = df[df["labor_count"] == max_labors]
        best_idx = best["total_distance"].astype(float).idxmin()
        return int(best_idx)

    def _concat_run_results(self, run_results: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not run_results:
            return pd.DataFrame(), pd.DataFrame()

        results_parts = [res for res, _ in run_results if isinstance(res, pd.DataFrame) and not res.empty]
        moves_parts = [moves for _, moves in run_results if isinstance(moves, pd.DataFrame) and not moves.empty]

        results_df = pd.concat(results_parts, axis=0, copy=False) if results_parts else pd.DataFrame()
        moves_df = pd.concat(moves_parts, axis=0, copy=False) if moves_parts else pd.DataFrame()

        return results_df, moves_df

    def _merge_results_into_input(self, input_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align results onto original input rows.
        Implementation depends on your keys (labor_id? service_id+lseq?).
        """
        # TODO: implement merge logic once we confirm keys
        raise NotImplementedError


def _infer_day_str(labors_df: pd.DataFrame) -> Optional[str]:
    if labors_df is None or labors_df.empty:
        return None
    schedule_dates = labors_df.get("schedule_date")
    if schedule_dates is None or schedule_dates.empty:
        return None
    day = pd.to_datetime(schedule_dates.iloc[0]).date()
    return str(day)


def _run_single_iteration(args: Dict[str, Any]) -> Dict[str, Any]:
    labors_df = args.get("labors_df")
    if labors_df is None or labors_df.empty:
        return {
            "city_key": args.get("city_key"),
            "iter": args.get("iter_idx"),
            "results": pd.DataFrame(),
            "moves": pd.DataFrame(),
            "postponed_labors": [],
            "success": False,
        }

    day_str = args.get("day_str") or _infer_day_str(labors_df)
    city_key = args.get("city_key")

    results_df, moves_df, postponed_labors = run_assignment_algorithm(
        labors_df=labors_df,
        day_str=day_str,
        city_key=city_key,
        dist_method=args.get("distance_method"),
        alpha=args.get("alpha", 1),
        iter_idx=args.get("iter_idx", 0),
        model_params=args.get("model_params"),
        master_data=args.get("master_data")
    )

    return {
        "city_key": city_key,
        "iter": args.get("iter_idx"),
        "results": results_df,
        "moves": moves_df,
        "postponed_labors": postponed_labors,
        "success": True,
    }
