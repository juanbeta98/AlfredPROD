import logging
import multiprocessing as mp
import os
import pandas as pd
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from src.geo.location import series_location_key
from src.optimization.algorithms.base import OptimizationAlgorithm
from src.optimization.algorithms.offline.offline_algorithms import run_assignment_algorithm
from src.optimization.common.distance_utils import batch_distance_matrix
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker-process shared state
# ---------------------------------------------------------------------------
# Large DataFrames (labors_df, directorio_df, etc.) are identical across all
# iterations of a city run. Passing them through pool.imap/map would pickle
# them once per task. Instead we push them into each worker once via the pool
# initializer and only send the tiny per-iteration args over IPC.

_WORKER_SHARED: Dict[str, Any] = {}


def _init_worker(shared_args: Dict[str, Any]) -> None:
    global _WORKER_SHARED
    _WORKER_SHARED = shared_args


def _run_single_iteration_shared(slim_args: Dict[str, Any]) -> Dict[str, Any]:
    return _run_single_iteration({**_WORKER_SHARED, **slim_args})


# Keys that are constant for every iteration within one city call.
_SHARED_KEYS = frozenset({
    "labors_df", "dist_dict", "directorio_df", "duraciones_df",
    "distance_method", "alpha", "model_params", "master_data",
})


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
    n_warmup: int = 5
    precompute_distances: bool = True  # use OSRM Table API to batch-compute all distances before iterations
    max_iterations_by_city: Optional[Dict[Any, int]] = None  # city_key -> max_iter
    log_progress: bool = False  # show tqdm bar per city; set via request.json algorithm.params.log_progress


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
            n_processes=_resolve_n_processes(params.get("n_processes")),
            n_warmup=int(params.get("n_warmup", 5)),
            precompute_distances=bool(params.get("precompute_distances", True)),
            max_iterations_by_city=max_iterations,
            log_progress=bool(params.get("log_progress", False)),
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

            # --- OSRM batch pre-computation ---
            # Compute the full driver→labor distance matrix in one Table API call so
            # all iterations start with a warm dist_dict and make zero OSRM HTTP calls.
            if self.config.distance_method == "osrm" and self.config.precompute_distances:
                _osrm_url = os.environ.get("OSRM_URL", "")
                if _osrm_url:
                    _driver_positions = [
                        f"POINT ({row.longitud} {row.latitud})"
                        for _, row in directorio_df.iterrows()
                    ]
                    _labor_starts = df_city["map_start_point"].dropna().unique().tolist()
                    _labor_ends   = df_city["map_end_point"].dropna().unique().tolist()
                    _origins      = list(dict.fromkeys(_driver_positions + _labor_starts))
                    _destinations = list(dict.fromkeys(_labor_starts + _labor_ends))
                    _precomp = batch_distance_matrix(_origins, _destinations, _osrm_url)
                    if _precomp:
                        dist_dict = {**_precomp, **dist_dict}   # existing entries take priority
                        logger.info(
                            "osrm_precompute city=%s origins=%d destinations=%d pairs=%d",
                            city_key, len(_origins), len(_destinations), len(_precomp),
                        )
                    else:
                        logger.warning(
                            "osrm_precompute_failed city=%s — iterations will use per-call fallback",
                            city_key,
                        )

            max_iter = self._get_max_iter(city_key)
            _mode = "parallel" if (self.config.n_processes or 1) > 1 else "sequential"
            logger.info(
                "city_iteration_start city=%s n_iter=%d mode=%s",
                city_key, max_iter, _mode,
            )
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

            _best_results = inc_state.get("results")
            if isinstance(_best_results, pd.DataFrame) and not _best_results.empty:
                _ot_series = pd.to_numeric(
                    _best_results.get("overtime_minutes", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0)
                _ot_count = int((_ot_series > 0).sum())
                if _ot_count > 0:
                    logger.warning(
                        "best_iteration_has_overtime city=%s iter=%s overtime_labors=%s total_overtime_min=%.1f",
                        city_key,
                        inc_state.get("iter"),
                        _ot_count,
                        float(_ot_series.sum()),
                    )

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
        - set log_progress=True (via request.json algorithm.params) to show a tqdm bar.
        - n_warmup sequential iterations run before the pool only when distance_method='osrm'.
        """
        if not iter_args:
            return []

        city_key = iter_args[0].get("city_key", "?")

        if self.config.n_processes and self.config.n_processes > 1:
            warmup_results: List[Dict[str, Any]] = []
            warm_dist_dict: Dict[str, Any] = {}

            if self.config.distance_method == "osrm":
                # --- Warmup: run n_warmup iterations sequentially to build the distance cache ---
                # Only needed for OSRM; haversine/manhattan compute distances in-process
                # and never populate dist_dict, so warmup would add overhead with no benefit.
                warmup_list = iter_args[:self.config.n_warmup]
                remaining   = iter_args[self.config.n_warmup:]

                warmup_iterable = warmup_list
                if self.config.log_progress:
                    from tqdm import tqdm
                    warmup_iterable = tqdm(
                        warmup_list,
                        desc=f"city={city_key} [warmup]",
                        unit="iter",
                    )
                for w_args in warmup_iterable:
                    result = _run_single_iteration({**w_args, "dist_dict": warm_dist_dict})
                    warm_dist_dict = result.get("dist_dict") or warm_dist_dict
                    warmup_results.append(result)

                if not remaining:
                    return warmup_results
            else:
                remaining = iter_args

            # --- Parallel: all workers start with the (possibly pre-populated) cache ---
            shared = {k: v for k, v in iter_args[0].items() if k in _SHARED_KEYS}
            shared["dist_dict"] = warm_dist_dict
            slim = [
                {k: v for k, v in args.items() if k not in _SHARED_KEYS}
                for args in remaining
            ]
            chunksize = max(1, len(slim) // (self.config.n_processes * 4))
            with mp.Pool(
                processes=self.config.n_processes,
                initializer=_init_worker,
                initargs=(shared,),
            ) as pool:
                if self.config.log_progress:
                    from tqdm import tqdm
                    # chunksize=1 so tqdm ticks per completed iteration, not per chunk.
                    # IPC overhead is negligible: slim args are tiny (iter_idx + city_key);
                    # large shared DataFrames live in each worker's _WORKER_SHARED global.
                    pool_results = list(tqdm(
                        pool.imap_unordered(
                            _run_single_iteration_shared, slim, chunksize=1,
                        ),
                        total=len(slim),
                        desc=f"city={city_key}",
                        unit="iter",
                    ))
                else:
                    pool_results = pool.map(_run_single_iteration_shared, slim, chunksize=chunksize)

            return warmup_results + pool_results

        # --- Sequential: thread the cache forward across all iterations ---
        accumulated_dist_dict: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        if self.config.log_progress:
            from tqdm import tqdm
            iterable = tqdm(iter_args, desc=f"city={city_key}", unit="iter")
        else:
            iterable = iter_args

        for args in iterable:
            result = _run_single_iteration({**args, "dist_dict": accumulated_dist_dict})
            accumulated_dist_dict = result.get("dist_dict") or accumulated_dist_dict
            results.append(result)

        return results

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

    results_df, moves_df, postponed_labors, dist_dict_out = run_assignment_algorithm(
        labors_df=labors_df,
        day_str=day_str,
        city_key=city_key,
        dist_method=args.get("distance_method"),
        dist_dict=args.get("dist_dict"),
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
        "dist_dict": dist_dict_out,
        "success": True,
    }
