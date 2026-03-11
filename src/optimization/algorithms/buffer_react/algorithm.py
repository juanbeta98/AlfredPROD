import logging
import os
import pandas as pd
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from src.optimization.algorithms.base import OptimizationAlgorithm
from src.optimization.algorithms.offline.algorithm import (
    OfflineAlgorithm,
    _city_dist_slice,
    _resolve_n_processes,
)
from src.optimization.algorithms.buffer_react.buffer_react_algorithms import (
    build_post_freeze_driver_states,
    split_labors_by_freeze_cutoff,
    strip_assignment_columns,
)
from src.optimization.common.distance_utils import batch_distance_matrix
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD
from src.utils.datetime_utils import now_colombia

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferReactAlgoConfig:
    """
    Configuration for the BUFFER_REACT algorithm.

    Identical to OfflineAlgoConfig plus ``time_previous_freeze``, which controls
    how many minutes before a labor's scheduled start it becomes frozen.
    """
    time_previous_freeze: int = 0          # minutes; 0 = only truly active labors are frozen
    distance_method: str = DEFAULT_DISTANCE_METHOD
    time_method: str = "speed_based"       # "speed_based" | "osrm_times"
    n_processes: Optional[int] = None
    precompute_distances: bool = True
    max_iterations_by_city: Optional[Dict[Any, int]] = None
    log_progress: bool = False


class BufferReactAlgorithm(OfflineAlgorithm):
    """
    BUFFER_REACT algorithm.

    Takes an existing schedule from context["preassigned"], freezes labors that
    are at or near their start time, and re-optimizes all remaining labors
    (together with any new unassigned labors in ``df``) using the OFFLINE
    assignment logic.

    Freeze rule
    -----------
    freeze_cutoff = decision_time + timedelta(minutes=time_previous_freeze)
    Labors with schedule_date ≤ freeze_cutoff AND an assigned driver are frozen.

    Decision time
    -------------
    Taken from ``params["decision_time"]`` (set by the solver from the request
    timestamp).  Falls back to the current Colombia time when not provided.
    """

    name = "BUFFER_REACT"

    def __init__(self, params: Dict[str, Any] | None = None):
        # Call the base OptimizationAlgorithm init directly — we build our own
        # config, so we skip OfflineAlgorithm's __init__.
        OptimizationAlgorithm.__init__(self, params)
        params = params or {}
        max_iterations = params.get("max_iterations_by_city") or params.get("max_iterations")
        self.config = BufferReactAlgoConfig(
            time_previous_freeze=int(params.get("time_previous_freeze", 0)),
            distance_method=params.get("distance_method") or DEFAULT_DISTANCE_METHOD,
            time_method=params.get("time_method", "speed_based"),
            n_processes=_resolve_n_processes(params.get("n_processes")),
            precompute_distances=bool(params.get("precompute_distances", True)),
            max_iterations_by_city=max_iterations,
            log_progress=bool(params.get("log_progress", False)),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Execute BUFFER_REACT.

        Returns
        -------
        results_df : DataFrame with assignment decisions.
        metrics    : KPIs and timing.
        artifacts  : ``{"moves_df": ..., "distance_method": ...}``.
        """
        t0 = perf_counter()

        self._validate_preconditions(df)

        # ---- Decision time & freeze cutoff -----------------------------------
        decision_time_raw = self.params.get("decision_time")
        if decision_time_raw is None:
            decision_time = now_colombia()
        else:
            decision_time = pd.Timestamp(decision_time_raw)
            if decision_time.tzinfo is None:
                decision_time = decision_time.tz_localize("America/Bogota")

        freeze_cutoff = decision_time + timedelta(minutes=self.config.time_previous_freeze)

        logger.info(
            "buffer_react_solve decision_time=%s freeze_cutoff=%s time_previous_freeze=%d",
            decision_time, freeze_cutoff, self.config.time_previous_freeze,
        )

        # ---- Existing schedule -----------------------------------------------
        preassigned = self.params.get("preassigned", {})
        preassigned_labors: pd.DataFrame = preassigned.get("labors_df", pd.DataFrame())
        preassigned_moves: pd.DataFrame  = preassigned.get("moves_df",  pd.DataFrame())

        # ---- Freeze / split --------------------------------------------------
        if not preassigned_labors.empty:
            frozen_labors, reassignable_labors, frozen_moves = split_labors_by_freeze_cutoff(
                preassigned_labors, preassigned_moves, freeze_cutoff
            )
        else:
            frozen_labors     = pd.DataFrame()
            reassignable_labors = pd.DataFrame()
            frozen_moves      = pd.DataFrame()

        post_freeze_states = build_post_freeze_driver_states(frozen_labors)

        # ---- Combine reassignable + new labors --------------------------------
        parts: List[pd.DataFrame] = []
        if not reassignable_labors.empty:
            parts.append(strip_assignment_columns(reassignable_labors))
        if not df.empty:
            parts.append(df)

        parts = [p.dropna(axis=1, how="all") for p in parts]
        combined_df = (
            pd.concat(parts, ignore_index=True).infer_objects(copy=False) if parts else pd.DataFrame()
        )

        # ---- Short-circuit when nothing to optimize --------------------------
        if combined_df.empty:
            logger.info("buffer_react_solve: no labors to optimize; returning frozen state")
            results_df = frozen_labors if not frozen_labors.empty else pd.DataFrame()
            moves_df   = frozen_moves  if not frozen_moves.empty  else pd.DataFrame()
            metrics = self._build_metrics(
                t0, cities_processed=0, postponed=0,
                moves_df=moves_df,
                frozen_count=len(frozen_labors),
                reassigned_count=0,
                new_count=len(df),
            )
            return results_df, metrics, {"moves_df": moves_df, "distance_method": self.config.distance_method}

        # ---- Master data -----------------------------------------------------
        master_data = self.params.get("master_data")
        directorio_df = master_data.directorio_df if master_data else pd.DataFrame()
        dist_dict_all = master_data.dist_dict       if master_data else {}
        alpha = self.params.get("alpha", 1)

        # ---- Per-city iteration loop -----------------------------------------
        city_keys  = self._city_keys(combined_df)
        cities     = city_keys.dropna().drop_duplicates().tolist()

        run_results: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        postponed_labors: List[Any] = []
        merged_time_dict: Dict[Any, Any] = {}

        for city_key in cities:
            df_city = combined_df[city_keys == city_key]
            if df_city.empty:
                continue

            dist_dict: Dict[Any, Any] = _city_dist_slice(dist_dict_all, city_key)
            time_dict: Dict[Any, Any] = {}

            # OSRM batch pre-computation (mirrors OfflineAlgorithm)
            if self.config.distance_method == "osrm" and self.config.precompute_distances:
                _osrm_url = os.environ.get("OSRM_URL", "")
                if _osrm_url:
                    _driver_positions = [
                        f"POINT ({row.longitud} {row.latitud})"
                        for _, row in directorio_df.iterrows()
                    ]
                    _labor_starts = df_city["map_start_point"].dropna().unique().tolist()
                    _labor_ends   = df_city["map_end_point"].dropna().unique().tolist()
                    _all_points   = list(dict.fromkeys(
                        _driver_positions + _labor_starts + _labor_ends
                    ))
                    _precomp_dist, _precomp_time = batch_distance_matrix(
                        _all_points, _all_points, _osrm_url,
                        include_times=(self.config.time_method == "osrm_times"),
                    )
                    if _precomp_dist:
                        dist_dict = {**_precomp_dist, **dist_dict}
                        time_dict = _precomp_time
                        merged_time_dict.update(time_dict)
                        logger.info(
                            "buffer_react osrm_precompute city=%s unique_points=%d pairs=%d",
                            city_key, len(_all_points), len(_precomp_dist),
                        )
                    else:
                        logger.warning(
                            "buffer_react osrm_precompute_failed city=%s", city_key,
                        )

            max_iter = self._get_max_iter(city_key)
            _mode = "parallel" if (self.config.n_processes or 1) > 1 else "sequential"
            logger.info(
                "buffer_react city_iteration_start city=%s n_iter=%d mode=%s",
                city_key, max_iter, _mode,
            )

            iter_args = [
                {
                    "city_key":        city_key,
                    "iter_idx":        i,
                    "labors_df":       df_city,
                    "dist_dict":       dist_dict,
                    "time_dict":       time_dict,
                    "directorio_df":   directorio_df,
                    "duraciones_df":   master_data.duraciones_df if master_data else pd.DataFrame(),
                    "distance_method": self.config.distance_method,
                    "time_method":     self.config.time_method,
                    "alpha":           alpha,
                    "model_params":    self.params.get("model_params"),
                    "master_data":     master_data,
                    "initial_drivers": post_freeze_states,  # override driver state with frozen end-positions
                }
                for i in range(1, max_iter + 1)
            ]

            results = self._run_iterations_parallel(iter_args)

            df_results = pd.DataFrame(results)
            if df_results.empty:
                continue

            best_idx  = self._select_best_iteration(df_results)
            inc_state = df_results.iloc[best_idx]

            postponed_labors.extend(inc_state.get("postponed_labors", []))
            run_results.append((inc_state["results"], inc_state["moves"]))

        # ---- Merge new results with frozen state -----------------------------
        new_results_df, new_moves_df = self._concat_run_results(run_results)

        all_results = [p for p in [frozen_labors, new_results_df] if isinstance(p, pd.DataFrame) and not p.empty]
        all_moves   = [p for p in [frozen_moves,  new_moves_df]   if isinstance(p, pd.DataFrame) and not p.empty]

        results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        moves_df   = pd.concat(all_moves,   ignore_index=True) if all_moves   else pd.DataFrame()

        metrics = self._build_metrics(
            t0,
            cities_processed=len(cities),
            postponed=len(postponed_labors),
            moves_df=moves_df,
            frozen_count=len(frozen_labors),
            reassigned_count=len(reassignable_labors),
            new_count=len(df),
        )

        artifacts = {
            "moves_df":        moves_df,
            "distance_method": self.config.distance_method,
            "time_method":     self.config.time_method,
            "time_dict":       merged_time_dict,
        }

        return results_df, metrics, artifacts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metrics(
        self,
        t0: float,
        *,
        cities_processed: int,
        postponed: int,
        moves_df: pd.DataFrame,
        frozen_count: int,
        reassigned_count: int,
        new_count: int,
    ) -> Dict[str, Any]:
        return {
            "algorithm":               self.name,
            "elapsed_seconds":         perf_counter() - t0,
            "cities_processed":        cities_processed,
            "postponed_labors_count":  postponed,
            "moves_rows":              len(moves_df) if hasattr(moves_df, "__len__") else None,
            "frozen_labors_count":     frozen_count,
            "reassigned_labors_count": reassigned_count,
            "new_labors_count":        new_count,
            "time_previous_freeze":    self.config.time_previous_freeze,
        }

    def _validate_preconditions(self, df: pd.DataFrame) -> None:
        required_cols = {
            "department_code",
            "schedule_date",
            "labor_id",
            "service_id",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"BUFFER_REACT algorithm missing required columns: {sorted(missing)}")
