import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from src.data.id_normalization import normalize_id_columns
from src.optimization.algorithms.buffer_react import algorithm
from src.utils.datetime_utils import normalize_datetime_columns_to_colombia, now_colombia

from .algorithms.registry import get_algorithm
from src.data.loading.master_data_loader import MasterData, load_master_data
from .settings.solver_settings import OptimizationSettings
logger = logging.getLogger(__name__)


class SolverError(Exception): ...
class SolverValidationError(SolverError): ...
class SolverExecutionError(SolverError): ...


class OptimizationSolver:
    SOLVER_VERSION = "0.2.0"

    def __init__(
        self,
        input_df: pd.DataFrame,
        settings: OptimizationSettings,
        context: Optional[Dict[str, Any]] = None,
        master_data_override: Optional[MasterData] = None,
    ):
        self.input_df = input_df.copy()
        self.settings = settings
        self.context = context or {}

        self.run_id = str(uuid.uuid4())
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # store algo metrics after run
        self._algo_metrics: Dict[str, Any] = {}
        self._algo_artifacts: Dict[str, Any] = {}

        self._validate_input()

        # Build and validate shared model parameters ONCE
        self.model_params = self.settings.model_params
        self.model_params.validate()
        self.master_data = master_data_override or load_master_data(self.settings.master_data)


    def solve(self) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        logger.debug("Solver execution started", extra={"run_id": self.run_id})

        self._start_timer()

        try:
            self._prepare_data()

            # ---------------------------------------------
            # Algorithm selection & execution
            # ---------------------------------------------
            algo_name = self.settings.algorithm.strip().upper()

            algo_params = self.settings.for_algorithm(algo_name)
            algo_params["master_data"] = self.master_data
            algo_params["run_id"] = self.run_id
            algo_params["preassigned"] = self.context.get("preassigned", {})

            logger.info(
                f"solver_selected algorithm={algo_name}",
                extra={
                    # "run_id": self.run_id,
                    "algorithm": algo_name,
                    "input_rows": len(self.input_df),
                    "planning_day": self.input_df["planning_day"].iloc[0]
                    if "planning_day" in self.input_df.columns and not self.input_df.empty
                    else None,
                    "algo_params": {k: v for k, v in algo_params.items() if k not in ("master_data", "preassigned")},
                },
            )

            algorithm = get_algorithm(algo_name, algo_params)

            solve_result = algorithm.solve(self.input_df)
            if not isinstance(solve_result, tuple) or len(solve_result) != 3:
                raise SolverExecutionError("Algorithm must return a 3-tuple (results_df, metrics, artifacts)")
            results_df, algo_metrics, algo_artifacts = solve_result
            if not isinstance(algo_artifacts, dict):
                raise SolverExecutionError("Algorithm artifacts must be a dict")

            self._algo_metrics = algo_metrics or {}
            self._algo_artifacts = algo_artifacts or {}
            if "dist_dict" not in self._algo_artifacts and self.master_data is not None:
                self._algo_artifacts["dist_dict"] = self.master_data.dist_dict

            results_df = self._postprocess_results(results_df)

        except SolverError:
            raise 
        except Exception as exc:
            logger.exception("Unexpected solver failure")
            raise SolverExecutionError(str(exc)) from exc
        finally:
            self._stop_timer()

        metrics = self._collect_metrics(results_df)

        logger.debug("Solver execution finished", extra={"run_id": self.run_id})
        return results_df, metrics, self._algo_artifacts

    # --------------------------------------------------
    # Solver preconditions (minimal)
    # --------------------------------------------------

    def _validate_input(self) -> None:
        if not isinstance(self.input_df, pd.DataFrame):
            raise SolverValidationError("Input data must be a pandas DataFrame")
        if self.input_df.empty:
            raise SolverValidationError("Input DataFrame is empty")

        # minimal contract columns that parsing should provide
        required_cols = {"service_id", "labor_id", "schedule_date"}
        missing = required_cols - set(self.input_df.columns)
        if missing:
            raise SolverValidationError(f"Missing required input columns: {sorted(missing)}")

        required_location_cols = {"department_code"}
        missing_location = required_location_cols - set(self.input_df.columns)
        if missing_location:
            raise SolverValidationError(
                f"Missing location columns in input DataFrame: {sorted(missing_location)}"
            )

        if self.input_df.index.has_duplicates:
            raise SolverValidationError("Input DataFrame index has duplicates")

    # --------------------------------------------------
    # Normalize / derive fields for all algorithms
    # --------------------------------------------------

    def _prepare_data(self) -> None:
        df = self.input_df

        normalize_id_columns(
            df,
            columns=(
                "service_id",
                "labor_id",
                "start_address_id",
                "end_address_id",
                "city_code",
                "department_code",
                "labor_type",
                "assigned_driver",
                "shop_address_id",
                "shop_id",
            ),
            include_detected=True,
        )

        # Normalize datetimes (robust)
        normalize_datetime_columns_to_colombia(
            df,
            ["schedule_date", "created_at", "actual_start", "actual_end"],
        )
        self._ensure_map_points(df)

        if df["schedule_date"].isna().any():
            # note: business validation should catch this earlier,
            # but we guard anyway to avoid silent failures.
            raise SolverValidationError("schedule_date contains invalid datetime values")

        planning_days = df["schedule_date"].dt.date.unique()
        if len(planning_days) != 1:
            raise SolverValidationError(
                f"Expected single planning day but found {len(planning_days)}: {planning_days}"
            )

        df["planning_day"] = df["schedule_date"].dt.date.astype(str)

        for col in ("city_code", "department_code"):
            if col not in df.columns:
                continue
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].eq(""), col] = pd.NA

        self.input_df = df

    @staticmethod
    def _clean_point(value: Any) -> str | None:
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        text = str(value).strip()
        return text or None

    @classmethod
    def _ensure_map_points(cls, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        if "map_start_point" not in df.columns:
            df["map_start_point"] = pd.NA
        if "map_end_point" not in df.columns:
            df["map_end_point"] = pd.NA

        if "service_id" not in df.columns:
            return

        ordering = df.copy()
        if "labor_sequence" in ordering.columns:
            ordering["__labor_sequence_num"] = pd.to_numeric(
                ordering["labor_sequence"],
                errors="coerce",
            )
        else:
            ordering["__labor_sequence_num"] = pd.NA

        if "schedule_date" in ordering.columns:
            ordering["__schedule_ts"] = pd.to_datetime(
                ordering["schedule_date"],
                errors="coerce",
                utc=True,
            )
        else:
            ordering["__schedule_ts"] = pd.NaT

        ordering["__orig_idx"] = ordering.index
        ordering = ordering.sort_values(
            ["service_id", "__labor_sequence_num", "__schedule_ts", "__orig_idx"],
            kind="stable",
        )

        def _row_stop_point(row: pd.Series) -> str | None:
            # VT labors travel between stops and have no shop of their own.
            # Returning None lets the surrounding logic pick the correct
            # prev_stop / next_stop as the leg endpoints.
            if str(row.get("labor_category") or "").strip().upper() == "VEHICLE_TRANSPORTATION":
                return None
            for col in ("address_point", "shop_address_point", "end_address_point", "start_address_point"):
                if col not in row.index:
                    continue
                candidate = cls._clean_point(row.get(col))
                if candidate:
                    return candidate
            return None

        for _, group in ordering.groupby("service_id", sort=False):
            group_idx = list(group.index)
            if not group_idx:
                continue

            rows = [df.loc[idx] for idx in group_idx]
            stop_points = [_row_stop_point(row) for row in rows]
            group_size = len(rows)

            for pos, idx in enumerate(group_idx):
                row = df.loc[idx]

                existing_start = cls._clean_point(row.get("map_start_point"))
                existing_end = cls._clean_point(row.get("map_end_point"))
                if existing_start and existing_end:
                    continue

                current_stop = stop_points[pos]
                prev_stop = stop_points[pos - 1] if pos > 0 else None
                next_stop = stop_points[pos + 1] if pos < (group_size - 1) else None
                start_addr = cls._clean_point(row.get("start_address_point"))
                end_addr = cls._clean_point(row.get("end_address_point"))
                prior_stop = cls._clean_point(row.get("prior_stop_point"))

                category = str(row.get("labor_category") or "").strip().upper()

                if group_size == 1:
                    computed_start = prior_stop or start_addr or current_stop or end_addr
                    computed_end = end_addr or current_stop or computed_start
                elif category == "VEHICLE_TRANSPORTATION":
                    if pos == 0:
                        computed_start = prior_stop or start_addr or prev_stop or current_stop or end_addr
                        computed_end = next_stop or current_stop or end_addr or computed_start
                    else:
                        computed_start = prev_stop or start_addr or current_stop or end_addr
                        computed_end = current_stop or next_stop or end_addr or computed_start
                else:
                    anchor = current_stop or prev_stop or next_stop or end_addr or start_addr
                    computed_start = anchor
                    computed_end = anchor

                if not existing_start and computed_start:
                    df.at[idx, "map_start_point"] = computed_start
                if not existing_end:
                    fallback_end = computed_end or computed_start or end_addr or start_addr
                    if fallback_end:
                        df.at[idx, "map_end_point"] = fallback_end

    # --------------------------------------------------
    # Dispatcher to algorithms
    # --------------------------------------------------

    def _run_optimization(self) -> pd.DataFrame:
        algo_name = self.settings.algorithm.strip().upper()

        algo_params = self.settings.for_algorithm(algo_name)

        logger.debug(
            "Running algorithm",
            extra={
                "run_id": self.run_id,
                "algorithm": algo_name,
                "algo_params": {k: v for k, v in algo_params.items() if k != "token"},
            },
        )

        algorithm = get_algorithm(algo_name, algo_params)

        # algorithms still receive DataFrame
        solve_result = algorithm.solve(self.input_df)
        if not isinstance(solve_result, tuple) or len(solve_result) != 3:
            raise SolverExecutionError("Algorithm must return a 3-tuple (results_df, metrics, artifacts)")
        results_df, algo_metrics, algo_artifacts = solve_result
        if not isinstance(algo_artifacts, dict):
            raise SolverExecutionError("Algorithm artifacts must be a dict")

        self._algo_metrics = algo_metrics or {}
        self._algo_artifacts = algo_artifacts or {}
        if "dist_dict" not in self._algo_artifacts and self.master_data is not None:
            self._algo_artifacts["dist_dict"] = self.master_data.dist_dict
        return results_df

    # --------------------------------------------------
    # Output contract enforcement
    # --------------------------------------------------

    def _postprocess_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(results_df, pd.DataFrame):
            raise SolverExecutionError("Algorithm did not return a DataFrame")

        # common expectation: results correspond to input rows
        # prefer matching by labor_id if it's unique
        if len(results_df) != len(self.input_df):
            logger.warning(
                "Result row count differs from input",
                extra={"in": len(self.input_df), "out": len(results_df)},
            )

        return results_df

    # --------------------------------------------------
    # Timing / metrics
    # --------------------------------------------------

    def _start_timer(self) -> None:
        self._start_time = time.perf_counter()
        self.started_at = now_colombia()

    def _stop_timer(self) -> None:
        self._end_time = time.perf_counter()
        self.ended_at = now_colombia()

    def _collect_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        execution_time = None
        if self._start_time is not None and self._end_time is not None:
            execution_time = self._end_time - self._start_time

        algo_name = self.settings.algorithm.strip().upper()
        algo_params = self.settings.for_algorithm(algo_name)

        return {
            "run_id": self.run_id,
            "solver_version": self.SOLVER_VERSION,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "execution_time_seconds": execution_time,
            "algorithm": algo_name,
            "hyperparameters": algo_params,
            "input_rows": len(self.input_df),
            "output_rows": len(results_df),
            "algorithm_metrics": self._algo_metrics,
        }
