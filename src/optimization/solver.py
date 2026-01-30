import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

import pandas as pd

from src.optimization.algorithms.buffer_react import algorithm

from .algorithms.registry import get_algorithm
from src.data.master_data_loader import load_master_data
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
        self.master_data = load_master_data(self.settings.master_data)


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

            logger.info(
                f"solver_selected algorithm={algo_name}",
                extra={
                    # "run_id": self.run_id,
                    "algorithm": algo_name,
                    "input_rows": len(self.input_df),
                    "planning_day": self.input_df["planning_day"].iloc[0]
                    if "planning_day" in self.input_df.columns and not self.input_df.empty
                    else None,
                    "algo_params": {k: v for k, v in algo_params.items() if k != "master_data"},
                },
            )

            algorithm = get_algorithm(algo_name, algo_params)

            solve_result = algorithm.solve(self.input_df)
            if not isinstance(solve_result, tuple):
                raise SolverExecutionError("Algorithm did not return a tuple")
            if len(solve_result) == 2:
                results_df, algo_metrics = solve_result
                algo_artifacts = {}
            elif len(solve_result) == 3:
                results_df, algo_metrics, algo_artifacts = solve_result
                if not isinstance(algo_artifacts, dict):
                    raise SolverExecutionError("Algorithm artifacts must be a dict")
            else:
                raise SolverExecutionError("Algorithm returned an unexpected tuple shape")

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
        required_cols = {"service_id", "labor_id", "city", "schedule_date"}
        missing = required_cols - set(self.input_df.columns)
        if missing:
            raise SolverValidationError(f"Missing required input columns: {sorted(missing)}")

        if self.input_df.index.has_duplicates:
            raise SolverValidationError("Input DataFrame index has duplicates")

    # --------------------------------------------------
    # Normalize / derive fields for all algorithms
    # --------------------------------------------------

    def _prepare_data(self) -> None:
        df = self.input_df

        # Normalize datetimes (robust)
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        if "actual_start" in df.columns:
            df["actual_start"] = pd.to_datetime(df["actual_start"], errors="coerce", utc=True)
        if "actual_end" in df.columns:
            df["actual_end"] = pd.to_datetime(df["actual_end"], errors="coerce", utc=True)

        if df["schedule_date"].isna().any():
            # note: business validation should catch this earlier,
            # but we guard anyway to avoid silent failures.
            raise SolverValidationError("schedule_date contains invalid datetime values")

        # TEMP: force all schedule_date values to first row's date for testing
        first_date = df["schedule_date"].dt.date.iloc[0]
        df["schedule_date"] = df["schedule_date"].apply(
            lambda ts: ts.replace(
                year=first_date.year,
                month=first_date.month,
                day=first_date.day,
            )
            if pd.notna(ts)
            else ts
        )
        logger.warning("TEMP: schedule_date normalized to first row date for testing")

        # Enforce single planning day (optional strictness)
        # If "single date" is truly guaranteed, enforce it here.
        planning_days = df["schedule_date"].dt.date.unique()
        if len(planning_days) != 1:
            raise SolverValidationError(
                f"Expected single planning day but found {len(planning_days)}: {planning_days}"
            )

        df["planning_day"] = df["schedule_date"].dt.date.astype(str)

        # Normalize city key
        df["city"] = df["city"].astype(str).str.strip()

        self.input_df = df

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
        if not isinstance(solve_result, tuple):
            raise SolverExecutionError("Algorithm did not return a tuple")
        if len(solve_result) == 2:
            results_df, algo_metrics = solve_result
            algo_artifacts = {}
        elif len(solve_result) == 3:
            results_df, algo_metrics, algo_artifacts = solve_result
            if not isinstance(algo_artifacts, dict):
                raise SolverExecutionError("Algorithm artifacts must be a dict")
        else:
            raise SolverExecutionError("Algorithm returned an unexpected tuple shape")

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
        self.started_at = datetime.now(timezone.utc)

    def _stop_timer(self) -> None:
        self._end_time = time.perf_counter()
        self.ended_at = datetime.now(timezone.utc)

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
