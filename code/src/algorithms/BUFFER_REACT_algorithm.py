import time
from time import perf_counter
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from datetime import timedelta

from src.algorithms.BUFFER_REACT_logic import (
    # run_single_iteration_worker,
    process_iterations_for_batch,
    filter_dfs_for_REACT,
)
from src.algorithms.online_algorithms import attach_service_batch_to_reassign
# Required helpers (assumed available in your codebase)
# from src.algorithms.REACT_BUFFER_algorithm import filter_dfs_for_REACT, attach_service_batch_to_reassign, prep_dfs_for_REACT
# from src.algorithms.REACT_BUFFER_run import process_iterations_for_batch  # or wherever you placed it
# from src.utils.filtering import flexible_filter
# from src.utils.utils import get_city_name_from_code, clear_last_n_lines

def BUFFER_REACT_wrapper(
    assignment_state: Tuple[pd.DataFrame, pd.DataFrame],
    pending_labors: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
    """
    Wrapper for the BUFFER_REACT algorithm that processes a single batch (pending_labors).

    Parameters
    ----------
    assignment_state : (labors_state_df, moves_state_df)
        Current assignment state (skeleton). labors_state_df contains the schedule currently active.
    pending_labors : pd.DataFrame
        The batch of labors to be considered in this run (services to attach / reassign).
    params : dict
        Parameters required by the wrapper. Required keys (recommended):
            - city (str)
            - fecha (str in 'YYYY-MM-DD')
            - duraciones_df (DataFrame)
            - dist_dict_city (dict)         # distances cache for this city (may be {})
            - global_dist_dict (dict)       # global dist dict used for metrics
            - directorio_hist_filtered_df (DataFrame)
            - alpha (float)
            - optimization_obj (str)
            - distance_method (str)
            - assignment_type (str)
            - driver_init_mode (str)
            - time_previous_freeze (int)    # minutes
            - n_iterations (int)
            - iterations_nums_city (list)   # optional, default []
            - multiprocessing (bool)        # optional, default True
            - n_processes (int | None)      # optional
            - base_seed (int)               # optional, default 12345
        Any other params required internally may be passed through.

    Returns
    -------
    (assignment_state_out, orchestration_log)
        assignment_state_out: tuple (labors_df, moves_df)
        orchestration_log: dict with run metadata (success, times, best_iter, etc.)
    """
    # Basic init & validate
    log = {
        "success": False,
        "error": None,
        "city": params.get("city"),
        "fecha": params.get("fecha"),
        "n_services_in_batch": int(pending_labors["service_id"].nunique()) if pending_labors is not None and not pending_labors.empty else 0,
        "best_iter": None,
        "best_objective": None,
        "processing_time_s": None,
        "postponed_labors_count": 0,
        "notes": None
    }

    # Quick returns
    if pending_labors is None or pending_labors.empty:
        log["success"] = True
        log["notes"] = "Empty batch: nothing to process"
        return assignment_state, log

    # Unpack assignment state (labors skeleton + moves skeleton)
    try:
        labors_state_base, moves_state_base = assignment_state
        if labors_state_base is None:
            # Ensure we have empty DataFrames if None provided
            labors_state_base = pd.DataFrame()
        if moves_state_base is None:
            moves_state_base = pd.DataFrame()
    except Exception as e:
        log["error"] = f"Invalid assignment_state: {repr(e)}"
        return assignment_state, log

    # Extract required params with defaults
    city = params.get("city")
    fecha = params.get("fecha")
    duraciones_df = params.get("duraciones_df")
    dist_dict_city = params.get("dist_dict_city", {})
    global_dist_dict = params.get("global_dist_dict", {})
    directorio_hist_filtered_df = params.get("directorio_hist_filtered_df")
    alpha = params.get("alpha", 1.0)
    optimization_obj = params.get("optimization_obj", "driver_distance")
    distance_method = params.get("distance_method", "haversine")
    assignment_type = params.get("assignment_type", "algorithm")
    driver_init_mode = params.get("driver_init_mode", None)
    time_previous_freeze = int(params.get("time_previous_freeze", 10))
    n_iterations = int(params.get("n_iterations", params.get("max_iter", 10)))
    iterations_nums_city = params.get("iterations_nums_city", [])
    multiprocessing_flag = bool(params.get("multiprocessing", True))
    n_processes = params.get("n_processes", None)
    base_seed = int(params.get("base_seed", 12345))

    # Validate essential params
    if city is None or fecha is None:
        log["error"] = "Missing required params: 'city' or 'fecha'"
        return assignment_state, log

    # Build freeze cutoff (the run expects a decision time; here we treat decision_time as the created_at of last element in batch)
    # In your run_REACT_BUFFER the decision_time is batch_end (passed externally). We assume pending_labors were selected for a given decision_time.
    # process_iterations_for_batch expects labors_reassign_df (with frozen logic pre-applied)
    start = perf_counter()
    try:
        # The wrapper expects that labors_state_base is the working final schedule for the city/date.
        # We will apply filter_dfs_for_REACT to compute: labors_reassign_df, labors_frozen_df, moves_frozen_df, directorio_online_df
        # For this we need the function filter_dfs_for_REACT which uses freeze_cutoff = decision_time - time_previous_freeze
        # However `pending_labors` does not carry decision_time directly; we expect caller to have passed `decision_time` in params optionally.
        decision_time = params.get("decision_time")  # should be a pandas.Timestamp or string parseable
        if decision_time is None:
            # Fallback: compute decision time as max(created_at) in pending_labors
            if "created_at" in pending_labors.columns:
                decision_time = pd.to_datetime(pending_labors["created_at"]).max()
            else:
                # As last resort, use fecha at midnight + 23:59
                decision_time = pd.to_datetime(fecha + " 23:59:59")
        else:
            decision_time = pd.to_datetime(decision_time)

        freeze_cutoff = pd.to_datetime(decision_time) - timedelta(minutes=time_previous_freeze)

        # Call filter_dfs_for_REACT to extract reassign/frozen sets.
        # NOTE: filter_dfs_for_REACT signature in your code:
        #   filter_dfs_for_REACT(labors_df, moves_df, directorio_df, freeze_cutoff) -> labors_reassign_df, labors_frozen_df, moves_frozen_df, directorio_online_df
        labors_reassign_df, labors_frozen_df, moves_frozen_df, directorio_online_df = filter_dfs_for_REACT(
            labors_df=labors_state_base,
            moves_df=moves_state_base,
            directorio_df=directorio_hist_filtered_df,
            freeze_cutoff=freeze_cutoff
        )

        # Attach the incoming batch (services) to the reassign set — preserves full services
        labors_reassign_iter_df = attach_service_batch_to_reassign(labors_reassign_df, pending_labors)

        # Prepare parameters for the internal iteration runner
        max_iter_city = n_iterations
        iters_nums = iterations_nums_city

        # Call the iteration processing function (it manages multiprocessing internally)
        inc_state, traces = process_iterations_for_batch(
            labors_reassign_df=labors_reassign_iter_df,
            directorio_online_df=directorio_online_df,
            duraciones_df=duraciones_df,
            dist_dict_city=dist_dict_city,
            global_dist_dict=global_dist_dict,
            fecha=fecha,
            city=city,
            alpha=alpha,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            assignment_type=assignment_type,
            driver_init_mode=driver_init_mode,
            time_previous_freeze=time_previous_freeze,
            max_iter=max_iter_city,
            iterations_nums_city=iters_nums,
            multiprocessing=multiprocessing_flag,
            n_processes=n_processes,
            base_seed=base_seed
        )
        elapsed = round(perf_counter() - start, 2)
        log["processing_time_s"] = elapsed

        # If no valid incumbent was found, return previous assignment_state unchanged
        if inc_state is None:
            log.update({
                "success": False,
                "error": "No successful iterations (inc_state is None)",
                "notes": "No incumbent found for this batch",
                "traces_count": len(traces) if traces is not None else 0
            })
            return assignment_state, log

        # Compose updated final assignment state:
        # final_labors = frozen_past + incumbent results (which already contains reassigned & future labors)
        try:
            final_labors = pd.concat([labors_frozen_df, inc_state["results"]], ignore_index=True).reset_index(drop=True)
        except Exception:
            # if labors_frozen_df empty, inc_state['results'] is the final
            final_labors = inc_state["results"].copy().reset_index(drop=True)

        try:
            final_moves = pd.concat([moves_frozen_df, inc_state["moves"]], ignore_index=True).reset_index(drop=True)
        except Exception:
            final_moves = inc_state["moves"].copy().reset_index(drop=True)

        # Build orchestration summary
        log.update({
            "success": True,
            "best_iter": int(inc_state.get("iter")) if inc_state.get("iter") is not None else None,
            "best_objective": float(inc_state.get("dist") if optimization_obj == "driver_distance" else inc_state.get("extra_time", inc_state.get("vt_labors", None))),
            "postponed_labors_count": len(inc_state.get("postponed_labors", [])) if inc_state.get("postponed_labors") is not None else 0,
            "traces_count": len(traces) if traces is not None else 0,
            "metrics": inc_state.get("metrics"),
        })

        assignment_state_out = (final_labors, final_moves)
        return assignment_state_out, log

    except Exception as e:
        log["success"] = False
        log["error"] = repr(e)
        return assignment_state, log
