from typing import List, Dict, Any
from datetime import timedelta  
import pandas as pd

import traceback

from time import perf_counter

# adjust imports to your package layout
from src.algorithms.INSERT_algorithm import insert_single_labor, get_drivers   # your existing insertion function
from src.config.config import *
from src.config.experimentation_config import *
from src.utils.filtering import flexible_filter
from src.utils.utils import get_city_name_from_code, prep_online_algorithm_inputs, clear_last_n_lines, compute_workday_end
from src.algorithms.online_algorithms import filter_dynamic_df, get_batches_for_date  # existing helpers
from src.data.metrics import compute_metrics_with_moves, compute_iteration_metrics, concat_run_results

# -------------------------------------------------------------------------
# Worker: try to insert entire batch into base_schedule (no break / no reassign)
# -------------------------------------------------------------------------
def _insertion_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task keys:
      - base_labors_df: DataFrame (the unchanged schedule snapshot where we try to insert)
      - base_moves_df: DataFrame (same shape)
      - batch_df: DataFrame (labors to insert; may contain many services)
      - seed: int (used to shuffle service order)
      - context: minimal dict with duraciones_df, dist_dict_city, drivers, fecha, city, distance_method, assignment_type, etc.
    """
    base_labors_df = task["base_labors_df"].copy()
    base_moves_df = task["base_moves_df"].copy()
    batch_df = task["batch_df"].copy()
    seed = task["seed"]
    ctx = task["context"]

    # Shuffle service order (preserve labor_sequence within each service)
    service_ids = (
        batch_df[["service_id"]]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        ["service_id"]
        .tolist()
    )

    working_labors = base_labors_df.copy()
    working_moves = base_moves_df.copy()
    total_inserted = 0

    workday_end_dt = compute_workday_end(
            day_str=ctx['fecha'],
            workday_end_str=WORKDAY_END,
            tzinfo="America/Bogota"
        )
    new_postponed_labors = []

    # Try insert services one by one (service's labors in labor_sequence order)
    for svc in service_ids:
        svc_labors = batch_df[batch_df["service_id"] == svc].sort_values("labor_sequence")
        # temp copies for this service so we can rollback if service fails
        tmp_labors = working_labors.copy()
        tmp_moves = working_moves.copy()
        curr_end_time = None
        curr_end_pos = None
        svc_ok = True

        service_block_after_first_afterhours_nontransport = False

        for _, labor in svc_labors.iterrows():
            success, postponed, tmp_labors, tmp_moves, curr_end_time, curr_end_pos = insert_single_labor(
                labor=labor,
                labors_df=tmp_labors,
                moves_df=tmp_moves,
                curr_end_time=curr_end_time,
                curr_end_pos=curr_end_pos,
                directorio_hist_df=ctx["directorio_hist_filtered_df"],
                unassigned_services=ctx.get("unassigned_services", []),
                drivers=ctx["drivers"],
                city=ctx["city"],
                fecha=ctx["fecha"],
                distance_method=ctx["distance_method"],
                dist_dict=ctx["dist_dict_city"],
                duraciones_df=ctx["duraciones_df"],
                vehicle_transport_speed=ctx.get("vehicle_transport_speed"),
                alfred_speed=ctx.get("alfred_speed"),
                tiempo_alistar=ctx.get("tiempo_alistar"),
                tiempo_finalizacion=ctx.get("tiempo_finalizacion"),
                tiempo_gracia=ctx.get("tiempo_gracia"),
                early_buffer=ctx.get("early_buffer"),
                workday_end_dt=workday_end_dt,
                service_block_after_first_afterhours_nontransport=service_block_after_first_afterhours_nontransport,
                selection_mode="random",
                instance=ctx.get("instance"),
                update_dist_dict=False,   # PARALLEL SAFE
            )

            if postponed:
                new_postponed_labors.append({
                    "labor_row": labor,
                    "service_id": svc,
                    "reason": 'After workday hours',
                    "prev_end": curr_end_time,
                    "workday_end": workday_end_dt
                })
            
            if not success:
                svc_ok = False
                break

        if svc_ok:
            # commit successful service insertion
            working_labors = tmp_labors
            working_moves = tmp_moves
            total_inserted += len(svc_labors)
        else:
            # service couldn't be fully inserted -> rollback (do nothing)
            continue

    # compute metrics
    try:
        metrics = compute_metrics_with_moves(
            working_labors, 
            working_moves, 
            fechas=[ctx["fecha"]],
            dist_dict=ctx["dist_dict_city"], 
            workday_hours=8,
            city=ctx["city"], 
            skip_weekends=False,
            assignment_type=ctx["assignment_type"],
            dist_method=ctx["distance_method"]
        )
        vt_labors, extra_time, dist = compute_iteration_metrics(metrics)
    except Exception as e:
        print("Exception:", e)
        print("Traceback:")
        print(traceback.format_exc())
        return {"valid": False, "error": str(e), "seed": seed}

    return {
        "valid": True,
        "seed": seed,
        "num_inserted": total_inserted,
        "vt_labors": vt_labors,
        "extra_time": extra_time,
        "dist": dist,
        "results": working_labors,
        "moves": working_moves,
        'postponed_labors': new_postponed_labors
    }

# -------------------------------------------------------------------------
# Select best: prefer more inserted labors, tie-breaker min distance
# -------------------------------------------------------------------------
def _select_best_insert(results: List[Dict[str, Any]]):
    valid = [r for r in results if r and r.get("valid")]
    if not valid:
        print('No valid result')
        return None
    return sorted(valid, key=lambda r: (-r["num_inserted"], r["dist"] if r["dist"] is not None else float("inf")))[0]


def BUFFER_FIXED_wrapper(
    assignment_state: tuple,
    pending_labors: pd.DataFrame,
    params: Dict[str, Any],
):
    """
    Online INSERT-BUFFER step for orchestrator.

    INPUTS:
        assignment_state = (labors_state_city, moves_state_city)
        pending_labors   = df of incoming labors for this decision moment (filtered already)
        params           = dict that contains:
              - fecha
              - city
              - distance_method
              - assignment_type
              - duraciones_df
              - directorio_hist_df
              - dist_dict_city
              - instance
              - batch_start
              - batch_end
              - multiprocessing
              - n_iter
              - alfred speeds / buffer config...
              - etc.

    OUTPUT:
        (updated_labors_df, updated_moves_df), orchestration_log
    """
    t0 = perf_counter()

    # unpack current schedule
    labors_state_city, moves_state_city = assignment_state

    labors_state_city["latest_arrival_time"] = labors_state_city["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)
    pending_labors["latest_arrival_time"] = pending_labors["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)

    # Create orchestration log
    orch_log = {
        "algorithm": "INSERT_BUFFER",
        "city": params.get("city"),
        "fecha": params.get("fecha"),
        "batch_start": params.get("batch_start"),
        "batch_end": params.get("batch_end"),
        "n_services": pending_labors["service_id"].nunique() if not pending_labors.empty else 0,
        "success": False,
        "error": None,
        "best_iter": None,
        "best_objective": None,
        "processing_time_s": None,
    }

    # If no labors arrived, return unchanged state
    if pending_labors.empty:
        orch_log["success"] = True
        orch_log["processing_time_s"] = round(perf_counter() - t0, 4)
        return (labors_state_city, moves_state_city), orch_log

    try:
        # ------------------------------------------------------
        # Build context exactly like run_INSERT_BUFFER
        # ------------------------------------------------------
        ctx = {
            "directorio_hist_filtered_df": params["directorio_hist_filtered_df"],
            "duraciones_df": params["duraciones_df"],
            "dist_dict_city": params["dist_dict"],
            "city": params["city"],
            "fecha": params["fecha"],
            "assignment_type": params["assignment_type"],
            "distance_method": params["distance_method"],
            "drivers": get_drivers(labors_algo_df=labors_state_city, 
                                   directorio_hist_df=params["directorio_hist_filtered_df"], 
                                   city=params["city"], 
                                   fecha=params["fecha"], get_all=True),
            'vehicle_transport_speed': VEHICLE_TRANSPORT_SPEED,
            'alfred_speed': ALFRED_SPEED,
            "tiempo_alistar": TIEMPO_ALISTAR,
            "tiempo_finalizacion": TIEMPO_FINALIZACION,
            "tiempo_gracia": TIEMPO_GRACIA,
            "early_buffer": TIEMPO_PREVIO,
            "instance": params.get("instance")
        }

        # ------------------------------------------------------
        # Build tasks exactly like run_INSERT_BUFFER
        # ------------------------------------------------------
        n_iter = params.get("n_iterations", 20)
        multiprocessing_flag = params.get("multiprocessing", False)
        n_proc = params.get("n_processes", None)

        tasks = []
        for seed in range(1, n_iter + 1):
            tasks.append({
                "base_labors_df": labors_state_city,
                "base_moves_df": moves_state_city,
                "batch_df": pending_labors,
                "seed": seed,
                "context": ctx
            })

        # ------------------------------------------------------
        # Run insertion iterations
        # ------------------------------------------------------
        results = []
        if multiprocessing_flag:
            from multiprocessing import Pool, cpu_count
            n_workers = n_proc or max(cpu_count() - 1, 1)
            with Pool(processes=n_workers) as pool:
                for res in pool.imap_unordered(_insertion_worker, tasks):
                    results.append(res)
        else:
            for t in tasks:
                results.append(_insertion_worker(t))

        # ------------------------------------------------------
        # Select best iteration
        # ------------------------------------------------------
        best = _select_best_insert(results)
        if best is None:
            print('No best iteration')
            orch_log["error"] = "No valid insertion iteration"
            orch_log["processing_time_s"] = round(perf_counter() - t0, 4)
            return (labors_state_city, moves_state_city), orch_log

        # ------------------------------------------------------
        # Commit best result
        # ------------------------------------------------------
        updated_labors = best["results"]
        updated_moves = best["moves"]

        orch_log["success"] = True
        orch_log["best_iter"] = best["seed"]
        orch_log["best_objective"] = best["dist"]
        orch_log["num_inserted"] = best["num_inserted"]
        orch_log["processing_time_s"] = round(perf_counter() - t0, 4)

        return (updated_labors, updated_moves), orch_log

    except Exception as e:
        orch_log["error"] = str(e)
        orch_log["traceback"] = traceback.format_exc()
        orch_log["processing_time_s"] = round(perf_counter() - t0, 4)
        print("Exception:", e)
        print("Traceback:")
        print(traceback.format_exc())
        return (labors_state_city, moves_state_city), orch_log
