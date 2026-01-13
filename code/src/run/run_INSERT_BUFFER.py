# src/algorithms/INSERTION_BUFFER_fixed.py
import os
import pickle
from time import perf_counter
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

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
            working_labors, working_moves, 
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
        return None
    return sorted(valid, key=lambda r: (-r["num_inserted"], r["dist"] if r["dist"] is not None else float("inf")))[0]

# -------------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------------
def run_INSERT_BUFFER(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    batch_interval_minutes: int = 30,
    start_of_day_str: str = "07:00",
    save_results: bool = True,
    multiprocessing: bool = True,
    n_processes: int = None,
    experiment_type: str = 'online_operation'
):
    """
    Insertion buffer: batches incoming orders, then tries randomized insertion iterations for each batch.
    No breaking of schedule, no reassign of past labors.
    """
    global_start = perf_counter()

    (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df,
        directorio_hist_df,
        global_dist_dict,
        fechas,
        alpha,
        labors_dynamic_df,
        labors_algo_dynamic_df,
        moves_algo_dynamic_df,
        postponed_labors
    ) = prep_online_algorithm_inputs(
        instance, 
        distance_method, 
        optimization_obj,
        experiment_type=experiment_type
    )
    
    labors_dynamic_df["latest_arrival_time"] = labors_dynamic_df["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)

    run_results = []
    all_batch_metrics = []
    all_traces = []

    new_postponed_labors = []

    # num workers
    n_workers = (n_processes or max(cpu_count() - 1, 1)) if multiprocessing else 1

    for fecha in fechas:
        print(f"{'-'*120}\n▶ Processing date: {fecha} / {fechas[-1]}")

        for city in valid_cities:

            # working snapshot for the day/city schedule
            labors_algo_day = flexible_filter(labors_algo_dynamic_df, city=city, schedule_date=fecha).copy().reset_index(drop=True)
            moves_algo_day = flexible_filter(moves_algo_dynamic_df, city=city, schedule_date=fecha).copy().reset_index(drop=True)

            # incoming dynamic labors for city/day
            labors_dynamic_filtered_df = filter_dynamic_df(labors_dynamic_df=labors_dynamic_df, city=city, fecha=fecha)
            if labors_dynamic_filtered_df.empty:
                run_results.append([labors_algo_day, moves_algo_day])
                continue

            # batches
            batches = get_batches_for_date(labors_dynamic_filtered_df, batch_interval_minutes, start_of_day_str)

            dist_dict_city = global_dist_dict.get(city, {})

            final_labors = labors_algo_day.copy()
            final_moves = moves_algo_day.copy()

            batch_id = 0
                
            for batch_idx, (batch_start, batch_end, batch_df) in enumerate(
                tqdm(
                    batches, 
                    desc=f"City: {city}", 
                    unit="batch",
                    leave=False
                )
            ):
                batch_id += 1
                n_services = batch_df["service_id"].nunique() if not batch_df.empty else 0
                if batch_df.empty:
                    all_batch_metrics.append({
                        "city": city, "date": fecha, "batch_id": batch_id,
                        "batch_start": batch_start, "batch_end": batch_end,
                        "decision_time": batch_end, "n_services": 0,
                        "best_iter": None, "best_objective": None, "processing_time_s": 0
                    })
                    continue

                # prepare context for workers (minimal)
                context = {
                    "directorio_hist_filtered_df": flexible_filter(directorio_hist_df, city=city, date=fecha),
                    "duraciones_df": duraciones_df,
                    "dist_dict_city": dist_dict_city,
                    "city": city,
                    "fecha": fecha,
                    "assignment_type": assignment_type,
                    "distance_method": distance_method,
                    "drivers": get_drivers(labors_algo_df=final_labors, directorio_hist_df=directorio_hist_df, city=city, fecha=fecha, get_all=True),
                    'vehicle_transport_speed': VEHICLE_TRANSPORT_SPEED,
                    'alfred_speed': ALFRED_SPEED,
                    "tiempo_alistar": TIEMPO_ALISTAR,
                    "tiempo_finalizacion": TIEMPO_FINALIZACION,
                    "tiempo_gracia": TIEMPO_GRACIA,
                    "early_buffer": TIEMPO_PREVIO,
                    "instance": instance
                }

                # Build tasks: each task is trying a different randomized insertion ordering (seed)
                max_iter_city = int(max_iterations.get(city, max(max_iterations.values())))
                tasks = []
                for seed in range(1, max_iter_city + 1):
                    tasks.append({
                        "base_labors_df": final_labors,
                        "base_moves_df": final_moves,
                        "batch_df": batch_df,
                        "seed": seed,
                        "context": context
                    })

                # run workers
                iter_start = perf_counter()
                iteration_results = []
                if multiprocessing:
                    with Pool(processes=n_workers) as pool:
                        for res in pool.imap_unordered(_insertion_worker, tasks):
                            if res:
                                iteration_results.append(res)
                else:
                    for t in tasks:
                        iteration_results.append(_insertion_worker(t))
                iter_time = round(perf_counter() - iter_start, 1)

                # choose best
                best = _select_best_insert(iteration_results)
                # store traces
                for r in iteration_results:
                    all_traces.append({
                        "city": city, "date": fecha, "batch_id": batch_id,
                        "seed": r.get("seed"),
                        "num_inserted": r.get("num_inserted"),
                        "vt_labors": r.get("vt_labors"),
                        "extra_time": r.get("extra_time"),
                        "dist": r.get("dist"),
                        "valid": r.get("valid", False)
                    })

                if best is None:
                    all_batch_metrics.append({
                        "city": city, "date": fecha, "batch_id": batch_id,
                        "batch_start": batch_start, "batch_end": batch_end,
                        "decision_time": batch_end, "n_services": n_services,
                        "best_iter": None, "best_objective": None, "processing_time_s": iter_time
                    })
                    continue

                # commit best into final schedule (frozen past + new inserted labors)
                # final_labors = pd.concat([final_labors, best["results"]]).reset_index(drop=True)
                # final_moves = pd.concat([final_moves, best["moves"]]).reset_index(drop=True)

                final_labors = best['results']
                final_moves = best['moves']

                new_postponed_labors += best['postponed_labors']

                all_batch_metrics.append({
                    "city": city, "date": fecha, "batch_id": batch_id,
                    "batch_start": batch_start, "batch_end": batch_end,
                    "decision_time": batch_end, "n_services": n_services,
                    "best_iter": best["seed"], "best_objective": best["dist"],
                    "num_inserted": best["num_inserted"], "processing_time_s": iter_time
                })

            # end batches for city/date: append single final snapshot
            run_results.append([final_labors, final_moves])

        clear_last_n_lines(2)

    postponed_labors += new_postponed_labors

    # consolidate and save
    results_df, moves_df = concat_run_results(run_results)
    if save_results:
        output_dir = os.path.join(data_path, "resultados", experiment_type, instance, distance_method)
        os.makedirs(output_dir, exist_ok=True)

        extra_output_dir = os.path.join(output_dir, 'extra_info')
        os.makedirs(extra_output_dir, exist_ok=True)  # Creates folder if missing

        with open(os.path.join(output_dir, "res_algo_BUFFER_FIXED.pkl"), "wb") as f:
            pickle.dump([results_df, moves_df, postponed_labors], f)

        pd.DataFrame(all_batch_metrics).to_csv(os.path.join(extra_output_dir, "BUFFER_FIXED_batch_metrics.csv"), index=False)
        pd.DataFrame(all_traces).to_csv(os.path.join(extra_output_dir, "BUFFER_FIXED_traces.csv"), index=False)

    print(f"\n ✅ INSERT_BUFFER finished in {round(perf_counter() - global_start, 1)}s\n")
    return True
