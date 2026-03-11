import pandas as pd
import numpy as np

import os
import pickle

from time import perf_counter
from datetime import datetime, timedelta, time as dt_time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any

import random
from tqdm import tqdm

from functools import partial

from src.algorithms.offline_algorithms import run_assignment_algorithm
from src.algorithms.online_algorithms import (
    filter_dynamic_df,
    get_batches_for_date,
    attach_service_batch_to_reassign
)
from src.algorithms.REACT_BUFFER_algorithm import (
    filter_dfs_for_REACT,
    compute_disruption_factor,
    rename_old_solution_columns,
    prep_dfs_for_REACT
)


from src.data.metrics import compute_metrics_with_moves, compute_iteration_metrics, concat_run_results
from src.config.experimentation_config import *
from src.config.config import *
from src.utils.filtering import flexible_filter
from src.utils.utils import get_city_name_from_code, clear_last_n_lines, prep_online_algorithm_inputs
from src.algorithms.solution_search_utils import should_update_incumbent, update_incumbent_state
from src.algorithms.pipeline import select_best_iteration

# -------------------------------------------------------------------
# Worker that runs one iteration (must be top-level for multiprocessing)
# -------------------------------------------------------------------
def run_single_iteration_worker(
    iter_idx: int,
    labors_reassign_df: pd.DataFrame,
    directorio_online_df: pd.DataFrame,
    duraciones_df: pd.DataFrame,
    dist_dict_city: dict,
    global_dist_dict: dict,
    fecha: str,
    city: str,
    alpha: float,
    optimization_obj: str,
    distance_method: str,
    assignment_type: str,
    driver_init_mode: str,
    seed: int,
):
    """
    Runs one iteration of the assignment algorithm (worker).
    Returns dict or None on failure.
    """
    # set per-iteration seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    try:
        results_df, moves_df, postponed_labors = run_assignment_algorithm(
            df_cleaned_template=labors_reassign_df,
            directorio_df=directorio_online_df,
            duraciones_df=duraciones_df,
            day_str=fecha,
            ciudad=get_city_name_from_code(city),
            dist_method=distance_method,
            dist_dict=dist_dict_city,
            assignment_type=assignment_type,
            alpha=alpha,
            instance=None,
            driver_init_mode=driver_init_mode,
            update_dist_dict=False,  # workers must not mutate global dict
        )
    except Exception as e:
        # Worker failed; return failure info (kept minimal and informative)
        return {
            "iter": iter_idx,
            "success": False,
            "error": repr(e)
        }

    if results_df is None or moves_df is None or moves_df.empty:
        return {
            "iter": iter_idx,
            "success": False,
            "error": "empty_results"
        }

    # compute metrics and scalar objectives (cheap)
    try:
        metrics_dict = compute_metrics_with_moves(
            results_df,
            moves_df,
            fechas=[fecha],
            dist_dict=global_dist_dict,
            workday_hours=8,
            city=city,
            skip_weekends=False,
            assignment_type=assignment_type,
            dist_method=distance_method,
        )
        vt_labors, extra_time, dist = compute_iteration_metrics(metrics_dict)
    except Exception as e:
        return {
            "iter": iter_idx,
            "success": False,
            "error": f"metrics_failed: {repr(e)}"
        }

    return {
        "iter": iter_idx,
        "success": True,
        "vt_labors": vt_labors,
        "extra_time": extra_time,
        "dist": dist,
        "metrics": metrics_dict,
        "results": results_df,
        "moves": moves_df,
        'postponed_labors': postponed_labors
    }


def call_worker_wrapper(args):
    iter_idx, seed, worker_partial = args
    return worker_partial(iter_idx=iter_idx, seed=seed)

# -------------------------------------------------------------------
# Parallel-aware process_iterations_for_batch
# -------------------------------------------------------------------
def process_iterations_for_batch(
    labors_reassign_df: pd.DataFrame,
    directorio_online_df: pd.DataFrame,
    duraciones_df: pd.DataFrame,
    dist_dict_city: dict,
    global_dist_dict: dict,
    fecha: str,
    city: str,
    alpha: float,
    optimization_obj: str,
    distance_method: str,
    assignment_type: str,
    driver_init_mode: str,
    time_previous_freeze: int,
    max_iter: int,
    iterations_nums_city: List[int],
    multiprocessing: bool = True,
    n_processes: int = None,
    base_seed: int = 12345
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Parallelized version:
    - Launches max_iter workers (or runs sequentially if n_processes in {None,1}).
    - Collects results and picks incumbent using select_best_iteration.
    - Returns (inc_state or None, traces list).
    """
    start_time = perf_counter()
    traces = []

    # Prepare seeds per iteration
    seeds = [base_seed + i for i in range(1, max_iter + 1)]

    # Prepare worker partial with constant args (so Pool only gets iter_idx and seed)
    worker_partial = partial(
        run_single_iteration_worker,
        labors_reassign_df=labors_reassign_df,
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
    )

    results_list = []

    # Decide parallel vs sequential
    if not multiprocessing:
        # sequential run (identical semantics)
        for i, seed in enumerate(seeds, start=1):
            r = worker_partial(iter_idx=i, seed=seed)
            if r is not None:
                results_list.append(r)
    else:
        tasks = [(i, seeds[i - 1], worker_partial) for i in range(1, max_iter + 1)]

        with Pool(processes=n_processes) as pool:
            for r in pool.imap_unordered(call_worker_wrapper, tasks):
                if r is not None:
                    results_list.append(r)

    # Filter successful results and construct dataframe-like structure
    success_rows = [r for r in results_list if r.get("success")]
    if not success_rows:
        # no successful iteration
        return None, traces

    # Build results DataFrame-like list
    rows_for_df = []
    for r in success_rows:
        rows_for_df.append({
            "iter": r["iter"],
            "vt_labors": r["vt_labors"],
            "extra_time": r["extra_time"],
            "dist": r["dist"],
            "results": r["results"],
            "moves": r["moves"],
            "metrics": r["metrics"],
            'postponed_labors': r['postponed_labors']
        })

    df_results = pd.DataFrame(rows_for_df).sort_values("iter").reset_index(drop=True)

    # Use your select_best_iteration to pick the index (returns index label => convert to position)
    try:
        # select_best_iteration expects df with 'dist' and 'extra_time' columns
        best_loc = select_best_iteration(df_results, optimization_obj)
        # if select_best_iteration returns an index (pandas idx), convert to integer location
        if isinstance(best_loc, (np.integer, int)):
            best_pos = int(df_results.index.get_loc(best_loc)) if best_loc in df_results.index else int(best_loc)
        else:
            # fallback: pick best by objective
            if optimization_obj == "driver_distance":
                best_pos = int(df_results["dist"].idxmin())
            else:
                best_pos = int(df_results["extra_time"].idxmin())
    except Exception:
        # robust fallback
        if optimization_obj == "driver_distance":
            best_pos = int(df_results["dist"].idxmin())
        else:
            best_pos = int(df_results["extra_time"].idxmin())

    # retrieve the chosen row (incumbent)
    chosen = df_results.iloc[best_pos]
    inc_state = {
        "iter": int(chosen["iter"]),
        "vt_labors": int(chosen["vt_labors"]),
        "extra_time": float(chosen["extra_time"]),
        "dist": float(chosen["dist"]),
        "results": chosen["results"],
        "moves": chosen["moves"],
        "metrics": chosen["metrics"],
        'postponed_labors': chosen['postponed_labors']
    }

    # Build iteration checkpoint traces for iterations in iterations_nums_city.
    # For each trace point, find the best incumbent among successful_iters <= checkpoint
    if iterations_nums_city:
        # Sort successful rows by iter
        success_sorted = sorted(success_rows, key=lambda x: x["iter"])
        # iterate checkpoints
        for cp in iterations_nums_city:
            # find successful rows with iter <= cp
            eligible = [r for r in success_sorted if r["iter"] <= cp]
            if not eligible:
                traces.append({
                    "city": city,
                    "date": fecha,
                    "alpha": alpha,
                    "iter": cp,
                    "vt_labors": 0,
                    "extra_time": None,
                    "dist": None,
                    "successful_iters": 0,
                    "duration_s": round(perf_counter() - start_time, 1)
                })
                continue

            # build temporary df to pick best among eligible
            tmp_rows = [{
                "iter": r["iter"], "vt_labors": r["vt_labors"],
                "extra_time": r["extra_time"], "dist": r["dist"]
            } for r in eligible]
            tmp_df = pd.DataFrame(tmp_rows)
            # pick best
            try:
                pick_idx = select_best_iteration(tmp_df, optimization_obj)
                if isinstance(pick_idx, (np.integer, int)):
                    pick_pos = int(pick_idx)
                else:
                    pick_pos = int(tmp_df.index.get_loc(pick_idx)) if pick_idx in tmp_df.index else int(pick_idx)
            except Exception:
                pick_pos = int(tmp_df["dist"].idxmin()) if optimization_obj == "driver_distance" else int(tmp_df["extra_time"].idxmin())
            pick_row = tmp_df.iloc[pick_pos]
            traces.append({
                "city": city,
                "date": fecha,
                "alpha": alpha,
                "iter": cp,
                "vt_labors": int(pick_row["vt_labors"]),
                "extra_time": float(pick_row["extra_time"]) if pd.notna(pick_row["extra_time"]) else None,
                "dist": float(pick_row["dist"]) if pd.notna(pick_row["dist"]) else None,
                "successful_iters": len(eligible),
                "duration_s": round(perf_counter() - start_time, 1)
            })

    return inc_state, traces

def run_REACT_BUFFER(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    time_previous_freeze: int,
    batch_interval_minutes: int = 30,
    start_of_day_str: str = "07:00",
    save_results: bool = True,
    multiprocessing: bool = True,
    n_processes: int = None,
    experiment_type: str = 'online_operation'
):
    """
    Run REACT_BUFFER (fixed: avoid duplicate rows by appending only final snapshot per city/date).
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
        experiment_type=experiment_type)

    labors_algo_dynamic_df = rename_old_solution_columns(labors_algo_dynamic_df)

    run_results = []
    all_batch_metrics = []
    all_traces = []

    new_postponed_labors = []

    for fecha in fechas:
        print(f"{'-'*120}\n▶ Processing date: {fecha} / {fechas[-1]}")

        for city in valid_cities:

            directorio_hist_filtered_df = flexible_filter(directorio_hist_df, city=city, date=fecha)

            dist_dict_city = global_dist_dict.get(city, {})

            labors_dynamic_filtered_df, labors_algo_dynamic_filt_df, moves_algo_dynamic_filt_df = prep_dfs_for_REACT(
                labors_dynamic_df=labors_dynamic_df,
                labors_algo_df=labors_algo_dynamic_df,
                moves_algo_df=moves_algo_dynamic_df,
                city=city,
                fecha=fecha
            )

            # If no dynamic labors for this city/date -> append the unchanged working copy once
            if labors_dynamic_filtered_df.empty:
                run_results.append([labors_algo_dynamic_filt_df, moves_algo_dynamic_filt_df])
                continue

            # Build batches
            batches = get_batches_for_date(
                labors_df_date=labors_dynamic_filtered_df,
                batch_interval_minutes=batch_interval_minutes,
                start_of_day_str=start_of_day_str,
                end_of_day=None
            )

            # Keep a final working solution for this city/date (update as we process batches).
            final_labors_for_city_date = labors_algo_dynamic_filt_df.copy()
            final_moves_for_city_date = moves_algo_dynamic_filt_df.copy()

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
                n_services = batch_df['service_id'].nunique() if not batch_df.empty else 0

                # Decision time is batch_end (for REACT batch_end == created_at)
                decision_time = batch_end

                if batch_df.empty:
                    # no services in this batch — keep frozen state (but DO NOT append to run_results here)
                    all_batch_metrics.append({
                        "city": city,
                        "date": fecha,
                        "batch_id": batch_id,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "decision_time": decision_time,
                        "n_services": 0,
                        "best_iter": None,
                        "best_objective": None,
                        "processing_time_s": 0,
                    })
                    # still continue to next batch (final_* remain unchanged)
                    continue
                
                freeze_cutoff = pd.to_datetime(decision_time) - timedelta(minutes=time_previous_freeze)

                labors_reassign_df, labors_frozen_df, moves_frozen_df, directorio_online_df = filter_dfs_for_REACT(
                    labors_df=final_labors_for_city_date,
                    moves_df=final_moves_for_city_date,
                    directorio_df=directorio_hist_filtered_df,
                    freeze_cutoff=freeze_cutoff
                )

                # attach batch services to reassign set
                labors_reassign_iter_df = attach_service_batch_to_reassign(labors_reassign_df, batch_df)

                # prepare iteration parameters
                max_iter_city = int(max_iterations.get(city, max(max_iterations.values())))
                iterations_nums_city = iterations_nums.get(city, [])

                # run iterations for this batch (sequential worker)
                iter_start = perf_counter()
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
                    iterations_nums_city=iterations_nums_city,
                    multiprocessing=multiprocessing,
                    n_processes=n_processes
                )
                iter_time = round(perf_counter() - iter_start, 1)

                # collect traces always (even if empty)
                all_traces.extend(traces)

                if inc_state is None:
                    all_batch_metrics.append({
                        "city": city,
                        "date": fecha,
                        "batch_id": batch_id,
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "decision_time": decision_time,
                        "n_services": n_services,
                        "best_iter": None,
                        "best_objective": None,
                        "processing_time_s": iter_time,
                    })
                    continue

                # Update the final working solution for downstream batches:
                final_labors_for_city_date = pd.concat([labors_frozen_df, inc_state['results']], ignore_index=True).reset_index(drop=True)
                final_moves_for_city_date = pd.concat([moves_frozen_df, inc_state['moves']], ignore_index=True).reset_index(drop=True)
                new_postponed_labors += inc_state['postponed_labors']

                # record batch metrics
                best_obj = inc_state['dist'] if optimization_obj == 'driver_distance' else (
                    inc_state['extra_time'] if optimization_obj == 'driver_extra_time' else inc_state['vt_labors']
                )

                all_batch_metrics.append({
                    "city": city,
                    "date": fecha,
                    "batch_id": batch_id,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "decision_time": decision_time,
                    "n_services": n_services,
                    "best_iter": inc_state['iter'],
                    "best_objective": best_obj,
                    "processing_time_s": iter_time,
                })

            # END batches loop for this city/date
            # Append a single final snapshot for this city/date (avoids duplicates)
            run_results.append([final_labors_for_city_date, final_moves_for_city_date])
        
        clear_last_n_lines(2)

    # final consolidation
    results_df, moves_df = concat_run_results(run_results)
    postponed_labors += new_postponed_labors

    if batch_interval_minutes == 0:
        algorithm = 'REACT'
    else:
        algorithm = 'BUFFER_REACT'

    if save_results:
        output_dir = os.path.join(data_path, "resultados", experiment_type, instance, distance_method)
        os.makedirs(output_dir, exist_ok=True)

        extra_output_dir = os.path.join(output_dir, 'extra_info')
        os.makedirs(extra_output_dir, exist_ok=True)  # Creates folder if missing

        with open(os.path.join(output_dir, f"res_algo_{algorithm}.pkl"), "wb") as f:
            pickle.dump([results_df, moves_df, postponed_labors], f)

        # save batch metrics and traces
        batch_metrics_df = pd.DataFrame(all_batch_metrics)
        batch_metrics_df.to_csv(os.path.join(extra_output_dir, f"{algorithm}_batch_metrics.csv"), index=False)

        trace_df = pd.DataFrame(all_traces)
        if not trace_df.empty:
            trace_df.to_csv(os.path.join(extra_output_dir, f"{algorithm}_traces.csv"), index=False)

    print(f"\n ✅ Completed {algorithm} algorithm in {round(perf_counter()-global_start,1)}s total.")
    
    return results_df, moves_df, pd.DataFrame(all_batch_metrics), pd.DataFrame(all_traces)
