import multiprocessing as mp
from time import perf_counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

import traceback

from src.algorithms.pipeline import run_single_iteration, select_best_iteration

# Assumes these are available in your codebase (import them accordingly)
# from src.algorithms.pipeline import run_single_iteration
# from src.algorithms.pipeline import select_best_iteration

def OFFLINE_algorithm_wrapper(
    assignment_state: tuple,
    pending_labors: pd.DataFrame,
    params: Dict[str, Any],
    # n_processes: Optional[int] = None,
    # show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Any]]:
    """
    OFFLINE wrapper: run many iterations (parallel) over the provided pending_labors,
    select best incumbent, and return the chosen assignments.

    Parameters
    ----------
    pending_labors : pd.DataFrame
        Pre-filtered dataframe containing the labors that should be assigned
        (may contain previously assigned values but they will be re-assigned).
    moves_state : pd.DataFrame | None
        Current moves_state (ignored by OFFLINE but included for uniform signature).
    params : dict
        Configuration parameters. Required keys used inside:
          - 'city' (optional) -- only for logging
          - 'fecha' / 'date' / 'day_str' (optional) -- only for logging
          - 'n_iterations' (int) -- number of iterations to run
          - 'run_single_iteration_args' (dict) -- base args passed to each single iteration (if your run_single_iteration expects a dict)
          - 'optimization_obj' (str) -- forwarded to select_best_iteration
          - 'tqdm_desc' (str) optional
        NOTE: exact shape of `run_single_iteration_args` must match your project's
        `run_single_iteration` expected argument dictionary (this wrapper passes each item to it).
    n_processes : int | None
        Number of worker processes for Pool. None -> os.cpu_count() (default).
    show_progress : bool
        Whether to show tqdm progress bar for the iterations.

    Returns
    -------
    (labors_df, moves_df, postponed_labors)
      - labors_df: pd.DataFrame (incumbent best results)
      - moves_df : pd.DataFrame (incumbent best moves)
      - postponed_labors: list (concatenated postponed_labors from chosen iteration)
    """

    start_all = perf_counter()

    # Basic param extraction & defaults
    n_iter = int(params.get("n_iterations", params.get("max_iter", 1)))
    optimization_obj = params.get("optimization_obj", params.get("opt_obj", "driver_distance"))
    n_processes = params.get('n_processes', None)

    base_args = params.get("base_args", {}) 

    if n_iter <= 0:
        raise ValueError("n_iterations must be >= 1")

    # Build list of arg dicts to feed run_single_iteration (one per iteration)
    iter_arg_list = []
    for i in range(1, n_iter + 1):
        arg = dict(base_args)
        arg['df_day'] = pending_labors
        arg["iter_idx"] = i
        base_seed = int(params.get("base_seed", 12345))
        arg["seed"] = int(base_seed + i)

        iter_arg_list.append(arg)

    # Prepare multiprocessing
    n_workers = n_processes if n_processes is not None else (mp.cpu_count() or 1)

    results_list = []

    # If only 1 iteration or 1 worker, run sequentially to avoid multiprocessing overhead
    if n_workers == 1 or n_iter == 1:
        for arg in iter_arg_list:
            try:
                # run_single_iteration should accept the dict (your existing code used it that way)
                r = run_single_iteration(arg)
            except Exception as e:
                print("Error in iteration:", arg.get("iter_idx"))
                print("Exception:", e)
                print("Traceback:")
                print(traceback.format_exc())
                r = {"iter": arg.get("iter_idx", None), "success": False, "error": repr(e)}
            if r is not None:
                results_list.append(r)

    else:
        # Parallel execution using Pool
        with mp.Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(run_single_iteration, iter_arg_list)

            for r in it:
                if r is not None:
                    results_list.append(r)

    # Collect successful rows
    success_rows = [r for r in results_list if r.get("success")]
    if not success_rows:
        # nothing successful
        # Produce empty structures consistent with run_assignment outputs
        empty_labors = pd.DataFrame(columns=pending_labors.columns)
        empty_moves = pd.DataFrame()
        postponed = []
        # small logging
        print(f"[offline_wrapper] WARNING: no successful iterations (n_iter={n_iter}). Returning empty result.")
        return (empty_labors, empty_moves), postponed

    # Build a DataFrame-like summary to use select_best_iteration
    # We expect each successful r to contain 'iter', 'vt_labors', 'extra_time', 'dist', 'results', 'moves'
    rows_for_df = []
    for r in success_rows:
        rows_for_df.append({
            "iter": int(r.get("iter")),
            "vt_labors": int(r.get("vt_labors", 0)),
            "extra_time": float(r.get("extra_time")) if r.get("extra_time") is not None else np.nan,
            "dist": float(r.get("dist")) if r.get("dist") is not None else np.nan,
            "results": r.get("results"),
            "moves": r.get("moves"),
            "postponed_labors": r.get("postponed_labors", [])
        })
    df_results = pd.DataFrame(rows_for_df).sort_values("iter").reset_index(drop=True)

    # Use your select_best_iteration() to choose best row
    try:
        best_idx = select_best_iteration(df_results, optimization_obj)
        # select_best_iteration might return a pandas index label; normalize to integer position
        if isinstance(best_idx, (np.integer, int)):
            # if it is an index position, ensure it maps to df_results.index
            if best_idx in df_results.index:
                chosen_pos = int(df_results.index.get_loc(best_idx))
            else:
                chosen_pos = int(best_idx)
        else:
            # if function returned a label, try to get .get_loc
            try:
                chosen_pos = int(df_results.index.get_loc(best_idx))
            except Exception:
                # final fallback: pick by objective directly
                if optimization_obj == "driver_distance":
                    chosen_pos = int(df_results["dist"].idxmin())
                else:
                    chosen_pos = int(df_results["extra_time"].idxmin())
    except Exception:
        # robust fallback
        if optimization_obj == "driver_distance":
            chosen_pos = int(df_results["dist"].idxmin())
        else:
            chosen_pos = int(df_results["extra_time"].idxmin())

    chosen_row = df_results.iloc[chosen_pos]

    # Extract final structures
    labors_df = chosen_row["results"]
    moves_df = chosen_row["moves"]
    postponed_labors = chosen_row.get("postponed_labors", [])

    # Basic sanity: if results are None or empty, fallback to empties but log
    if labors_df is None or (isinstance(labors_df, pd.DataFrame) and labors_df.empty):
        print("[offline_wrapper] WARNING: chosen iteration returned empty labors_df.")
        labors_df = pd.DataFrame(columns=pending_labors.columns)

    if moves_df is None:
        moves_df = pd.DataFrame()


    return (labors_df, moves_df), postponed_labors
