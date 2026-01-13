from time import perf_counter


def should_update_incumbent(
    optimization_variable,
    iter_dist,
    iter_extra_time,
    inc_dist,
    inc_extra_time,
    dist_norm_factor=None,
    extra_time_norm_factor=None
):
    """
    Decide whether to update the incumbent solution based on the optimization objective.
    """
    update = False
    new_score, inc_score = None, None

    if optimization_variable == "driver_distance":
        update = iter_dist < inc_dist

    elif optimization_variable == "driver_extra_time":
        update = iter_extra_time < inc_extra_time or inc_extra_time == 0

    elif optimization_variable == "hybrid":
        if dist_norm_factor is None or extra_time_norm_factor is None:
            raise ValueError("Normalization factors must be provided for hybrid optimization.")

        new_score = (iter_dist / dist_norm_factor) + (iter_extra_time / extra_time_norm_factor)
        inc_score = (inc_dist / dist_norm_factor) + (inc_extra_time / extra_time_norm_factor)

        update = new_score < inc_score

    else:
        raise ValueError(f"Unknown optimization_variable: {optimization_variable}")

    return update, new_score, inc_score


def update_incumbent_state(
    iter_idx,
    iter_vt_labors,
    iter_extra_time,
    iter_dist,
    results_df,
    moves_df,
    metrics,
):
    """
    Update the incumbent state with values from the current iteration.

    Parameters
    ----------
    iter_idx : int
        Current iteration index.
    iter_vt_labors : int
        Number of VT labors in the current iteration.
    iter_extra_time : float
        Extra time in the current iteration.
    iter_dist : float
        Distance in the current iteration.
    results_df : pd.DataFrame
        Results dataframe for the current iteration.
    moves_df : pd.DataFrame
        Moves dataframe for the current iteration.
    metrics : dict
        Computed metrics for the current iteration.
    start_time : float
        Timer reference from perf_counter() at the start of optimization.

    Returns
    -------
    inc_state : dict
        Dictionary holding the updated incumbent state:
        - vt_labors
        - extra_time
        - dist
        - values (tuple: iter, extra_time, dist, duration)
        - results
        - moves
        - metrics
    """

    inc_state = {
        "vt_labors": iter_vt_labors,
        "extra_time": iter_extra_time,
        "dist": iter_dist,
        "values": (iter_idx, iter_extra_time, iter_dist),
        "results": results_df,
        "moves": moves_df,
        "metrics": metrics,
    }

    return inc_state

