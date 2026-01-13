import pandas as pd
import pickle
import os

from time import perf_counter

from .metrics import collect_results_to_df, compute_metrics_with_moves 
from .filtering import flexible_filter

# Parámetros de fecha    
fechas = pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist()
# fechas = pd.date_range("2025-07-21", "2025-07-27").strftime("%Y-%m-%d").tolist()

### Iteration parameters
alphas = [i/10 for i in range(0,11)]

metrics = ['hybrid', 'driver_distance', 'driver_extra_time']

dist_norm_factor = 5497             # normalización para driver distance en híbrido
extra_time_norm_factor = 12469      # normalización para tiempo extra en híbrido


def compute_baseline_metrics(
    data_path: str,
    instance: str,
    fechas: list[str],
    valid_cities: list[str],
    dist_dict: dict,
    assignment_type: str = "historic",
    workday_hours: int = 8,
    dist_method: str = "haversine"
) -> pd.DataFrame:
    """
    Compute baseline metrics (historic solution) aggregated by city.

    Parameters
    ----------
    data_path : str
        Path to data directory.
    instance : str
        Instance name (e.g., 'inst3').
    fechas : list[str]
        List of dates (YYYY-MM-DD).
    valid_cities : list[str]
        Cities to include in the baseline.
    dist_dict : dict
        Precomputed distances.
    assignment_type : str, optional
        Assignment type, default 'historic'.
    workday_hours : int, optional
        Working day duration in hours, default 8.
    distance_method : str, optional
        Distance calculation method, default 'haversine'.

    Returns
    -------
    pd.DataFrame
        Aggregated baseline metrics per city with renamed columns:
        - vt_labors_baseline
        - tiempo_extra_baseline
        - driver_move_distance_baseline
    """
    # --- 1. Build historic results ---
    labors_hist_df, moves_hist_df = collect_results_to_df(
        data_path, instance, fechas, assignment_type=assignment_type
    )

    # --- 2. Compute metrics per city ---
    baseline_df = pd.DataFrame()
    for city in valid_cities:
        metrics_real_df = compute_metrics_with_moves(
            labors_hist_df,
            moves_hist_df,
            fechas,
            dist_dict,
            workday_hours=workday_hours,
            city=city,
            assignment_type=assignment_type,
            skip_weekends=False,
            dist_method=dist_method,
        )
        metrics_real_df["city"] = city
        baseline_df = pd.concat([baseline_df, metrics_real_df])

    # --- 3. Aggregate and rename ---
    baseline_df = (
        baseline_df.groupby(["city"])
        .agg({
            "driver_move_distance": "sum",
            "vt_count": "sum",
            "driver_extra_time": "sum",
        })
        .rename(columns={
            "vt_count": "vt_labors_baseline",
            "driver_extra_time": "driver_extra_time_baseline",
            "driver_move_distance": "driver_move_distance_baseline"
        })
        .reset_index()
    )

    return baseline_df


def collect_alpha_results_to_df(
    data_path: str, 
    instance: str, 
    dist_method: str, 
    metrics: list, 
    alphas: list, 
    iterations_nums: list
) -> tuple:
    labors_algo_df = pd.DataFrame()
    moves_algo_df = pd.DataFrame()

    for metric in metrics: 
        for alpha in alphas:
            for num_iter in iterations_nums:
                upload_path = (f'{data_path}/resultados/alpha_calibration/{instance}/'
                               f'{dist_method}/{metric}/res_{alpha:.1f}_{num_iter}.pkl')

                if not os.path.exists(upload_path):
                    continue
                with open(upload_path, "rb") as f:
                    res = pickle.load(f)
                    inc_values, duration, results_df, moves_df, metrics_df = res

                if not results_df.empty:
                    results_df = results_df.sort_values(["city", "date", "service_id", "labor_id"])
                if not moves_df.empty:
                    moves_df = moves_df.sort_values(["city", "date", "service_id", "labor_id"])

                # Normalize datetime columns to Bogotá tz
                datetime_cols = [
                    "labor_created_at",
                    "labor_start_date",
                    "labor_end_date",
                    "created_at",
                    "schedule_date",
                    "actual_start", 
                    "actual_end"
                    ]


                for df in (results_df, moves_df):
                    for col in datetime_cols:
                        if col in df.columns:
                            df[col] = (
                                pd.to_datetime(df[col], errors="coerce", utc=True)
                                .dt.tz_convert("America/Bogota")
                            )
                
                results_df['metric'] = metric
                results_df['alpha'] = alpha
                results_df['num_iter'] = num_iter

                moves_df['metric'] = metric
                moves_df['alpha'] = alpha
                moves_df['num_iter'] = num_iter

                labors_algo_df = pd.concat([labors_algo_df,results_df])
                moves_algo_df = pd.concat([moves_algo_df,moves_df])
    
    return labors_algo_df, moves_algo_df


def collect_alpha_metrics(
    data_path:str, 
    alphas:list, 
    iterations_nums:list,
    dist_method:str,
    instance:str,
    optimization=None
) -> pd.DataFrame:
    rows = []
    base_path = f"{data_path}/resultados/alpha_calibration/{instance}/{dist_method}/{optimization}"

    for alpha in alphas:
        for num_iter in iterations_nums:

            file_path = f'{base_path}/res_{alpha:.1f}_{num_iter}.pkl'

            if not os.path.exists(file_path):
                continue
            with open(file_path, "rb") as f:
                res = pickle.load(f)
                inc_values, duration, inc_results, inc_moves, inc_metrics = res

            # Loop over cities inside inc_metrics
            for city, metrics in inc_metrics.items():
                row = {
                    "alpha": alpha,
                    "num_iterations": num_iter,
                    "city": str(city),
                    "vt_labors": sum(metrics["vt_count"]),
                    "driver_extra_time": sum(metrics["driver_extra_time"]),
                    "driver_move_distance": sum(metrics["driver_move_distance"]),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def filter_df_on_hyperparameter_selection(df, hyperparameter_selection):
    mask = False  # start with empty mask
    for metric, configs in hyperparameter_selection.items():
        submask = (df['metric'] == metric)
        for param, value in configs.items():
            submask &= (df[param] == value)  # AND within config
        mask |= submask  # OR across configs
    return df[mask]


def compute_algo_vs_historic_metrics(
    labors_algo_df, moves_algo_df,
    labors_hist_df, moves_hist_df,
    fechas, dist_dict, valid_cities, metrics,
    hyperparameter_selection,
    dist_method="haversine", workday_hours=8
):
    """
    Compute algorithm vs historic metrics for each city and metric.
    Returns a daily-level comparison DataFrame.
    """
    comparison = []

    for metric in metrics:
        for city in valid_cities:
            # Select alpha & iteration for this metric
            alpha = hyperparameter_selection[metric]["alpha"]
            num_iter = hyperparameter_selection[metric]["num_iter"]

            # Filter algorithm outputs
            labors_algo_temp = flexible_filter(
                labors_algo_df, metric=metric, alpha=alpha, num_iter=num_iter
            )
            moves_algo_temp = flexible_filter(
                moves_algo_df, metric=metric, alpha=alpha, num_iter=num_iter
            )

            # Compute metrics
            metrics_hist_df = compute_metrics_with_moves(
                labors_hist_df, moves_hist_df,
                fechas=fechas,
                dist_dict=dist_dict,
                workday_hours=workday_hours,
                city=city,
                assignment_type="historic",
                skip_weekends=False,
                dist_method=dist_method
            )

            metrics_algo_df = compute_metrics_with_moves(
                labors_algo_temp, moves_algo_temp,
                fechas=fechas,
                dist_dict=dist_dict,
                workday_hours=workday_hours,
                city=city,
                assignment_type="algorithm",
                skip_weekends=False,
                dist_method=dist_method
            )

            # Compute absolute difference (Hist - Algo)
            diff = (metrics_hist_df.drop(columns=["day"]) -
                    metrics_algo_df.drop(columns=["day"]))
            diff["day"] = metrics_hist_df["day"]
            diff["city"] = city
            diff["metric"] = metric

            # Add perceptual gaps (% vs historic, avoiding division by zero)
            for col in ["vt_count", "num_drivers", "labor_extra_time",
                        "driver_extra_time", "driver_move_distance"]:
                hist_vals = metrics_hist_df[col].replace(0, pd.NA)
                diff[f"{col}_pct"] = ((hist_vals - metrics_algo_df[col]) /
                                      hist_vals) * 100

            comparison.append(diff)

    return pd.concat(comparison, ignore_index=True)


def aggregate_by_city(comparison, salary, extra_time):
    """
    Aggregate comparison metrics at city level with cost and percentage gaps.
    """
    agg = comparison.groupby(["metric", "city"]).agg({
        "vt_count": "sum",
        "num_drivers": "sum",
        "labor_extra_time": "sum",
        "driver_extra_time": "sum",
        "driver_move_distance": "sum",
        "vt_count_pct": "mean",
        "num_drivers_pct": "mean",
        "labor_extra_time_pct": "mean",
        "driver_extra_time_pct": "mean",
        "driver_move_distance_pct": "mean"
    }).reset_index()

    # Monetary cost estimates
    agg["extra_time_cost"] = agg["driver_extra_time"] * (extra_time / 60)
    agg["salary_cost"] = agg["num_drivers"] * salary
    agg["total_cost"] = agg["extra_time_cost"] + agg["salary_cost"]

    return agg


def aggregate_by_metric(comparison, salary, extra_time):
    """
    Aggregate comparison metrics at global (metric) level with cost and projections.
    """
    agg = comparison.groupby(["metric"]).agg({
        "vt_count": "sum",
        "num_drivers": "sum",
        "labor_extra_time": "sum",
        "driver_extra_time": "sum",
        "driver_move_distance": "sum",
        "vt_count_pct": "mean",
        "num_drivers_pct": "mean",
        "labor_extra_time_pct": "mean",
        "driver_extra_time_pct": "mean",
        "driver_move_distance_pct": "mean"
    }).reset_index()

    # Derived metrics
    agg["driver_extra_time_h"] = agg["driver_extra_time"] / 60
    agg["salary_cost"] = agg["num_drivers"] * salary
    agg["extra_time_cost"] = (agg["driver_extra_time"] * (extra_time / 60)).round()
    agg["total"] = (agg["salary_cost"] + agg["extra_time_cost"]).round()

    # Projections
    agg["total_con_ajuste"] = (agg["total"] * 0.85).round()
    agg["total_con_ajuste_mes"] = (agg["total_con_ajuste"] * 4).round()
    agg["total_con_ajuste_anio"] = (agg["total_con_ajuste_mes"] * 12).round()

    return agg


def print_header(instance:str, distance_type:str, assignment_type:str, optimization_variable:str):
    print(f'\n\n------------------ Running alpha tuning for -{instance}- ------------------')
    print(f'------- Run mode: sequential')
    print(f'------- Distances: {distance_type}')
    print(f'------- Assignment: {assignment_type}')
    print(f'------- Optimization variable: {optimization_variable}\n')


def print_iter_header(alpha:float):
    print(f'\n------ alpha: {alpha}')
    print('iter    found_on \trun_time \tnum_ser \textra_time \tdriv_dist')
