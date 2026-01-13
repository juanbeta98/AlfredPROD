#!/usr/bin/env python3
"""
simulate_artificial_day.py

Create simulated single-day instances sampling real SERVICES from a source month,
preserving per-city proportions and static/dynamic mix, and ensuring whole services
(i.e. all labors of a sampled service are included).

Outputs (per seed, per scenario):
  instances/simu_inst/N{n_services}/{scenario}/seed_{seed}/
    - labors_sim_df.csv
    - labors_sim_static_df.csv
    - labors_sim_dynamic_df.csv
    - metadata/summary.json
    - metadata/sampling_plan.csv
    - directorio_hist_df.csv

Author: patched version (applies the "service-level" sampling patches)
"""

import os
import json
import argparse
from typing import Dict, Tuple, List

from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

# PROJECT imports (adjust paths if necessary)
from src.data.data_load import load_tables
from src.utils.inst_generation_utils import filter_invalid_services, create_hist_directory, add_labor_sequence
from src.config.config import REPO_PATH
from src.config.SD_experimentation_config import *

# ---------------------------------------------------------------------
# UTIL: shift a timestamp to a new day preserving local time & tz
# ---------------------------------------------------------------------
def _shift_to_new_day(orig_ts, new_day, tz: str = "America/Bogota"):
    """
    Move a timestamp to a new calendar day preserving the local clock time.
    Handles numpy.datetime64 by coercing to pandas.Timestamp.

    orig_ts : timestamp-like
    new_day : date-like (YYYY-MM-DD or Timestamp) -- used as the target day (midnight base)
    tz : timezone string used for localization / conversion
    """
    if pd.isna(orig_ts) or pd.isna(new_day):
        return pd.NaT

    orig_ts = pd.Timestamp(orig_ts)
    new_day = pd.Timestamp(new_day)

    # Normalize original to tz
    if orig_ts.tzinfo is None:
        orig_local = orig_ts.tz_localize(tz)
    else:
        orig_local = orig_ts.tz_convert(tz)

    # Base of the new day in tz
    base = pd.Timestamp(new_day)
    if base.tzinfo is None:
        base = base.tz_localize(tz)
    else:
        base = base.tz_convert(tz)
    base = base.normalize()

    shifted = base + pd.Timedelta(
        hours=orig_local.hour,
        minutes=orig_local.minute,
        seconds=orig_local.second,
        microseconds=orig_local.microsecond
    )
    return shifted

# ---------------------------------------------------------------------
# Helper: classify static/dynamic at service level (DATE only)
# ---------------------------------------------------------------------
def build_service_table(month_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table with one row per service_id with columns:
    ['service_id', 'city', 'schedule_date', 'created_at', 'is_static', 'is_dynamic'].
    Static/dynamic classification is done using DATE only (no time).
    """
    # group at service level: pick first created_at (earliest) and schedule_date (first)
    service_df = (
        month_df.sort_values("created_at")
        .groupby("service_id", sort=False)
        .agg({
            "city": "first",
            "schedule_date": "first",
            "created_at": "first"
        })
        .reset_index()
    )

    # date-only comparison
    service_df["created_date_only"] = pd.to_datetime(service_df["created_at"]).dt.date
    service_df["schedule_date_only"] = pd.to_datetime(service_df["schedule_date"]).dt.date

    service_df["is_static"] = service_df["created_date_only"] < service_df["schedule_date_only"]
    service_df["is_dynamic"] = service_df["created_date_only"] == service_df["schedule_date_only"]

    return service_df

# ---------------------------------------------------------------------
# Compute sampling plan per city using service_df (NOT labors)
# ---------------------------------------------------------------------
def compute_city_plan(
    service_df: pd.DataFrame,
    n_services: int,
    scenario: str,
    rng: np.random.Generator,
    scenario_multiplier_map: Dict[str, Tuple[float,float]] = None
) -> Dict[str, dict]:
    """
    Compute sampling plan per city:
      - number of services to sample per city (proportional to city share),
      - historic static/dynamic means (per service),
      - simulated static proportion = historic_static_mean * sampled_multiplier (clamped 0..1),
      - derived integers n_static, n_dynamic (sum to n_city).
    """
    if scenario_multiplier_map is None:
        # multipliers chosen so that easy -> lower dynamic (more static), hard -> more dynamic
        scenario_multiplier_map = {
            "hard": (0.65, 0.85),    # multiplier applied to historic static mean
            "normal": (0.9, 1.1),
            "easy": (1.15, 1.35),
        }

    if scenario not in scenario_multiplier_map:
        raise ValueError(f"Unknown scenario '{scenario}' (valid: {list(scenario_multiplier_map.keys())})")

    low, high = scenario_multiplier_map[scenario]

    city_counts = service_df["city"].value_counts().sort_index()
    city_share = city_counts / city_counts.sum()
    n_city_samples = (city_share * n_services).round().astype(int)

    # adjust rounding diff to keep total exactly n_services
    diff = n_services - n_city_samples.sum()
    if diff != 0:
        top_city = city_share.idxmax()
        n_city_samples[top_city] += diff

    plan = {}
    for city, n_city in n_city_samples.items():
        city_service_df = service_df[service_df["city"] == city]
        historic_static_mean = float(city_service_df["is_static"].mean()) if len(city_service_df) > 0 else 0.0
        historic_dynamic_mean = float(city_service_df["is_dynamic"].mean()) if len(city_service_df) > 0 else 0.0

        mult = float(rng.uniform(low, high))
        static_prop_sim = historic_static_mean * mult
        static_prop_sim = min(max(static_prop_sim, 0.0), 1.0)
        dynamic_prop_sim = 1.0 - static_prop_sim

        n_static = int(round(static_prop_sim * n_city))
        n_dynamic = int(n_city - n_static)

        plan[city] = {
            "n_total": int(n_city),
            "historic_static_mean": historic_static_mean,
            "historic_dynamic_mean": historic_dynamic_mean,
            "multiplier": mult,
            "static_prop_sim": static_prop_sim,
            "dynamic_prop_sim": dynamic_prop_sim,
            "n_static": n_static,
            "n_dynamic": n_dynamic
        }

    return plan

# ---------------------------------------------------------------------
# Sample service_ids for a city according to plan (service-level)
# ---------------------------------------------------------------------
def sample_city_services(service_df_city: pd.DataFrame, plan_entry: dict, seed: int) -> List:
    """
    Return an array/list of sampled service_ids for the given city according to plan.
    This ensures whole services (all labors) will be included later.
    """
    rng = np.random.default_rng(seed)

    static_ids = service_df_city.loc[service_df_city["is_static"], "service_id"].unique()
    dynamic_ids = service_df_city.loc[service_df_city["is_dynamic"], "service_id"].unique()

    n_static = min(plan_entry["n_static"], len(static_ids))
    n_dynamic = min(plan_entry["n_dynamic"], len(dynamic_ids))

    sampled = []
    if n_static > 0:
        sampled_static = rng.choice(static_ids, size=n_static, replace=False)
        sampled.append(sampled_static)
    if n_dynamic > 0:
        sampled_dynamic = rng.choice(dynamic_ids, size=n_dynamic, replace=False)
        sampled.append(sampled_dynamic)

    if sampled:
        sampled_ids = np.concatenate(sampled)
    else:
        sampled_ids = np.array([], dtype=object)

    return list(sampled_ids)

# ---------------------------------------------------------------------
# Preserve deltas and shift schedule/created and other datetime columns
# ---------------------------------------------------------------------
def preserve_deltas_and_shift(
    df: pd.DataFrame,
    sim_day: str,
    tz: str = "America/Bogota",
    schedule_col: str = "schedule_date",
    created_col: str = "created_at",
    extra_datetime_cols: Tuple[str] = ("labor_start_date", "labor_end_date", "actual_start", "actual_end")
) -> pd.DataFrame:
    """
    Shift schedule_date to sim_day and move created_at preserving delta days:
      new_schedule_date = sim_day at same local time as original schedule_date
      delta_days = floor(original_schedule_date) - floor(created_at)
      new_created_at = new_schedule_date - delta_days (preserving time-of-day)
    Also shift other datetime columns that are offsets relative to schedule_date preserving their day offsets.

    Returns modified dataframe (copy).
    """
    df = df.copy()

    if schedule_col not in df.columns:
        raise KeyError(f"schedule_col '{schedule_col}' not found in dataframe")

    # normalize to Timestamp
    df[schedule_col] = pd.to_datetime(df[schedule_col], errors="coerce")
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce")

    target_day = pd.Timestamp(sim_day)

    # compute integer delta days (schedule_date.date - created_at.date)
    delta_days = (df[schedule_col].dt.floor("D") - df[created_col].dt.floor("D")).dt.days
    df["_delta_days"] = delta_days.fillna(0).astype(int)
    df["_orig_created_at"] = df[created_col]

    # shift schedule_date to sim_day preserving time-of-day
    df[schedule_col] = df[schedule_col].apply(lambda ts: _shift_to_new_day(ts, target_day, tz))

    # compute new created_at by shifting schedule back by delta_days preserving time-of-day
    def compute_new_created(row):
        sched_new = row[schedule_col]
        d = int(row["_delta_days"]) if not pd.isna(row["_delta_days"]) else 0
        # day base for created = sched_new - d days (midnight)
        new_day_for_created = (pd.Timestamp(sched_new) - pd.Timedelta(days=d)).normalize()
        return _shift_to_new_day(row["_orig_created_at"], new_day_for_created, tz)

    df[created_col] = df.apply(compute_new_created, axis=1)

    # Build original schedule (pre-shift) to compute offsets for other columns.
    # Original schedule = original_created_at + delta_days
    df["_orig_schedule"] = pd.to_datetime(df["_orig_created_at"], errors="coerce") + df["_delta_days"].apply(lambda d: pd.Timedelta(days=int(d)))

    # shift each extra datetime column preserving offset in days to original schedule
    for col in extra_datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # day offset (integer) relative to original schedule
            orig_offset = (df[col].dt.floor("D") - df["_orig_schedule"].dt.floor("D")).dt.days.fillna(0).astype(int)

            def compute_shifted_col(r, orig_col=col):
                if pd.isna(r[orig_col]):
                    return pd.NaT
                # new day base = shifted schedule normalized + offset days
                new_day_base = pd.Timestamp(r[schedule_col]).normalize() + pd.Timedelta(days=int(r["_tmp_offset"]))
                return _shift_to_new_day(r[orig_col], new_day_base, tz)

            df["_tmp_offset"] = orig_offset
            df[col] = df.apply(compute_shifted_col, axis=1)
            df.drop(columns=["_tmp_offset"], inplace=True, errors=True)

    # cleanup temporaries
    df.drop(columns=["_delta_days", "_orig_created_at", "_orig_schedule"], inplace=True, errors=True)
    return df

# ---------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------
def ensure_directory_layout(base_path: str, n_services: int, scenario: str, seed: int):
    base_dir = os.path.join(base_path, "instances", "simu_inst", f"N{n_services}", scenario, f"seed_{seed}")
    meta_dir = os.path.join(base_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    return base_dir, meta_dir

def save_outputs(
    simulated_df: pd.DataFrame,
    hist_directory: pd.DataFrame,
    sampling_plan: dict,
    data_path: str,
    n_services: int,
    scenario: str,
    seed: int,
    sim_day: str
) -> Dict[str, str]:
    base_dir, meta_dir = ensure_directory_layout(data_path, n_services, scenario, seed)
    labors_path = os.path.join(base_dir, "labors_sim_df.csv")
    labors_static_path = os.path.join(base_dir, "labors_sim_static_df.csv")
    labors_dynamic_path = os.path.join(base_dir, "labors_sim_dynamic_df.csv")
    hist_directory_path = os.path.join(base_dir, "directorio_hist_df.csv")

    simulated_df.to_csv(labors_path, index=False)
    simulated_df[simulated_df["is_static"]].to_csv(labors_static_path, index=False)
    simulated_df[simulated_df["is_dynamic"]].to_csv(labors_dynamic_path, index=False)
    hist_directory.to_csv(hist_directory_path, index=False)

    # sampling plan CSV
    sampling_rows = []
    for city, values in sampling_plan.items():
        row = values.copy()
        row["city"] = city
        sampling_rows.append(row)
    sampling_df = pd.DataFrame(sampling_rows)
    sampling_csv_path = os.path.join(meta_dir, "sampling_plan.csv")
    sampling_df.to_csv(sampling_csv_path, index=False)

    # summary JSON
    summary = {
        "sim_day": sim_day,
        "n_services": n_services,
        "scenario": scenario,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_city": sampling_plan,
    }
    summary_path = os.path.join(meta_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    return {
        "labors_path": labors_path,
        "labors_static_path": labors_static_path,
        "labors_dynamic_path": labors_dynamic_path,
        "sampling_csv_path": sampling_csv_path,
        "summary_path": summary_path,
        "hist_directory_path": hist_directory_path
    }

# ---------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------
def simulate_artificial_day(
    data_path: str,
    month: str,
    sim_day: str,
    n_services: int,
    seed: int = 42,
    scenario: str = "normal",
    tz: str = "America/Bogota"
):
    """
    Full pipeline to create a simulated single day instance.
    n_services counts SERVICES (not labors). All labors of a sampled service are included.
    """
    rng = np.random.default_rng(seed)

    # Load tables
    directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

    # Filter invalid services
    labors_filtered_df = filter_invalid_services(labors_raw_df, min_delay_minutes=120, only_unilabor_services=False)

    # ------ Define labor sequence order ------
    labors_filtered_df = add_labor_sequence(labors_filtered_df)

    # Ensure datetime columns exist and are datetime
    # Accept either 'schedule_date' or 'labor_start_date' as the schedule column; prefer 'schedule_date'.
    if "schedule_date" not in labors_filtered_df.columns and "labor_start_date" in labors_filtered_df.columns:
        labors_filtered_df = labors_filtered_df.rename(columns={"labor_start_date": "schedule_date",
                                                                "labor_end_date": "schedule_end_date"})
    # coerce datetimes for key columns safely
    for col in ["schedule_date", "created_at", "labor_start_date", "labor_end_date", "labor_start_date"]:
        if col in labors_filtered_df.columns:
            labors_filtered_df[col] = pd.to_datetime(labors_filtered_df[col], errors="coerce")

    # Determine timezone from schedule_date column if available
    from pandas.api.types import DatetimeTZDtype

    if isinstance(labors_filtered_df["schedule_date"].dtype, DatetimeTZDtype):
        tz = str(labors_filtered_df["schedule_date"].dt.tz)
        tz = tz.split("'")[1] if "'" in tz else tz  # extract clean tz name
    else:
        tz = "America/Bogota"

    month_start = pd.to_datetime(month + "-01").tz_localize(tz)
    # Month end: last microsecond of the month
    month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    month_df = labors_filtered_df[
        (labors_filtered_df["schedule_date"] >= month_start) &
        (labors_filtered_df["schedule_date"] <= month_end)
    ].copy()


    if month_df.empty:
        raise ValueError(f"No services found for month {month}")

    # Build service-level table and classify static/dynamic at service level
    service_df = build_service_table(month_df)

    # compute per-city plan (service-level)
    sampling_plan = compute_city_plan(service_df, n_services, scenario, rng)

    # sample service_ids per city
    sampled_service_ids = []
    for city, plan in sampling_plan.items():
        city_service_df = service_df[service_df["city"] == city]
        if city_service_df.empty:
            continue
        ids = sample_city_services(city_service_df, plan, seed + (abs(hash(city)) % 99999))
        sampled_service_ids.extend(ids)

    sampled_service_ids = np.array(list(dict.fromkeys(sampled_service_ids)))  # unique preserve order-ish

    # Ensure exact n_services: add or trim at service-level (try to be deterministic via seed)
    if len(sampled_service_ids) > n_services:
        sampled_service_ids = rng.choice(sampled_service_ids, size=n_services, replace=False)
    elif len(sampled_service_ids) < n_services:
        deficit = n_services - len(sampled_service_ids)
        remaining = service_df.loc[~service_df["service_id"].isin(sampled_service_ids), "service_id"].unique()
        if len(remaining) >= deficit:
            extra = rng.choice(remaining, size=deficit, replace=False)
            sampled_service_ids = np.concatenate([sampled_service_ids, extra])
        else:
            # Not enough remaining services to reach n_services; include them all (won't reach target)
            sampled_service_ids = np.concatenate([sampled_service_ids, remaining])

    # Reconstruct labor-level dataframe including all labors for sampled services
    simulated_df = month_df[month_df["service_id"].isin(sampled_service_ids)].copy().reset_index(drop=True)
    if simulated_df.empty:
        raise RuntimeError("No labors found after reconstructing sampled services (unexpected)")

    # Merge service-level flags into labors
    simulated_df = simulated_df.merge(
        service_df[["service_id", "is_static", "is_dynamic"]],
        on="service_id",
        how="left"
    )

    # SHIFT DATES: preserve deltas and shift schedule_date -> sim_day while preserving created_at delta
    # If your schedule column is 'schedule_date' in data, we pass that; otherwise adapt.
    simulated_shifted = preserve_deltas_and_shift(
        simulated_df,
        sim_day,
        tz=tz,
        schedule_col="schedule_date",
        created_col="created_at",
        extra_datetime_cols=("labor_start_date", "labor_end_date")
    )

    # Ensure common 'schedule_date' present
    if "schedule_date" not in simulated_shifted.columns and "labor_start_date" in simulated_shifted.columns:
        simulated_shifted["schedule_date"] = simulated_shifted["labor_start_date"]

    # create historic directory for the instance
    hist_directory = create_hist_directory(simulated_shifted)

    # Save outputs
    out_paths = save_outputs(
        simulated_shifted,
        hist_directory,
        sampling_plan,
        data_path,
        n_services,
        scenario,
        seed,
        sim_day
    )

    # Return for programmatic use
    return simulated_shifted, sampling_plan, out_paths    


# ---------------------------------------------------------------------
# CLI / Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an artificial single-day instance (v2).")
    data_path = f'{REPO_PATH}/data'
    parser.add_argument("--month", default="2025-06", help="Source month (YYYY-MM)")
    parser.add_argument("--sim_day", default="2026-11-11", help="Simulated day (YYYY-MM-DD)")
    args = parser.parse_args()

    for n_serv in n_services:
        for scenario in scenarios:
            for seed in seeds:
                df_sim, plan, paths = simulate_artificial_day(
                    data_path=data_path,
                    month=args.month,
                    sim_day=args.sim_day,
                    n_services=n_serv,
                    seed=seed,
                    scenario=scenario
                )
                
                print(f"\n✅ Simulated: {len(df_sim['service_id'].unique())} services, {len(df_sim)} labors")

    # print("Saved to:", paths)
