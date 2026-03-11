"""
Helper functions for the BUFFER_REACT algorithm.

These functions handle the "freeze horizon" logic: splitting an existing schedule
into frozen labors (keep as-is) and reassignable labors (re-optimize), and building
the driver state that reflects where each driver will be after completing frozen labors.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def split_labors_by_freeze_cutoff(
    preassigned_labors: pd.DataFrame,
    preassigned_moves: pd.DataFrame,
    freeze_cutoff: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an existing schedule into frozen and reassignable sets.

    Frozen labors are those with ``schedule_date <= freeze_cutoff`` AND an
    ``assigned_driver`` (i.e., truly active or starting imminently).
    Everything else goes to the reassignable set so it can be re-optimized.

    Parameters
    ----------
    preassigned_labors:
        Full existing schedule (all cities, any date).
    preassigned_moves:
        Movement timeline for the existing schedule.  Rows must carry a
        ``labor_id`` column that links them to a labor in preassigned_labors.
    freeze_cutoff:
        Timezone-aware timestamp.  Labors scheduled on or before this moment
        (AND already assigned) are frozen.

    Returns
    -------
    frozen_labors : pd.DataFrame
        Labors to keep unchanged.
    reassignable_labors : pd.DataFrame
        Labors to re-optimize (assignments stripped).
    frozen_moves : pd.DataFrame
        Movement rows whose labor_id belongs to a frozen labor.
    """
    if preassigned_labors.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    schedule_dates = pd.to_datetime(preassigned_labors["schedule_date"], errors="coerce", utc=True)

    # Align freeze_cutoff timezone
    if freeze_cutoff.tzinfo is None:
        freeze_cutoff = freeze_cutoff.tz_localize("UTC")
    else:
        freeze_cutoff = freeze_cutoff.tz_convert("UTC")

    in_freeze_window = schedule_dates <= freeze_cutoff

    # Only freeze labors that already have a driver assigned
    has_assignment = pd.Series(False, index=preassigned_labors.index)
    if "assigned_driver" in preassigned_labors.columns:
        has_assignment = preassigned_labors["assigned_driver"].notna()

    frozen_mask = in_freeze_window & has_assignment
    frozen_labors = preassigned_labors[frozen_mask].copy()
    reassignable_labors = preassigned_labors[~frozen_mask].copy()

    n_frozen = len(frozen_labors)
    n_reassignable = len(reassignable_labors)
    logger.info(
        "buffer_react_split freeze_cutoff=%s frozen=%d reassignable=%d",
        freeze_cutoff, n_frozen, n_reassignable,
    )

    # Extract moves for frozen labors only
    frozen_moves = pd.DataFrame()
    if (
        not preassigned_moves.empty
        and not frozen_labors.empty
        and "labor_id" in preassigned_moves.columns
    ):
        frozen_ids = set(frozen_labors["labor_id"].dropna().astype(str))
        frozen_moves = preassigned_moves[
            preassigned_moves["labor_id"].astype(str).isin(frozen_ids)
        ].copy()

    return frozen_labors, reassignable_labors, frozen_moves


def build_post_freeze_driver_states(
    frozen_labors: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Build partial driver-state overrides that reflect where each driver will be
    after completing all of their frozen labors.

    For each driver, finds the last frozen labor (by ``actual_end``) and returns
    ``{"position": <WKT map_end_point>, "available": <actual_end Timestamp>}``.
    Only ``position`` and ``available`` are overridden; ``work_start`` continues
    to come from the driver directory so workday constraints are preserved.

    Parameters
    ----------
    frozen_labors:
        DataFrame of frozen labors.  Must contain ``assigned_driver``,
        ``actual_end``, and ``map_end_point`` columns.

    Returns
    -------
    Dict mapping driver_key → partial DriverState override dict.
    Returns an empty dict when frozen_labors is empty or lacks required columns.
    """
    if frozen_labors.empty:
        return {}

    required = {"assigned_driver", "actual_end", "map_end_point"}
    if not required.issubset(frozen_labors.columns):
        logger.warning(
            "build_post_freeze_driver_states: missing columns %s — driver states not overridden",
            required - set(frozen_labors.columns),
        )
        return {}

    valid = frozen_labors.dropna(subset=["assigned_driver", "actual_end", "map_end_point"])
    if valid.empty:
        return {}

    overrides: Dict[str, Dict] = {}
    for driver_key, driver_frozen in valid.groupby("assigned_driver"):
        last = driver_frozen.sort_values("actual_end").iloc[-1]
        overrides[str(driver_key)] = {
            "position": str(last["map_end_point"]),
            "available": pd.Timestamp(last["actual_end"]),
        }

    logger.debug("buffer_react post_freeze_driver_states drivers=%d", len(overrides))
    return overrides


def strip_assignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clear assignment-specific columns so labors can be re-optimized from scratch.

    Modifies a *copy* of ``df``; the original is not mutated.
    """
    cols_to_clear = [
        "assigned_driver",
        "actual_start",
        "actual_end",
        "dist_km",
        "overtime_minutes",
        "actual_status",
    ]
    df = df.copy()
    for col in cols_to_clear:
        if col in df.columns:
            df[col] = pd.NA
    return df
