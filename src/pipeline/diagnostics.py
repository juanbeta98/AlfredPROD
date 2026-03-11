from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _finalize_assignment_diagnostics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()
    default_columns = {
        "is_infeasible": False,
        "infeasibility_cause_code": None,
        "infeasibility_cause_detail": None,
        "reassignment_candidate": False,
    }
    for col, default_value in default_columns.items():
        if col not in df.columns:
            df[col] = default_value

    failed_mask = df.get("actual_status", pd.Series(index=df.index, dtype="object")).astype("string").str.upper().eq("FAILED")
    reassignment_candidate_mask = df["reassignment_candidate"].fillna(False).infer_objects(copy=False).astype(bool)
    reassignment_failed_mask = failed_mask & reassignment_candidate_mask
    generic_failed_mask = failed_mask & ~reassignment_candidate_mask

    df.loc[failed_mask, "is_infeasible"] = True

    reassignment_code_missing = reassignment_failed_mask & (
        df["infeasibility_cause_code"].isna()
        | df["infeasibility_cause_code"].astype("string").str.strip().eq("")
    )
    df.loc[reassignment_code_missing, "infeasibility_cause_code"] = "reassignment_failed_unassigned"
    df.loc[reassignment_code_missing, "infeasibility_cause_detail"] = (
        "Service was moved to reassignment queue but no fully feasible assignment was found."
    )

    generic_code_missing = generic_failed_mask & (
        df["infeasibility_cause_code"].isna()
        | df["infeasibility_cause_code"].astype("string").str.strip().eq("")
    )
    df.loc[generic_code_missing, "infeasibility_cause_code"] = "assignment_failed_unassigned"
    df.loc[generic_code_missing, "infeasibility_cause_detail"] = (
        "No feasible driver assignment was found for this labor/service."
    )

    reassigned_success_mask = reassignment_candidate_mask & ~failed_mask
    df.loc[reassigned_success_mask, "is_infeasible"] = False
    solver_failure_code_mask = reassigned_success_mask & df["infeasibility_cause_code"].astype("string").isin(
        ["assignment_failed_unassigned", "reassignment_failed_unassigned"]
    )
    df.loc[solver_failure_code_mask, "infeasibility_cause_code"] = None
    df.loc[solver_failure_code_mask, "infeasibility_cause_detail"] = None

    return df


def _build_assignment_diagnostics_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    if results_df is None or results_df.empty:
        return {
            "rows_total": 0,
            "rows_infeasible": 0,
            "rows_warning": 0,
            "rows_unassigned": 0,
            "rows_reassignment_candidate": 0,
            "rows_reassignment_failed": 0,
            "rows_reassignment_success": 0,
            "infeasibility_causes": {},
            "warning_causes": {},
            "actual_status_counts": {},
        }

    df = results_df.copy()
    status_series = (
        df.get("actual_status", pd.Series(index=df.index, dtype="object"))
        .astype("string")
        .str.upper()
    )
    is_failed = status_series.eq("FAILED")
    reassignment_candidate = (
        df.get("reassignment_candidate", pd.Series(False, index=df.index))
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
    )
    is_infeasible = (
        df.get("is_infeasible", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    is_warning = (
        df.get("is_warning", pd.Series(False, index=df.index))
        .fillna(False)
        .astype(bool)
    )
    assigned_driver = df.get("assigned_driver", pd.Series(index=df.index, dtype="object"))
    rows_unassigned = int((assigned_driver.isna() | assigned_driver.astype("string").str.strip().isin(["", "nan", "None"])).sum())

    def _counts_to_dict(series: pd.Series) -> Dict[str, int]:
        if series is None or series.empty:
            return {}
        cleaned = series.dropna().astype("string").str.strip()
        cleaned = cleaned[cleaned.ne("") & cleaned.ne("<NA>") & cleaned.ne("nan")]
        if cleaned.empty:
            return {}
        return {str(k): int(v) for k, v in cleaned.value_counts().to_dict().items()}

    return {
        "rows_total": int(len(df)),
        "rows_infeasible": int(is_infeasible.sum()),
        "rows_warning": int(is_warning.sum()),
        "rows_unassigned": rows_unassigned,
        "rows_reassignment_candidate": int(reassignment_candidate.sum()),
        "rows_reassignment_failed": int((is_failed & reassignment_candidate).sum()),
        "rows_reassignment_success": int((~is_failed & reassignment_candidate).sum()),
        "infeasibility_causes": _counts_to_dict(df.get("infeasibility_cause_code", pd.Series(dtype="object"))),
        "warning_causes": _counts_to_dict(df.get("warning_code", pd.Series(dtype="object"))),
        "actual_status_counts": _counts_to_dict(status_series),
    }


def _stabilize_results_order(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()
    df["__service_id_num"] = pd.to_numeric(df.get("service_id"), errors="coerce")
    df["__labor_sequence_num"] = pd.to_numeric(df.get("labor_sequence"), errors="coerce")
    df["__labor_id_num"] = pd.to_numeric(df.get("labor_id"), errors="coerce")
    df["__schedule_ts"] = pd.to_datetime(df.get("schedule_date"), errors="coerce", utc=True)
    df["__actual_start_ts"] = pd.to_datetime(df.get("actual_start"), errors="coerce", utc=True)
    df["__created_ts"] = pd.to_datetime(df.get("created_at"), errors="coerce", utc=True)
    df["__orig_idx"] = range(len(df))
    df = df.sort_values(
        [
            "__service_id_num",
            "__labor_sequence_num",
            "__labor_id_num",
            "__schedule_ts",
            "__actual_start_ts",
            "__created_ts",
            "__orig_idx",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return df.drop(
        columns=[
            "__service_id_num",
            "__labor_sequence_num",
            "__labor_id_num",
            "__schedule_ts",
            "__actual_start_ts",
            "__created_ts",
            "__orig_idx",
        ],
        errors="ignore",
    )
