"""Two-solution comparison: KPI comparison, output writing, and Plotly figure builders.

All payload parsing and geometry helpers are imported from solution_evaluation
to keep a single source of truth.

Public API
----------
SolutionPair            Dataclass returned by load_and_prepare.
load_and_prepare        Load, filter, flatten, and reconstruct both solutions.
build_comparison        Build the full comparison dict (metrics + eval sections).
build_labor_detail      Build a per-labor side-by-side detail table.
compare                 Full compare pipeline: load → analyse → write → print.
build_comparison_gantt_figure          Side-by-side Gantt chart.
build_comparison_service_figure        Side-by-side distance-per-service bar.
build_comparison_driver_figure         Side-by-side distance-per-driver bar.
build_comparison_summary_table         Styled DataFrame with delta columns.
build_comparison_route_map             Folium map — both drivers' routes overlaid.

CLI
---
Run as:  python -m src.analysis.compare_solutions
Edit the CONFIGURATION block at the bottom of this file before running.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.data.parsing.driver_directory_parser import DriverDirectoryParser
from src.optimization.settings.model_params import ModelParams
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

from src.analysis.solution_evaluation import (
    BOGOTA_TZ,
    _ROUTE_COLOR,
    _SEGMENT_COLOR,
    _SEGMENT_LABEL,
    _SEGMENT_ORDER,
    _SUMMARY_METRICS,
    _VT_LABOR_TYPES,
    _deep_get,
    _fmt_value,
    _osrm_route,
    _to_plotly_dt,
    _wkt_to_latlon,
    build_coord_lookups,
    build_points_lookup_from_rows,
    filter_by_date,
    flatten_labors,
    infer_missing_durations,
    load_payload,
    reconstruct_timeline,
    recompute_move_distances,
    compute_payload_summary,
)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class SolutionPair:
    """All data derived from two loaded solutions, ready for plotting."""
    rows_a:       List[Dict[str, Any]]
    rows_b:       List[Dict[str, Any]]
    segments_a:   List[Dict[str, Any]]
    segments_b:   List[Dict[str, Any]]
    drivers_a:    List[str]
    drivers_b:    List[str]
    all_drivers:  List[str]
    all_services: List[str]
    # Per-labor coordinate lookup shared by both solutions (labor_id → (start_wkt, end_wkt)).
    # Populated automatically from payload addresses; used by build_comparison_route_map.
    points_lookup:      Optional[Dict[Any, Tuple[Optional[str], Optional[str]]]] = field(default=None)
    driver_home_lookup: Optional[Dict[str, str]] = field(default=None)


# ---------------------------------------------------------------------------
# Loading convenience
# ---------------------------------------------------------------------------


def load_and_prepare(
    sol_a: Path,
    sol_b: Path,
    input_file: Optional[Path] = None,
    driver_directory: Optional[Path] = None,
    planning_date: Optional[str] = None,
    distance_method: str = DEFAULT_DISTANCE_METHOD,
    speed_kmh: Optional[float] = None,
    metric_warn_threshold_pct: float = 5.0,
) -> SolutionPair:
    """Load, filter, flatten, and reconstruct both solutions into a SolutionPair.

    Parameters
    ----------
    sol_a / sol_b              : Paths to output_payload JSON files.
    input_file                 : Optional input mirror with full coordinate data.
                                 When provided, labor and move distances are
                                 recomputed from coordinates for fair comparison.
    driver_directory           : Optional driver directory JSON (same format as
                                 data/model_input/driver_directory.json).
                                 When provided, each driver's home coordinates seed
                                 the first-labor DRIVER_MOVE distance computation
                                 (home → first service) for solutions that lack addData.
    planning_date              : Drop labors before this date (YYYY-MM-DD).
    distance_method            : Passed to distance_utils; mirrors solver config.
    speed_kmh                  : Used to infer DRIVER_MOVE durations from distances.
                                 Defaults to ModelParams().alfred_speed_kmh.
    metric_warn_threshold_pct  : Divergence threshold for distance mismatch warnings.
    """
    params = ModelParams()
    if speed_kmh is None:
        speed_kmh = params.alfred_speed_kmh
    vt_speed_kmh = params.vehicle_transport_speed_kmh

    # Build driver home lookup from the driver directory (home → first-service move)
    driver_home_lookup: Optional[Dict[str, str]] = None
    if driver_directory is not None:
        if not driver_directory.exists():
            print(f"[warning] DRIVER_DIRECTORY not found: {driver_directory} — skipping home moves.")
        else:
            dir_json = json.loads(driver_directory.read_text(encoding="utf-8"))
            dir_df = DriverDirectoryParser.parse(dir_json)
            if not dir_df.empty:
                driver_home_lookup = {
                    str(row["driver_id"]): f"POINT({row['longitud']} {row['latitud']})"
                    for _, row in dir_df.iterrows()
                }
                print(f"[driver_home_lookup] {len(driver_home_lookup)} drivers loaded")

    # Build coordinate lookups from the shared input mirror
    coord_lookup = points_lookup = vt_labor_ids = None
    if input_file is not None:
        if not input_file.exists():
            print(f"[warning] INPUT_FILE not found: {input_file} — skipping computed distances.")
        else:
            raw = json.loads(input_file.read_text(encoding="utf-8"))
            input_svc = raw if isinstance(raw, list) else raw.get("data", [])
            coord_lookup, points_lookup, vt_labor_ids = build_coord_lookups(input_svc, distance_method)
            print(f"[coord_lookup] {len(coord_lookup)} labors | method={distance_method!r}")

    # Load and optionally filter by planning date
    services_a = load_payload(sol_a)
    services_b = load_payload(sol_b)
    if planning_date:
        services_a = filter_by_date(services_a, planning_date)
        services_b = filter_by_date(services_b, planning_date)

    # Augment vt_labor_ids from solution payloads that carry labor_type.
    # This covers labors whose services were filtered out of input_file
    # (e.g. input_solver.json has fewer services than the full solution scope).
    _extra_vt: set = set()
    for svc_list in (services_a, services_b):
        for svc in svc_list:
            for lab in svc.get("serviceLabors", []):
                if lab.get("labor_type") in _VT_LABOR_TYPES:
                    _extra_vt.add(lab["id"])
    if _extra_vt:
        vt_labor_ids = (vt_labor_ids or set()) | _extra_vt

    # Flatten to labor rows (coord_lookup provides consistent distances)
    rows_a, warnings_a = flatten_labors(services_a, coord_lookup, metric_warn_threshold_pct, vt_labor_ids)
    rows_b, warnings_b = flatten_labors(services_b, coord_lookup, metric_warn_threshold_pct, vt_labor_ids)

    # Recompute driver move distances from coordinates — apples-to-apples comparison.
    # Priority: stored map endpoints from addData (algorithm solutions) override
    # input-JSON-derived coordinates, so the recomputed distance matches the solver's
    # own calculation.  For labors without stored endpoints (e.g. baseline snapshots),
    # the input-JSON coordinates are used as-is.
    # The payload-stored driver_move_distance_km is authoritative when present; the
    # recomputed value is stored separately for reference and fills in missing values.
    if points_lookup or driver_home_lookup:
        for rows in (rows_a, rows_b):
            stored_pts = build_points_lookup_from_rows(rows)
            merged_pts = {**(points_lookup or {}), **stored_pts}
            move_dists = recompute_move_distances(
                rows, merged_pts, distance_method, driver_home_lookup
            )
            for r in rows:
                if r["labor_id"] in move_dists:
                    r["driver_move_distance_km_recomputed"] = move_dists[r["labor_id"]]
                    if not r.get("driver_move_distance_km"):
                        r["driver_move_distance_km"] = move_dists[r["labor_id"]]

    for label, warnings in [("sol_a", warnings_a), ("sol_b", warnings_b)]:
        for w in warnings:
            print(f"[{label}] {w}")

    # Infer duration for labors whose addData is absent (e.g. historic snapshots)
    for rows in (rows_a, rows_b):
        infer_missing_durations(
            rows, vt_speed_kmh,
            tiempo_alistar_min=params.tiempo_alistar_min,
            tiempo_finalizacion_min=params.tiempo_finalizacion_min,
        )

    # Reconstruct timelines
    segments_a = reconstruct_timeline(rows_a, speed_kmh)
    segments_b = reconstruct_timeline(rows_b, speed_kmh)

    drivers_a    = sorted({r["driver_id"] for r in rows_a if r["driver_id"] is not None})
    drivers_b    = sorted({r["driver_id"] for r in rows_b if r["driver_id"] is not None})
    all_drivers  = sorted(set(drivers_a) | set(drivers_b))
    all_services = sorted(
        {str(r["service_id"]) for r in rows_a + rows_b if r["service_id"] is not None}
    )

    # Build points_lookup from payload addresses if input_file didn't provide one.
    # This ensures route-map coordinate data is always available.
    if points_lookup is None:
        _, points_lookup, _ = build_coord_lookups(services_a)

    print(f"sol_a: {len(rows_a)} labors | {len(drivers_a)} drivers | {len(segments_a)} segments")
    print(f"sol_b: {len(rows_b)} labors | {len(drivers_b)} drivers | {len(segments_b)} segments")

    return SolutionPair(
        rows_a=rows_a, rows_b=rows_b,
        segments_a=segments_a, segments_b=segments_b,
        drivers_a=drivers_a, drivers_b=drivers_b,
        all_drivers=all_drivers, all_services=all_services,
        points_lookup=points_lookup,
        driver_home_lookup=driver_home_lookup,
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _delta(a: Any, b: Any) -> Optional[float]:
    try:
        return round(float(b) - float(a), 4)
    except (TypeError, ValueError):
        return None


def _delta_pct(a: Any, b: Any) -> Optional[float]:
    try:
        fa, fb = float(a), float(b)
        return round(100.0 * (fb - fa) / fa, 2) if fa != 0 else None
    except (TypeError, ValueError):
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Comparison building
# ---------------------------------------------------------------------------

_PAYLOAD_METRICS = [
    "services_count", "labors_count", "labors_vt_count", "drivers_count",
    "labors_assigned", "labors_infeasible", "labors_infeasible_pct",
    "labors_reassignment_candidate",
    "total_labor_distance_km", "total_driver_move_distance_km", "total_distance_km",
    "avg_labor_distance_km", "avg_driver_move_distance_km",
]

_EVAL_FLAT_METRICS = [
    "avg_vt_labor_distance_km", "avg_driver_move_distance_km_per_labor",
]

_EVAL_ENRICH_MAP: Dict[str, Tuple[str, str]] = {
    "avg_vt_labor_distance_km":              ("distance",   "avg_vt_labor_distance_km"),
    "avg_driver_move_distance_km_per_labor": ("distance",   "avg_driver_move_distance_km_per_labor"),
}


def _enrich_with_eval(
    payload_summary: Dict[str, Any],
    eval_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not eval_report:
        return payload_summary
    enriched = dict(payload_summary)
    for key, (section, field) in _EVAL_ENRICH_MAP.items():
        val = (eval_report.get(section) or {}).get(field)
        if val is not None:
            enriched[key] = val
    return enriched


def _flat_keys_of(d: Any, prefix: str = "") -> List[str]:
    if not isinstance(d, dict):
        return []
    keys: List[str] = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            keys.extend(_flat_keys_of(v, full))
        elif not isinstance(v, list):
            keys.append(full)
    return keys


def _metric_entry(label_a: str, label_b: str, va: Any, vb: Any) -> Dict[str, Any]:
    return {label_a: va, label_b: vb, "delta": _delta(va, vb), "delta_pct": _delta_pct(va, vb)}


def build_comparison(
    label_a: str,
    label_b: str,
    payload_summary_a: Dict[str, Any],
    payload_summary_b: Dict[str, Any],
    eval_a: Optional[Dict[str, Any]],
    eval_b: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the full comparison dict.

    Structure::

        labels         : [label_a, label_b]
        metrics        : { metric_name: {label_a, label_b, delta, delta_pct} }
        punctuality    : same shape, from eval reports
        time_allocation: same shape
        utilization    : same shape
    """
    sa = _enrich_with_eval(payload_summary_a, eval_a)
    sb = _enrich_with_eval(payload_summary_b, eval_b)
    all_flat_keys = _PAYLOAD_METRICS + [k for k in _EVAL_FLAT_METRICS if k not in _PAYLOAD_METRICS]

    metrics: Dict[str, Any] = {}
    for key in all_flat_keys:
        va, vb = sa.get(key), sb.get(key)
        if va is not None or vb is not None:
            metrics[key] = _metric_entry(label_a, label_b, va, vb)

    result: Dict[str, Any] = {"labels": [label_a, label_b], "metrics": metrics}

    for section in ("punctuality", "time_allocation", "utilization"):
        sec_a = (eval_a or {}).get(section)
        sec_b = (eval_b or {}).get(section)
        if sec_a is None and sec_b is None:
            continue
        all_keys = sorted(set(_flat_keys_of(sec_a)) | set(_flat_keys_of(sec_b)))
        result[section] = {
            k: _metric_entry(label_a, label_b, _deep_get(sec_a, k), _deep_get(sec_b, k))
            for k in all_keys
        }

    return result


# ---------------------------------------------------------------------------
# Per-labor detail
# ---------------------------------------------------------------------------


def build_labor_detail(
    label_a: str,
    label_b: str,
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One row per labor_id with side-by-side values from both solutions."""
    by_a = {r["labor_id"]: r for r in rows_a}
    by_b = {r["labor_id"]: r for r in rows_b}
    all_ids = sorted(set(by_a) | set(by_b), key=lambda x: (x is None, x))

    detail: List[Dict[str, Any]] = []
    for labor_id in all_ids:
        ra, rb = by_a.get(labor_id), by_b.get(labor_id)
        presence = "both" if (ra and rb) else (label_a if ra else label_b)

        def _v(r: Optional[Dict], f: str) -> Any:
            return r[f] if r is not None else None

        driver_a, driver_b = _v(ra, "driver_id"), _v(rb, "driver_id")
        row: Dict[str, Any] = {
            "labor_id":         labor_id,
            "service_id":       _v(ra, "service_id") or _v(rb, "service_id"),
            "presence":         presence,
            f"driver_id_{label_a}": driver_a,
            f"driver_id_{label_b}": driver_b,
            "driver_changed":   "" if (ra is None or rb is None) else str(driver_a != driver_b),
        }
        for src, col in [
            ("labor_distance_km",      "labor_dist_km"),
            ("driver_move_distance_km","move_dist_km"),
            ("duration_min",           "duration_min"),
        ]:
            va, vb = _v(ra, src), _v(rb, src)
            row[f"{col}_{label_a}"] = va
            row[f"{col}_{label_b}"] = vb
            row[f"delta_{col}"] = _delta(va, vb)

        row[f"is_infeasible_{label_a}"] = _v(ra, "is_infeasible")
        row[f"is_infeasible_{label_b}"] = _v(rb, "is_infeasible")
        row[f"reassignment_{label_a}"]  = _v(ra, "reassignment_candidate")
        row[f"reassignment_{label_b}"]  = _v(rb, "reassignment_candidate")
        detail.append(row)

    return detail


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})


def _comparison_to_csv_rows(
    comparison: Dict[str, Any], label_a: str, label_b: str
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for section, data in comparison.items():
        if section == "labels" or not isinstance(data, dict):
            continue
        for metric, values in data.items():
            if not isinstance(values, dict):
                continue
            rows.append({
                "section": section, "metric": metric,
                label_a: _fmt(values.get(label_a)),
                label_b: _fmt(values.get(label_b)),
                "delta": _fmt(values.get("delta")),
                "delta_pct": _fmt(values.get("delta_pct")),
            })
    return rows


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

_CONSOLE_GROUPS: List[Tuple[str, List[str]]] = [
    ("DIMENSIONS",    ["services_count", "labors_count", "labors_vt_count", "drivers_count"]),
    ("ASSIGNMENT",    ["labors_assigned", "labors_infeasible", "labors_infeasible_pct",
                       "labors_reassignment_candidate"]),
    ("DISTANCE (km)", ["total_labor_distance_km", "total_driver_move_distance_km",
                       "total_distance_km", "avg_labor_distance_km",
                       "avg_driver_move_distance_km", "avg_vt_labor_distance_km",
                       "avg_driver_move_distance_km_per_labor"]),
]
_PUNCT_KEYS = ["grace_minutes", "late_services_count", "late_services_pct",
               "total_lateness_min", "avg_lateness_min_all_considered", "normalized_tardiness_pct"]
_UTIL_KEYS  = ["system.utilization_without_moves_pct", "system.utilization_with_moves_pct",
               "system.driver_move_utilization_pct", "system.driver_move_share_of_active_pct"]
_TIME_KEYS  = ["timeline_total_min", "free_time_min", "driver_move_min", "labor_work_min",
               "free_time_pct", "driver_move_pct", "labor_work_pct"]
_W = 44


def _print_group(name: str, keys: List[str], data: Dict, label_a: str, label_b: str) -> None:
    lines = []
    for key in keys:
        entry = data.get(key)
        if entry is None:
            continue
        lines.append(
            f"  {key:<{_W}} {_fmt(entry.get(label_a)):>13} {_fmt(entry.get(label_b)):>13}"
            f" {_fmt(entry.get('delta')):>12} {_fmt(entry.get('delta_pct')):>9}"
        )
    if not lines:
        return
    print(f"  {name}")
    print(f"  {'Metric':<{_W}} {label_a:>13} {label_b:>13} {'Delta':>12} {'Delta%':>9}")
    print(f"  {'-'*_W} {'-'*13} {'-'*13} {'-'*12} {'-'*9}")
    for line in lines:
        print(line)
    print()


def _print_report(
    label_a: str, label_b: str, comparison: Dict[str, Any],
    n_a: int, n_b: int, n_shared: int, n_a_only: int, n_b_only: int,
    warnings: Optional[List[str]] = None,
) -> None:
    w = _W + 51
    print(); print("=" * w)
    print(f"  SOLUTION COMPARISON: {label_a!r} vs {label_b!r}")
    print("=" * w)
    print(f"  Labors in {label_a}: {n_a} | in {label_b}: {n_b}")
    print(f"  Shared: {n_shared} | only in {label_a}: {n_a_only} | only in {label_b}: {n_b_only}")
    print()
    metrics = comparison.get("metrics", {})
    for name, keys in _CONSOLE_GROUPS:
        _print_group(name, keys, metrics, label_a, label_b)
    if "punctuality"    in comparison: _print_group("PUNCTUALITY",    _PUNCT_KEYS, comparison["punctuality"],    label_a, label_b)
    if "time_allocation"in comparison: _print_group("TIME ALLOCATION",_TIME_KEYS,  comparison["time_allocation"],label_a, label_b)
    if "utilization"    in comparison: _print_group("UTILIZATION",    _UTIL_KEYS,  comparison["utilization"],    label_a, label_b)
    if warnings:
        print(f"  WARNINGS ({len(warnings)})")
        print(f"  {'-'*(w-2)}")
        for w_msg in warnings: print(f"  ! {w_msg}")
        print()


# ---------------------------------------------------------------------------
# Figure builders — two solutions
# ---------------------------------------------------------------------------


def _gantt_traces(
    segments: List[Dict[str, Any]],
    driver_set: set,
    show_legend: bool,
) -> List[go.Bar]:
    by_type: Dict[str, List] = defaultdict(list)
    for seg in segments:
        if seg["driver_id"] in driver_set:
            by_type[seg["segment_type"]].append(seg)

    traces = []
    for seg_type in _SEGMENT_ORDER:
        segs = by_type.get(seg_type, [])
        if not segs:
            continue
        # Minimum display width for zero-duration VT bars (historic snapshots lack addData)
        _MIN_VT_MS = 5 * 60 * 1000  # 5 minutes in ms
        base_vals, x_vals, y_vals, hovers = [], [], [], []
        for s in segs:
            dur_ms = (s["end"] - s["start"]).total_seconds() * 1000
            if dur_ms < 0:
                continue
            if seg_type == "VEHICLE_TRANSPORTATION" and dur_ms == 0:
                dur_ms = _MIN_VT_MS
            elif dur_ms == 0:
                continue
            base_vals.append(_to_plotly_dt(s["start"]))
            x_vals.append(dur_ms)
            y_vals.append(s["driver_id"])
            ht = (
                f"<b>{_SEGMENT_LABEL[seg_type]}</b><br>"
                f"Driver: {s['driver_id']}<br>"
                f"Labor: {s['labor_id']}<br>"
                f"Service: {s['service_id']}<br>"
                f"Start: {s['start'].strftime('%H:%M')}<br>"
                f"End:   {s['end'].strftime('%H:%M')}<br>"
                f"Duration: {s['duration_min']:.1f} min"
            )
            if s.get("distance_km", 0) > 0:
                ht += f"<br>Distance: {s['distance_km']:.2f} km"
            if s.get("is_infeasible"):
                ht += "<br><i>⚠ infeasible</i>"
            hovers.append(ht)

        traces.append(go.Bar(
            x=x_vals, y=y_vals, base=base_vals, orientation="h",
            name=_SEGMENT_LABEL[seg_type],
            marker_color=_SEGMENT_COLOR[seg_type],
            hovertext=hovers, hoverinfo="text",
            showlegend=show_legend,
            legendgroup=seg_type,
        ))
    return traces


def build_comparison_gantt_figure(
    pair: SolutionPair,
    label_a: str,
    label_b: str,
) -> go.Figure:
    """Side-by-side Gantt chart for two solutions on a shared Y axis."""
    driver_set = set(pair.all_drivers)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[label_a, label_b],
        shared_yaxes=True, horizontal_spacing=0.04,
    )
    for col, (segs, show_legend) in enumerate(
        [(pair.segments_a, True), (pair.segments_b, False)], start=1
    ):
        for trace in _gantt_traces(segs, driver_set, show_legend):
            fig.add_trace(trace, row=1, col=col)

    all_starts = [s["start"] for s in pair.segments_a + pair.segments_b if s["driver_id"] in driver_set]
    all_ends   = [s["end"]   for s in pair.segments_a + pair.segments_b if s["driver_id"] in driver_set]
    if all_starts and all_ends:
        pad = timedelta(minutes=30)
        t_min = _to_plotly_dt(min(all_starts) - pad)
        t_max = _to_plotly_dt(max(all_ends) + pad)
        for axis in ("xaxis", "xaxis2"):
            fig.update_layout(**{axis: dict(range=[t_min, t_max])})

    # Sort drivers by algorithm (sol_b) total driver-move distance descending
    move_by_driver: dict = defaultdict(float)
    for r in pair.rows_b:
        if r["driver_id"] is not None:
            move_by_driver[r["driver_id"]] += r.get("driver_move_distance_km") or 0.0
    driver_order = sorted(pair.all_drivers, key=lambda d: move_by_driver.get(d, 0.0), reverse=True)

    fig.update_xaxes(type="date", tickformat="%H:%M", title_text="Time (Bogotá)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(driver_order)),
        title_text="Driver", col=1,
    )
    row_h = max(35, 600 // max(len(pair.all_drivers), 1))
    fig.update_layout(
        title_text="Driver Timelines",
        barmode="overlay",
        height=max(420, row_h * len(pair.all_drivers) + 160),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="closest",
        margin=dict(l=80, r=20, t=100, b=60),
    )
    return fig


def build_comparison_service_figure(
    pair: SolutionPair,
    label_a: str,
    label_b: str,
) -> go.Figure:
    """Side-by-side stacked bar — distance per service, shared service order."""
    def _agg(rows):
        df = pd.DataFrame(rows)
        agg = (
            df.groupby("service_id", dropna=False)
            .agg(labor_dist=("labor_distance_km", "sum"), move_dist=("driver_move_distance_km", "sum"))
            .reset_index()
        )
        agg["total_dist"] = agg["labor_dist"] + agg["move_dist"]
        agg["service_id"] = agg["service_id"].astype(str)
        return agg.set_index("service_id")

    svc_a = _agg(pair.rows_a)
    svc_b = _agg(pair.rows_b)
    svc_order = (
        svc_a["total_dist"].reindex(pair.all_services, fill_value=0)
        .sort_values(ascending=False).index.tolist()
    )

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[label_a, label_b],
        shared_yaxes=True, horizontal_spacing=0.04,
    )
    for col, (svc_df, show_legend) in enumerate([(svc_a, True), (svc_b, False)], start=1):
        labor_vals = [round(svc_df.loc[s, "labor_dist"], 2) if s in svc_df.index else 0 for s in svc_order]
        move_vals  = [round(svc_df.loc[s, "move_dist"],  2) if s in svc_df.index else 0 for s in svc_order]
        fig.add_trace(go.Bar(
            name="Labor distance", x=svc_order, y=labor_vals,
            marker_color="rgba(55, 115, 200, 0.85)",
            hovertext=[f"Service {s}<br>Labor: {l:.2f} km" for s, l in zip(svc_order, labor_vals)],
            hoverinfo="text", showlegend=show_legend, legendgroup="labor_dist",
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            name="Move distance", x=svc_order, y=move_vals,
            marker_color="rgba(250, 155, 45, 0.85)",
            hovertext=[f"Service {s}<br>Move: {m:.2f} km" for s, m in zip(svc_order, move_vals)],
            hoverinfo="text", showlegend=show_legend, legendgroup="move_dist",
        ), row=1, col=col)

    fig.update_xaxes(title_text="Service ID", tickangle=45)
    fig.update_yaxes(title_text="Distance (km)", col=1)
    fig.update_layout(
        title_text="Distance per Service", barmode="stack", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="x", margin=dict(l=60, r=20, t=100, b=100),
    )
    return fig


def build_comparison_driver_figure(
    pair: SolutionPair,
    label_a: str,
    label_b: str,
) -> go.Figure:
    """Side-by-side stacked bar — distance per driver, shared driver order."""
    def _agg(rows):
        assigned = [r for r in rows if r["driver_id"] is not None]
        if not assigned:
            return pd.DataFrame(
                columns=["driver_id", "labor_dist", "move_dist", "labor_count"]
            ).set_index("driver_id")
        df = pd.DataFrame(assigned)
        agg = (
            df.groupby("driver_id")
            .agg(
                labor_dist=("labor_distance_km", "sum"),
                move_dist=("driver_move_distance_km", "sum"),
                labor_count=("labor_id", "count"),
            )
            .reset_index()
        )
        agg["total_dist"] = agg["labor_dist"] + agg["move_dist"]
        return agg.set_index("driver_id")

    drv_a = _agg(pair.rows_a)
    drv_b = _agg(pair.rows_b)
    drv_order = (
        drv_a["move_dist"].reindex(pair.all_drivers, fill_value=0)
        .sort_values(ascending=False).index.tolist()
    )
    counts_a = [int(drv_a.loc[d, "labor_count"]) if d in drv_a.index else 0 for d in drv_order]
    counts_b = [int(drv_b.loc[d, "labor_count"]) if d in drv_b.index else 0 for d in drv_order]

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[label_a, label_b],
        shared_yaxes=True, horizontal_spacing=0.04,
    )
    for col, (drv_df, show_legend, counts) in enumerate(
        [(drv_a, True, counts_a), (drv_b, False, counts_b)], start=1
    ):
        labor_vals = [round(drv_df.loc[d, "labor_dist"], 2) if d in drv_df.index else 0 for d in drv_order]
        move_vals  = [round(drv_df.loc[d, "move_dist"],  2) if d in drv_df.index else 0 for d in drv_order]
        fig.add_trace(go.Bar(
            name="Labor distance", x=drv_order, y=labor_vals,
            marker_color="rgba(55, 115, 200, 0.85)",
            hovertext=[f"Driver {d}<br>Labor: {l:.2f} km" for d, l in zip(drv_order, labor_vals)],
            hoverinfo="text", showlegend=show_legend, legendgroup="labor_dist",
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            name="Move distance", x=drv_order, y=move_vals,
            marker_color="rgba(250, 155, 45, 0.85)",
            hovertext=[f"Driver {d}<br>Move: {m:.2f} km" for d, m in zip(drv_order, move_vals)],
            hoverinfo="text", showlegend=show_legend, legendgroup="move_dist",
            text=counts, textposition="outside", textfont=dict(size=11),
        ), row=1, col=col)

    fig.update_xaxes(title_text="Driver ID", tickangle=45)
    fig.update_yaxes(title_text="Distance (km)", col=1)
    fig.update_layout(
        title_text="Distance per Driver", barmode="stack", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="x", margin=dict(l=60, r=20, t=100, b=100),
    )
    return fig


def build_overview_table(
    pair: SolutionPair,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Side-by-side overview DataFrame with aggregate metrics for both solutions."""
    _grace = ModelParams().tiempo_gracia_min
    # Pass segments so compute_payload_summary merges timeline-overlap infeasibilities
    # (detected during reconstruct_timeline) into labors_infeasible.
    sa = compute_payload_summary(pair.rows_a, tiempo_gracia_min=_grace, segments=pair.segments_a)
    sb = compute_payload_summary(pair.rows_b, tiempo_gracia_min=_grace, segments=pair.segments_b)

    sa["labors_non_vt_count"] = sa["labors_count"] - sa["labors_vt_count"]
    sb["labors_non_vt_count"] = sb["labors_count"] - sb["labors_vt_count"]
    sa["labors_unassigned"] = sa["labors_count"] - sa["labors_assigned"]
    sb["labors_unassigned"] = sb["labors_count"] - sb["labors_assigned"]

    _METRICS: List[Tuple[str, str]] = [
        ("services_count",                "services"),
        ("labors_count",                  "labors_total"),
        ("labors_vt_count",               "labors_vt"),
        ("labors_non_vt_count",           "labors_non_vt"),
        ("drivers_count",                 "drivers_used"),
        ("labors_assigned",               "labors_assigned"),
        ("labors_unassigned",             "labors_unassigned"),
        ("labors_infeasible",             "labors_infeasible"),
        ("labors_infeasible_pct",         "labors_infeasible_pct"),
        ("labors_in_grace",               "labors_in_grace"),
        ("total_grace_min",               "total_grace_min"),
        ("labors_reassignment_candidate", "reassignment_candidates"),
        ("total_labor_distance_km",       "total_labor_distance_km"),
        ("total_driver_move_distance_km", "total_driver_move_distance_km"),
        ("total_distance_km",             "total_distance_km"),
        ("avg_labor_distance_km",         "avg_labor_distance_km"),
        ("avg_driver_move_distance_km",   "avg_driver_move_distance_km"),
    ]

    rows: List[Dict[str, Any]] = []
    for key, metric in _METRICS:
        va, vb = sa.get(key), sb.get(key)
        d  = _delta(va, vb)
        dp = _delta_pct(va, vb)
        rows.append({
            "metric": metric,
            label_a:  va,
            label_b:  vb,
            "delta":  d,
            "delta_pct": dp,
        })

    return pd.DataFrame(rows).set_index("metric")


def build_comparison_summary_table(
    eval_a: Optional[Any],
    eval_b: Optional[Any],
    label_a: str,
    label_b: str,
) -> Optional["pd.io.formats.style.Styler"]:
    """Styled DataFrame comparing eval-report KPIs side-by-side with delta columns.

    Each of eval_a / eval_b can be a Path, a pre-loaded dict, or None.
    Returns None when both are absent.
    """
    def _load(src: Optional[Any]) -> Optional[Dict[str, Any]]:
        if src is None:
            return None
        if isinstance(src, Path):
            return json.loads(src.read_text(encoding="utf-8")) if src.exists() else None
        return src

    data_a = _load(eval_a)
    data_b = _load(eval_b)
    if data_a is None and data_b is None:
        print("No evaluation reports provided — set EVAL_A / EVAL_B to enable this section.")
        return None

    rows = []
    for dotted, display in _SUMMARY_METRICS:
        va = _deep_get(data_a, dotted) if data_a else None
        vb = _deep_get(data_b, dotted) if data_b else None
        d  = _delta(va, vb)
        dp = _delta_pct(va, vb)
        arrow = ("▲" if d > 0 else "▽") if d is not None and d != 0 else ""
        rows.append({
            "Metric":   display,
            label_a:    _fmt_value(va),
            label_b:    _fmt_value(vb),
            "Delta":    f"{arrow} {_fmt_value(d)}"  if d  is not None else "—",
            "Delta %":  f"{_fmt_value(dp)} %"       if dp is not None else "—",
        })

    df = pd.DataFrame(rows)
    return df.style.set_properties(**{"text-align": "right"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


# ---------------------------------------------------------------------------
# Route map — two solutions overlaid on a single folium map
# ---------------------------------------------------------------------------

# Colour schemes for each solution in the comparison map.
_CMP_COLORS: Dict[str, Dict[str, str]] = {
    "a": {"VEHICLE_TRANSPORTATION": "blue",   "DRIVER_MOVE": "cadetblue"},
    "b": {"VEHICLE_TRANSPORTATION": "red",    "DRIVER_MOVE": "orange"},
}


def _draw_route_on_map(
    m: Any,
    rows: List[Dict[str, Any]],
    points_lookup: Dict[Any, Tuple[Optional[str], Optional[str]]],
    driver_id: str,
    colors: Dict[str, str],
    driver_home_wkt: Optional[str],
    use_osrm: bool,
    osrm_url: Optional[str],
) -> None:
    """Render one solution's route onto an existing folium map (mutates m)."""
    driver_rows = sorted(
        [r for r in rows if r["driver_id"] == str(driver_id) and r["actual_start"] is not None],
        key=lambda r: r["actual_start"],
    )
    prev_end_wkt: Optional[str] = driver_home_wkt
    for row in driver_rows:
        lid = row["labor_id"]
        start_wkt, end_wkt = points_lookup.get(lid, (None, None))
        if start_wkt is None and end_wkt is None:
            continue

        start_ll = _wkt_to_latlon(start_wkt)
        end_ll = _wkt_to_latlon(end_wkt)

        # Driver-move leg
        if prev_end_wkt and start_wkt and prev_end_wkt != start_wkt:
            prev_ll = _wkt_to_latlon(prev_end_wkt)
            if prev_ll and start_ll:
                _add_polyline(m, prev_ll, start_ll, colors["DRIVER_MOVE"], use_osrm, osrm_url)

        # Labor leg
        is_vt = row.get("labor_type") in _VT_LABOR_TYPES
        seg_color = colors["VEHICLE_TRANSPORTATION"] if is_vt else colors["DRIVER_MOVE"]
        if start_ll and end_ll and start_wkt != end_wkt:
            _add_polyline(m, start_ll, end_ll, seg_color, use_osrm, osrm_url)

        # Waypoint marker
        import folium as _folium
        t_start = row["actual_start"].strftime("%H:%M") if row["actual_start"] else "—"
        t_end   = row["actual_end"].strftime("%H:%M")   if row["actual_end"]   else "—"
        popup_html = (
            f"<b>Labor {lid}</b><br>"
            f"Service: {row['service_id']}<br>"
            f"Start: {t_start} &nbsp; End: {t_end}"
        )
        target_ll = end_ll or start_ll
        if target_ll:
            _folium.CircleMarker(
                location=target_ll,
                radius=5,
                color=seg_color,
                fill=True,
                fill_color=seg_color,
                fill_opacity=0.7,
                popup=_folium.Popup(popup_html, max_width=220),
            ).add_to(m)

        prev_end_wkt = end_wkt or prev_end_wkt


def _add_polyline(
    m: Any,
    from_ll: Tuple[float, float],
    to_ll: Tuple[float, float],
    color: str,
    use_osrm: bool,
    osrm_url: Optional[str],
) -> None:
    import folium as _folium
    route_coords = None
    if use_osrm:
        try:
            route_coords = _osrm_route(from_ll, to_ll, osrm_url)
        except Exception:
            pass
    if route_coords:
        _folium.PolyLine(route_coords, color=color, weight=4, opacity=0.8).add_to(m)
    else:
        _folium.PolyLine([from_ll, to_ll], color=color, weight=2, dash_array="5,5").add_to(m)


def build_comparison_route_map(
    pair: "SolutionPair",
    driver_id: str,
    label_a: str = "baseline",
    label_b: str = "algorithm",
    use_osrm: bool = True,
    zoom_start: int = 12,
) -> Any:
    """Build a folium Map with both solutions' routes for a given driver overlaid.

    Solution A is rendered in blue (VT) / cadetblue (moves).
    Solution B is rendered in red (VT) / orange (moves).
    Dashed lines are used as fallback when OSRM is unavailable.

    Parameters
    ----------
    pair      : SolutionPair returned by load_and_prepare().
    driver_id : Driver whose route to render (as a string matching rows["driver_id"]).
    label_a   : Display name for solution A (shown in the legend).
    label_b   : Display name for solution B (shown in the legend).
    use_osrm  : When True, request realistic road geometry from OSRM.
    zoom_start: Initial zoom level for the folium map.

    Returns
    -------
    folium.Map
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError(
            "folium is required for route maps. Install with: pip install folium"
        ) from exc

    import os as _os

    if pair.points_lookup is None:
        raise ValueError(
            "SolutionPair.points_lookup is None — re-run load_and_prepare() to populate it."
        )

    in_a = any(r["driver_id"] == str(driver_id) for r in pair.rows_a)
    in_b = any(r["driver_id"] == str(driver_id) for r in pair.rows_b)
    if not in_a and not in_b:
        raise ValueError(
            f"Driver {driver_id!r} not found in either solution. "
            f"Available drivers: {pair.all_drivers}"
        )
    if not in_a:
        print(f"[route_map] Driver {driver_id!r} absent from {label_a!r} — rendering {label_b!r} only.")
    if not in_b:
        print(f"[route_map] Driver {driver_id!r} absent from {label_b!r} — rendering {label_a!r} only.")

    # Collect all waypoints to find the map center
    all_latlons: List[Tuple[float, float]] = []
    home_wkt = (pair.driver_home_lookup or {}).get(str(driver_id))
    if home_wkt:
        ll = _wkt_to_latlon(home_wkt)
        if ll:
            all_latlons.append(ll)
    for rows in (pair.rows_a, pair.rows_b):
        for r in rows:
            if r["driver_id"] == str(driver_id):
                for wkt in pair.points_lookup.get(r["labor_id"], (None, None)):
                    ll = _wkt_to_latlon(wkt)
                    if ll:
                        all_latlons.append(ll)

    if not all_latlons:
        raise ValueError(
            f"No coordinate data found for driver {driver_id!r}. "
            "Ensure the payload carries address coordinates."
        )

    center = [
        sum(ll[0] for ll in all_latlons) / len(all_latlons),
        sum(ll[1] for ll in all_latlons) / len(all_latlons),
    ]
    m = folium.Map(location=center, zoom_start=zoom_start)

    osrm_url = _os.environ.get("OSRM_URL")

    if in_a:
        _draw_route_on_map(
            m, pair.rows_a, pair.points_lookup, driver_id,
            _CMP_COLORS["a"], home_wkt, use_osrm, osrm_url,
        )
    if in_b:
        _draw_route_on_map(
            m, pair.rows_b, pair.points_lookup, driver_id,
            _CMP_COLORS["b"], home_wkt, use_osrm, osrm_url,
        )

    # Driver home marker (shared between solutions)
    if home_wkt:
        home_ll = _wkt_to_latlon(home_wkt)
        if home_ll:
            folium.Marker(
                location=home_ll,
                popup=folium.Popup(f"<b>Home</b><br>Driver: {driver_id}", max_width=180),
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)

    # HTML legend overlay
    legend_html = (
        "<div style='"
        "position:fixed;bottom:30px;left:30px;z-index:1000;"
        "background:white;padding:10px 14px;border-radius:6px;"
        "border:1px solid #ccc;font-size:13px;line-height:1.6'>"
        f"<b>Driver {driver_id}</b><br>"
        f"<span style='color:blue'>&#9644;</span> {label_a} — service<br>"
        f"<span style='color:cadetblue'>&#9644;</span> {label_a} — move<br>"
        f"<span style='color:red'>&#9644;</span> {label_b} — service<br>"
        f"<span style='color:orange'>&#9644;</span> {label_b} — move"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ---------------------------------------------------------------------------
# Full compare pipeline (used by CLI)
# ---------------------------------------------------------------------------


def compare(
    path_a: Path,
    path_b: Path,
    label_a: str,
    label_b: str,
    eval_path_a: Optional[Path],
    eval_path_b: Optional[Path],
    out_dir: Path,
    input_file: Optional[Path] = None,
    planning_date: Optional[str] = None,
    metric_warn_threshold_pct: float = 5.0,
    distance_method: str = DEFAULT_DISTANCE_METHOD,
) -> None:
    """Full compare pipeline: load → analyse → write outputs → print console report."""
    for p, name in [(path_a, "SOL_A"), (path_b, "SOL_B")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    pair = load_and_prepare(
        sol_a=path_a, sol_b=path_b,
        input_file=input_file, planning_date=planning_date,
        distance_method=distance_method,
        metric_warn_threshold_pct=metric_warn_threshold_pct,
    )

    eval_a = json.loads(eval_path_a.read_text(encoding="utf-8")) if eval_path_a and eval_path_a.exists() else None
    eval_b = json.loads(eval_path_b.read_text(encoding="utf-8")) if eval_path_b and eval_path_b.exists() else None

    summary_a = compute_payload_summary(pair.rows_a)
    summary_b = compute_payload_summary(pair.rows_b)
    comparison = build_comparison(label_a, label_b, summary_a, summary_b, eval_a, eval_b)
    detail_rows = build_labor_detail(label_a, label_b, pair.rows_a, pair.rows_b)

    ids_a = {r["labor_id"] for r in pair.rows_a}
    ids_b = {r["labor_id"] for r in pair.rows_b}

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv_path = out_dir / "comparison_summary.csv"
    _write_csv(summary_csv_path, ["section", "metric", label_a, label_b, "delta", "delta_pct"],
               _comparison_to_csv_rows(comparison, label_a, label_b))

    detail_csv_path = out_dir / "comparison_labor_detail.csv"
    if detail_rows:
        _write_csv(detail_csv_path, list(detail_rows[0].keys()), detail_rows)
    else:
        detail_csv_path.write_text("labor_id\n", encoding="utf-8")

    _print_report(
        label_a, label_b, comparison,
        n_a=len(pair.rows_a), n_b=len(pair.rows_b),
        n_shared=len(ids_a & ids_b), n_a_only=len(ids_a - ids_b), n_b_only=len(ids_b - ids_a),
    )
    print(f"Written: {json_path}")
    print(f"Written: {summary_csv_path}")
    print(f"Written: {detail_csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Edit this block before running:  python -m src.analysis.compare_solutions
_SOL_A          = Path("experiments/phase2/cases/file_snapshots/output_payload_scoped.json")
_SOL_B          = Path("experiments/phase2/results/output/output_payload.json")
_LABEL_A        = "baseline"
_LABEL_B        = "algorithm"
_EVAL_A: Optional[Path] = None
_EVAL_B: Optional[Path] = None
_OUT_DIR        = Path("experiments/phase2/comparisons")
_INPUT_FILE: Optional[Path] = Path("experiments/phase2/cases/input_solver.json")
_PLANNING_DATE: Optional[str] = "2026-02-18"
_METRIC_WARN_PCT: float = 5.0
_DISTANCE_METHOD: str = DEFAULT_DISTANCE_METHOD


def main() -> int:
    _root    = Path(__file__).resolve().parents[2]
    _abs     = lambda p: p if p.is_absolute() else _root / p
    _abs_opt = lambda p: None if p is None else _abs(p)
    compare(
        path_a=_abs(_SOL_A), path_b=_abs(_SOL_B),
        label_a=_LABEL_A, label_b=_LABEL_B,
        eval_path_a=_abs_opt(_EVAL_A), eval_path_b=_abs_opt(_EVAL_B),
        out_dir=_abs(_OUT_DIR),
        input_file=_abs_opt(_INPUT_FILE),
        planning_date=_PLANNING_DATE,
        metric_warn_threshold_pct=_METRIC_WARN_PCT,
        distance_method=_DISTANCE_METHOD,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
