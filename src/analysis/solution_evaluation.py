"""Single-solution analysis: parsing, KPI computation, and Plotly figure builders.

Shared utilities (payload loading, geometry, timeline reconstruction) are
imported by src.analysis.compare_solutions for the two-solution case.

Public API
----------
load_payload            Load an output_payload JSON → list of service dicts.
filter_by_date          Drop labors before a planning date.
flatten_labors          Flatten service/labor hierarchy → flat labor rows.
build_coord_lookups     Compute per-labor distance and endpoint lookups.
recompute_move_distances Compute driver-move distances from coordinates.
reconstruct_timeline    Reconstruct FREE_TIME / DRIVER_MOVE / VT segments per driver.
compute_payload_summary Aggregate KPIs from flat labor rows.
build_gantt_figure      Plotly Gantt chart — driver timelines.
build_service_distance_figure  Plotly stacked bar — distance per service.
build_driver_distance_figure   Plotly stacked bar — distance per driver.
build_summary_table     Styled DataFrame from an evaluation report.
build_route_map         Folium map — single driver's geographic route.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date as _date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go

from src.optimization.common.distance_utils import distance as _compute_distance
from src.optimization.settings.model_params import ModelParams
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOGOTA_TZ = ZoneInfo("America/Bogota")

_VT_LABOR_TYPES = frozenset({"alfred_initial_transport", "alfred_transport"})

_SEGMENT_COLOR: Dict[str, str] = {
    "FREE_TIME":              "rgba(190, 190, 190, 0.40)",
    "DRIVER_MOVE":            "rgba(250, 155, 45,  0.85)",
    "VEHICLE_TRANSPORTATION": "rgba(55,  115, 200, 0.88)",
}
_SEGMENT_LABEL: Dict[str, str] = {
    "FREE_TIME":              "Free time",
    "DRIVER_MOVE":            "Driver move",
    "VEHICLE_TRANSPORTATION": "Service (labor)",
}
_SEGMENT_ORDER: List[str] = ["FREE_TIME", "DRIVER_MOVE", "VEHICLE_TRANSPORTATION"]

# Evaluation-report metrics shown in build_summary_table.
# Each entry is (dot-separated path in the eval JSON, display label).
_SUMMARY_METRICS: List[Tuple[str, str]] = [
    ("summary.services_total",                          "Services total"),
    ("summary.labors_total",                            "Labors total"),
    ("summary.drivers_used",                            "Drivers used"),
    ("summary.services_successfully_assigned",          "Services assigned"),
    ("summary.service_assignment_rate_pct",             "Service assignment rate (%)"),
    ("summary.vt_assignment_rate_pct",                  "VT labor assignment rate (%)"),
    ("summary.total_labor_distance_km",                 "Total labor distance (km)"),
    ("summary.total_driver_move_distance_km",           "Total move distance (km)"),
    ("time_allocation.labor_work_pct",                  "Labor work time (%)"),
    ("time_allocation.driver_move_pct",                 "Driver move time (%)"),
    ("time_allocation.free_time_pct",                   "Free time (%)"),
    ("utilization.system.utilization_without_moves_pct","Utilization excl. moves (%)"),
    ("utilization.system.utilization_with_moves_pct",   "Utilization incl. moves (%)"),
    ("punctuality.late_services_pct",                   "Late services (%)"),
    ("punctuality.normalized_tardiness_pct",            "Normalized tardiness (%)"),
]


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string → timezone-aware datetime in Bogotá TZ."""
    if not s:
        return None
    raw = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BOGOTA_TZ)
    return dt.astimezone(BOGOTA_TZ)


def _to_plotly_dt(dt: datetime) -> str:
    """Convert to a Bogotá-naive ISO string for Plotly (avoids UTC display shift)."""
    return dt.astimezone(BOGOTA_TZ).replace(tzinfo=None).isoformat()


# ---------------------------------------------------------------------------
# Payload loading and filtering
# ---------------------------------------------------------------------------


def load_payload(path: Path) -> List[Dict[str, Any]]:
    """Load and normalize an output_payload JSON → list of service dicts."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    raise ValueError(f"Unrecognized payload format in {path}. Expected list or dict with 'data'.")


def filter_by_date(
    services: List[Dict[str, Any]],
    planning_date: str,
) -> List[Dict[str, Any]]:
    """Drop labors scheduled before planning_date (YYYY-MM-DD); drop empty services."""
    cutoff = _date.fromisoformat(planning_date)
    result: List[Dict[str, Any]] = []
    dropped = 0
    for svc in services:
        kept = []
        for lab in svc.get("serviceLabors", []):
            dt = _parse_dt(lab.get("schedule_date"))
            if dt is not None and dt.date() < cutoff:
                dropped += 1
                continue
            kept.append(lab)
        if kept:
            result.append({**svc, "serviceLabors": kept})
    if dropped:
        print(f"[filter_by_date] Dropped {dropped} labor(s) before {planning_date}.")
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _extract_point(addr: Any) -> Optional[Tuple[float, float]]:
    """Return (lon, lat) from an address dict, or None."""
    if not isinstance(addr, dict):
        return None
    pt = addr.get("point")
    if not isinstance(pt, dict):
        return None
    try:
        return float(pt["x"]), float(pt["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _to_wkt(point: Optional[Tuple[float, float]]) -> Optional[str]:
    """Convert a (lon, lat) tuple to a WKT POINT string."""
    if point is None:
        return None
    lon, lat = point
    return f"POINT({lon} {lat})"


def build_coord_lookups(
    services: List[Dict[str, Any]],
    distance_method: str = DEFAULT_DISTANCE_METHOD,
) -> Tuple[Dict[Any, Optional[float]], Dict[Any, Tuple[Optional[str], Optional[str]]], set]:
    """Build per-labor distance and endpoint lookups from service coordinate data.

    VT labors (alfred_initial_transport / alfred_transport) cover legs between
    consecutive stops. Non-VT labors are performed at a fixed shop (distance 0).

    Returns
    -------
    coord_lookup  : labor_id -> labor_distance_km (float or None)
    points_lookup : labor_id -> (map_start_wkt, map_end_wkt)
    vt_labor_ids  : set of labor IDs classified as VT in the input
    """
    coord_lookup: Dict[Any, Optional[float]] = {}
    points_lookup: Dict[Any, Tuple[Optional[str], Optional[str]]] = {}
    vt_labor_ids: set = set()

    for svc in services:
        labs = svc.get("serviceLabors", [])
        labs_sorted = sorted(
            labs,
            key=lambda l: (l.get("labor_sequence") is None, l.get("labor_sequence") or 0),
        )

        stops: List[Optional[Tuple[float, float]]] = [_extract_point(svc.get("start_address"))]
        non_vt = [l for l in labs_sorted if l.get("labor_type") not in _VT_LABOR_TYPES]
        for lab in non_vt:
            stops.append(_extract_point(lab.get("shop_address")))
        stops.append(_extract_point(svc.get("end_address")))

        vt_labs = [l for l in labs_sorted if l.get("labor_type") in _VT_LABOR_TYPES]
        for i, vt_lab in enumerate(vt_labs):
            lid = vt_lab["id"]
            vt_labor_ids.add(lid)
            p1_wkt = _to_wkt(stops[i] if i < len(stops) else None)
            p2_wkt = _to_wkt(stops[i + 1] if i + 1 < len(stops) else None)
            if p1_wkt and p2_wkt:
                d, _ = _compute_distance(p1_wkt, p2_wkt, method=distance_method)
                coord_lookup[lid] = round(d, 4) if not math.isnan(d) else None
            else:
                coord_lookup[lid] = None
            points_lookup[lid] = (p1_wkt, p2_wkt)

        for lab in non_vt:
            coord_lookup[lab["id"]] = 0.0
            shop_pt = _to_wkt(_extract_point(lab.get("shop_address")))
            points_lookup[lab["id"]] = (shop_pt, shop_pt)

    return coord_lookup, points_lookup, vt_labor_ids


def recompute_move_distances(
    rows: List[Dict[str, Any]],
    points_lookup: Dict[Any, Tuple[Optional[str], Optional[str]]],
    distance_method: str = DEFAULT_DISTANCE_METHOD,
    driver_home_lookup: Optional[Dict[str, str]] = None,
) -> Dict[Any, float]:
    """Compute driver-move distance to each assigned labor from coordinates.

    Groups by driver, sorts chronologically, and computes the distance from
    the previous labor's endpoint to the current labor's start point.

    When driver_home_lookup is provided (mapping driver_id -> home WKT), the
    first labor of each driver uses the home position as the starting point,
    enabling home→first-service distances for solutions without addData.

    Labors whose start coordinates are absent from points_lookup are omitted
    from the returned dict so the caller retains the payload-reported value.

    Returns
    -------
    dict : labor_id -> driver_move_distance_km  (only labors with computable distances)
    """
    by_driver: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        if r["driver_id"] is None:
            continue
        by_driver.setdefault(r["driver_id"], []).append(r)

    move_dists: Dict[Any, float] = {}
    for driver_id, driver_rows in by_driver.items():
        sorted_rows = sorted(
            driver_rows,
            key=lambda r: (
                str(r.get("labor_schedule_date") or ""),
                r.get("service_labor_index", 0),
                r.get("labor_id") or 0,
            ),
        )
        prev_end_wkt: Optional[str] = (
            driver_home_lookup.get(str(driver_id)) if driver_home_lookup else None
        )
        for r in sorted_rows:
            lid = r["labor_id"]
            start_wkt, end_wkt = points_lookup.get(lid, (None, None))
            if prev_end_wkt is not None and start_wkt is not None:
                d, _ = _compute_distance(prev_end_wkt, start_wkt, method=distance_method)
                move_dists[lid] = round(d, 4) if not math.isnan(d) else 0.0
            if end_wkt is not None:
                prev_end_wkt = end_wkt

    return move_dists


def infer_missing_durations(
    rows: List[Dict[str, Any]],
    vt_speed_kmh: float,
    tiempo_alistar_min: float = 30.0,
    tiempo_finalizacion_min: float = 15.0,
) -> None:
    """Fill in duration_min and actual_end for labors where addData was absent.

    Mirrors the solver formula from assign_task_to_driver:
        duration = tiempo_alistar + labor_distance_km / vt_speed_kmh * 60
                   + (tiempo_finalizacion if last VT labor of service else 0)

    Rules:
    - tiempo_alistar_min is added to every VT labor.
    - tiempo_finalizacion_min is added only when the overall last labor of the
      service (highest service_labor_index) is itself a VT labor — mirroring the
      solver's is_last_in_service check that spans all labor types.
    - Non-VT labors are not mutated (they have labor_distance_km == 0 so the
      guard already skips them, but the VT-type check is explicit for safety).

    Only mutates rows where duration_min == 0, labor_distance_km > 0,
    actual_start is not None, labor_type is a VT type, and vt_speed_kmh > 0.
    """
    if vt_speed_kmh <= 0:
        return

    # Pre-pass: find the labor_id of the last labor of each service when that
    # last labor is a VT type → those labors receive tiempo_finalizacion.
    last_per_service: Dict[Any, Tuple[int, Any, Any]] = {}  # sid → (max_idx, labor_id, labor_type)
    for r in rows:
        sid = r["service_id"]
        idx = r["service_labor_index"]
        entry = last_per_service.get(sid)
        if entry is None or idx > entry[0]:
            last_per_service[sid] = (idx, r["labor_id"], r["labor_type"])

    last_vt_labor_ids: set = {
        lid
        for (_idx, lid, lt) in last_per_service.values()
        if lt in _VT_LABOR_TYPES
    }

    for r in rows:
        if (
            r["duration_min"] == 0
            and r["labor_distance_km"]
            and r["actual_start"] is not None
            and r["labor_type"] in _VT_LABOR_TYPES
        ):
            dist_duration = r["labor_distance_km"] / vt_speed_kmh * 60
            finalizacion = tiempo_finalizacion_min if r["labor_id"] in last_vt_labor_ids else 0.0
            total = round(dist_duration + tiempo_alistar_min + finalizacion, 2)
            r["duration_min"] = total
            r["actual_end"] = r["actual_start"] + timedelta(minutes=total)


# ---------------------------------------------------------------------------
# Labor flattening
# ---------------------------------------------------------------------------


def flatten_labors(
    services: List[Dict[str, Any]],
    coord_lookup: Optional[Dict[Any, Optional[float]]] = None,
    warn_threshold_pct: float = 5.0,
    vt_labor_ids: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Flatten the service/labor hierarchy into flat labor-level dicts.

    When coord_lookup is provided, the computed labor_distance_km is used.
    If the solution also reports a value in addData that diverges by more than
    warn_threshold_pct percent, a warning message is collected.

    Returns
    -------
    rows     : list of flat labor dicts
    warnings : list of warning strings (empty when coord_lookup is None)
    """
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for svc in services:
        service_id = svc.get("service_id")
        for lab_idx, labor in enumerate(svc.get("serviceLabors", [])):
            labor_id = labor.get("id")
            add_data: Dict[str, Any] = labor.get("addData") or {}
            alfred: Dict[str, Any] = labor.get("alfred") or {}
            driver_id = alfred.get("id")

            actual_start = _parse_dt(labor.get("schedule_date"))
            duration_min = add_data.get("duration_min") or 0.0
            actual_end = (
                actual_start + timedelta(minutes=duration_min) if actual_start else None
            )

            reported_dist = add_data.get("labor_distance_km")
            if coord_lookup is not None and labor_id in coord_lookup:
                computed_dist = coord_lookup[labor_id]
                labor_distance_km: float = (
                    computed_dist if computed_dist is not None else (reported_dist or 0.0)
                )
                if reported_dist is not None and computed_dist is not None:
                    try:
                        base = max(abs(computed_dist), 1e-9)
                        diff_pct = abs(float(reported_dist) - computed_dist) / base * 100.0
                        if diff_pct > warn_threshold_pct:
                            warnings.append(
                                f"Labor {labor_id} (service {service_id}): "
                                f"reported={float(reported_dist):.4f} km "
                                f"vs computed={computed_dist:.4f} km ({diff_pct:.1f}%)"
                            )
                    except (TypeError, ValueError):
                        pass
            else:
                labor_distance_km = reported_dist or 0.0

            labor_type = labor.get("labor_type")
            if labor_type is None and vt_labor_ids is not None:
                labor_type = "alfred_transport" if labor_id in vt_labor_ids else "__non_vt__"

            rows.append({
                "labor_id":               labor_id,
                "service_id":             service_id,
                "labor_type":             labor_type,
                "labor_schedule_date":    labor.get("schedule_date"),
                "service_labor_index":    lab_idx,
                "driver_id":              str(driver_id) if driver_id is not None else None,
                "actual_start":           actual_start,
                "actual_end":             actual_end,
                "duration_min":           duration_min,
                "labor_distance_km":      labor_distance_km,
                "driver_move_distance_km": add_data.get("driver_move_distance_km") or 0.0,
                "is_infeasible":          bool(add_data.get("is_infeasible", False)),
                "reassignment_candidate": bool(add_data.get("reassignment_candidate", False)),
            })

    return rows, warnings


# ---------------------------------------------------------------------------
# Timeline reconstruction
# ---------------------------------------------------------------------------


def reconstruct_timeline(
    rows: List[Dict[str, Any]],
    speed_kmh: float,
) -> List[Dict[str, Any]]:
    """Reconstruct a 3-segment timeline per driver from flat labor rows.

    For each assigned labor (sorted chronologically per driver):
      1. FREE_TIME              — idle gap before the next move begins
      2. DRIVER_MOVE            — travel inferred from driver_move_distance_km / speed_kmh
      3. VEHICLE_TRANSPORTATION — actual_start to actual_end

    Returns a list of segment dicts with keys:
      driver_id, labor_id, service_id, segment_type,
      start (datetime), end (datetime), duration_min, distance_km
    """
    by_driver: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["driver_id"] is not None and r["actual_start"] is not None:
            by_driver[r["driver_id"]].append(r)

    segments: List[Dict[str, Any]] = []
    for driver_id, labors in by_driver.items():
        labors_sorted = sorted(labors, key=lambda x: x["actual_start"])
        prev_end: Optional[datetime] = None

        for labor in labors_sorted:
            actual_start: datetime = labor["actual_start"]
            actual_end: datetime = labor["actual_end"]
            move_dist: float = labor["driver_move_distance_km"] or 0.0
            move_min: float = (
                move_dist / speed_kmh * 60.0 if speed_kmh > 0 and move_dist else 0.0
            )
            move_start: datetime = actual_start - timedelta(minutes=move_min)

            base = {
                "driver_id":  driver_id,
                "labor_id":   labor["labor_id"],
                "service_id": labor["service_id"],
            }

            # Temporal infeasibility: driver must start moving before prev labor ends
            # by more than 1 minute (slack absorbs floating-point / rounding noise).
            overlaps_prev = (
                prev_end is not None
                and move_min > 0
                and move_start < prev_end - timedelta(minutes=1)
            )

            if prev_end is not None and move_start > prev_end:
                gap_min = (move_start - prev_end).total_seconds() / 60.0
                segments.append({**base,
                    "segment_type": "FREE_TIME",
                    "start":        prev_end,
                    "end":          move_start,
                    "duration_min": round(gap_min, 2),
                    "distance_km":  0.0,
                })

            if move_min > 0:
                segments.append({**base,
                    "segment_type": "DRIVER_MOVE",
                    "start":        move_start,
                    "end":          actual_start,
                    "duration_min": round(move_min, 2),
                    "distance_km":  move_dist,
                    "is_infeasible": overlaps_prev,
                })

            segments.append({**base,
                "segment_type": "VEHICLE_TRANSPORTATION",
                "start":        actual_start,
                "end":          actual_end,
                "duration_min": round(labor["duration_min"], 2),
                "distance_km":  labor["labor_distance_km"],
                "is_infeasible": labor["is_infeasible"] or overlaps_prev,
            })

            prev_end = actual_end

    return segments


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def compute_payload_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate KPIs directly from flat labor rows."""
    services: set = set()
    drivers: set = set()
    total_labor_dist = total_move_dist = total_duration = 0.0
    assigned_count = infeasible_count = reassignment_count = vt_count = 0
    n_labor_dist = n_move_dist = 0

    for r in rows:
        services.add(r["service_id"])
        if r.get("labor_type") in _VT_LABOR_TYPES:
            vt_count += 1
        if r["driver_id"] is not None:
            drivers.add(r["driver_id"])
            assigned_count += 1
        if r["labor_distance_km"] and r["driver_id"] is not None:
            total_labor_dist += r["labor_distance_km"]
            n_labor_dist += 1
        if r["driver_move_distance_km"]:
            total_move_dist += r["driver_move_distance_km"]
            n_move_dist += 1
        if r["duration_min"]:
            total_duration += r["duration_min"]
        if r.get("is_infeasible"):
            infeasible_count += 1
        if r.get("reassignment_candidate"):
            reassignment_count += 1

    n = len(rows)
    return {
        "services_count":                  len(services),
        "labors_count":                    n,
        "labors_vt_count":                 vt_count,
        "drivers_count":                   len(drivers),
        "labors_assigned":                 assigned_count,
        "labors_infeasible":               infeasible_count,
        "labors_infeasible_pct":           round(100.0 * _safe_div(infeasible_count, n), 2),
        "labors_reassignment_candidate":   reassignment_count,
        "total_labor_distance_km":         round(total_labor_dist, 2),
        "total_driver_move_distance_km":   round(total_move_dist, 2),
        "total_distance_km":               round(total_labor_dist + total_move_dist, 2),
        "avg_labor_distance_km":           round(_safe_div(total_labor_dist, n_labor_dist), 2),
        "avg_driver_move_distance_km":     round(_safe_div(total_move_dist, n_move_dist), 2),
        "total_duration_min":              round(total_duration, 2),
        "avg_duration_min":                round(_safe_div(total_duration, n), 2),
    }


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def _deep_get(d: Optional[Any], dotted_key: str) -> Any:
    """Navigate a nested dict using a dot-separated key string."""
    if not isinstance(d, dict):
        return None
    for part in dotted_key.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# Figure builders — single solution
# ---------------------------------------------------------------------------


def build_gantt_figure(
    segments: List[Dict[str, Any]],
    driver_ids: List[str],
    label: str,
) -> go.Figure:
    """Plotly horizontal bar Gantt chart — driver timelines for one solution."""
    driver_set = set(driver_ids)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seg in segments:
        if seg["driver_id"] in driver_set:
            by_type[seg["segment_type"]].append(seg)

    fig = go.Figure()
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

        fig.add_trace(go.Bar(
            x=x_vals, y=y_vals, base=base_vals,
            orientation="h",
            name=_SEGMENT_LABEL[seg_type],
            marker_color=_SEGMENT_COLOR[seg_type],
            hovertext=hovers, hoverinfo="text",
            legendgroup=seg_type,
        ))

    all_starts = [s["start"] for s in segments if s["driver_id"] in driver_set]
    all_ends   = [s["end"]   for s in segments if s["driver_id"] in driver_set]
    if all_starts and all_ends:
        pad = timedelta(minutes=30)
        fig.update_layout(xaxis=dict(
            range=[_to_plotly_dt(min(all_starts) - pad), _to_plotly_dt(max(all_ends) + pad)]
        ))

    fig.update_xaxes(type="date", tickformat="%H:%M", title_text="Time (Bogotá)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(driver_ids)),
        title_text="Driver",
    )
    row_h = max(35, 600 // max(len(driver_ids), 1))
    fig.update_layout(
        title_text=f"Driver Timelines — {label}",
        barmode="overlay",
        height=max(420, row_h * len(driver_ids) + 160),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="closest",
        margin=dict(l=80, r=20, t=100, b=60),
    )
    return fig


def build_service_distance_figure(
    rows: List[Dict[str, Any]],
    all_services: List[str],
    label: str,
) -> go.Figure:
    """Plotly stacked bar — labor + move distance per service, sorted by total desc."""
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("service_id", dropna=False)
        .agg(labor_dist=("labor_distance_km", "sum"), move_dist=("driver_move_distance_km", "sum"))
        .reset_index()
    )
    agg["total_dist"] = agg["labor_dist"] + agg["move_dist"]
    agg["service_id"] = agg["service_id"].astype(str)
    svc_df = agg.set_index("service_id")

    svc_order = (
        svc_df["total_dist"].reindex(all_services, fill_value=0)
        .sort_values(ascending=False).index.tolist()
    )

    labor_vals = [round(svc_df.loc[s, "labor_dist"], 2) if s in svc_df.index else 0 for s in svc_order]
    move_vals  = [round(svc_df.loc[s, "move_dist"],  2) if s in svc_df.index else 0 for s in svc_order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Labor distance", x=svc_order, y=labor_vals,
        marker_color="rgba(55, 115, 200, 0.85)",
        hovertext=[f"Service {s}<br>Labor: {l:.2f} km" for s, l in zip(svc_order, labor_vals)],
        hoverinfo="text",
    ))
    fig.add_trace(go.Bar(
        name="Move distance", x=svc_order, y=move_vals,
        marker_color="rgba(250, 155, 45, 0.85)",
        hovertext=[f"Service {s}<br>Move: {m:.2f} km" for s, m in zip(svc_order, move_vals)],
        hoverinfo="text",
    ))
    fig.update_xaxes(title_text="Service ID", tickangle=45)
    fig.update_yaxes(title_text="Distance (km)")
    fig.update_layout(
        title_text=f"Distance per Service — {label}",
        barmode="stack", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="x", margin=dict(l=60, r=20, t=100, b=100),
    )
    return fig


def build_driver_distance_figure(
    rows: List[Dict[str, Any]],
    driver_ids: List[str],
    label: str,
) -> go.Figure:
    """Plotly stacked bar — labor + move distance per driver, sorted by total desc."""
    assigned = [r for r in rows if r["driver_id"] is not None]
    if not assigned:
        return go.Figure()
    df = pd.DataFrame(assigned)
    agg = (
        df.groupby("driver_id")
        .agg(labor_dist=("labor_distance_km", "sum"), move_dist=("driver_move_distance_km", "sum"))
        .reset_index()
    )
    agg["total_dist"] = agg["labor_dist"] + agg["move_dist"]
    drv_df = agg.set_index("driver_id")

    drv_order = (
        drv_df["total_dist"].reindex(driver_ids, fill_value=0)
        .sort_values(ascending=False).index.tolist()
    )

    labor_vals = [round(drv_df.loc[d, "labor_dist"], 2) if d in drv_df.index else 0 for d in drv_order]
    move_vals  = [round(drv_df.loc[d, "move_dist"],  2) if d in drv_df.index else 0 for d in drv_order]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Labor distance", x=drv_order, y=labor_vals,
        marker_color="rgba(55, 115, 200, 0.85)",
        hovertext=[f"Driver {d}<br>Labor: {l:.2f} km" for d, l in zip(drv_order, labor_vals)],
        hoverinfo="text",
    ))
    fig.add_trace(go.Bar(
        name="Move distance", x=drv_order, y=move_vals,
        marker_color="rgba(250, 155, 45, 0.85)",
        hovertext=[f"Driver {d}<br>Move: {m:.2f} km" for d, m in zip(drv_order, move_vals)],
        hoverinfo="text",
    ))
    fig.update_xaxes(title_text="Driver ID", tickangle=45)
    fig.update_yaxes(title_text="Distance (km)")
    fig.update_layout(
        title_text=f"Distance per Driver — {label}",
        barmode="stack", height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        hovermode="x", margin=dict(l=60, r=20, t=100, b=100),
    )
    return fig


def build_summary_table(
    eval_data: Optional[Union[Path, Dict[str, Any]]],
    label: str,
) -> Optional["pd.io.formats.style.Styler"]:
    """Styled DataFrame of KPI metrics from a solution evaluation report.

    Parameters
    ----------
    eval_data : Path to the JSON report, a pre-loaded dict, or None.
    label     : Column header for the metric values.

    Returns None (with a printed message) when no data is available.
    """
    if eval_data is None:
        print("No evaluation report provided — set EVAL to enable this section.")
        return None
    if isinstance(eval_data, Path):
        if not eval_data.exists():
            print(f"Evaluation report not found: {eval_data}")
            return None
        data: Dict[str, Any] = json.loads(eval_data.read_text(encoding="utf-8"))
    else:
        data = eval_data

    table_rows = [
        {"Metric": display, label: _fmt_value(_deep_get(data, dotted))}
        for dotted, display in _SUMMARY_METRICS
    ]
    df = pd.DataFrame(table_rows)
    return df.style.set_properties(**{"text-align": "right"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )


# ---------------------------------------------------------------------------
# Route map helpers and builder (folium)
# ---------------------------------------------------------------------------

# Folium-compatible colours for each segment type (single-solution rendering).
_ROUTE_COLOR: Dict[str, str] = {
    "VEHICLE_TRANSPORTATION": "blue",
    "DRIVER_MOVE":            "orange",
}


def _wkt_to_latlon(wkt: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse a WKT POINT(lon lat) string → (lat, lon) tuple for folium."""
    if not wkt or not isinstance(wkt, str):
        return None
    try:
        lon, lat = map(float, wkt.lstrip("POINT").strip("() ").split())
        return lat, lon
    except (ValueError, AttributeError):
        return None


def _osrm_route(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    osrm_url: Optional[str] = None,
) -> List[List[float]]:
    """Fetch road-network route geometry from OSRM between two (lat, lon) points.

    Uses the OSRM_URL environment variable when osrm_url is not provided.
    Falls back to the public OSRM demo server as a last resort.

    Returns a list of [lat, lon] pairs tracing the road route.
    Raises on network or API errors.
    """
    import os as _os
    import requests as _requests

    lat1, lon1 = p1
    lat2, lon2 = p2
    base = (
        osrm_url
        or _os.environ.get("OSRM_URL")
        or "http://router.project-osrm.org/route/v1/driving/"
    )
    url = f"{base}{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    r = _requests.get(url, timeout=10)
    r.raise_for_status()
    coords = r.json()["routes"][0]["geometry"]["coordinates"]
    return [[lat, lon] for lon, lat in coords]


def build_route_map(
    services: Optional[List[Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    driver_id: str,
    driver_home_wkt: Optional[str] = None,
    label: str = "solution",
    use_osrm: bool = True,
    zoom_start: int = 12,
    points_lookup: Optional[Dict[Any, Tuple[Optional[str], Optional[str]]]] = None,
    colors: Optional[Dict[str, str]] = None,
    map_width: Union[int, str] = "100%",
    map_height: int = 500,
    tiles: str = "CartoDB positron",
) -> Any:
    """Build a folium Figure showing a single driver's geographic route.

    Segments are coloured by type:
    - VEHICLE_TRANSPORTATION (service labor): blue  (default)
    - DRIVER_MOVE (inter-service travel): orange     (default)
    - Straight-line fallback when OSRM is unavailable: dashed line

    Parameters
    ----------
    services        : Service list from load_payload() (carries address coordinates).
                      May be None when points_lookup is provided directly.
    rows            : Flat labor rows from flatten_labors().
    driver_id       : The driver whose route to render (must match rows["driver_id"]).
    driver_home_wkt : Optional WKT POINT for the driver's home — shown as a start
                      marker and used to draw the home → first-service move leg.
    label           : Title suffix for the map.
    use_osrm        : When True, request realistic road geometry from OSRM.
    zoom_start      : Initial zoom level for the folium map.
    points_lookup   : Pre-built labor_id → (start_wkt, end_wkt) dict. When provided,
                      skips the build_coord_lookups() call (useful when reusing a
                      SolutionPair's existing lookup).
    colors          : Override the default segment colours. Expected keys:
                      "VEHICLE_TRANSPORTATION" and "DRIVER_MOVE".
    map_width       : Width of the rendered figure. Integer is treated as pixels
                      (e.g. 640 → "640px"); string is passed through as-is ("100%").
    map_height      : Height in pixels for the rendered figure.
    tiles           : Folium tile provider name. Common options:
                      "CartoDB positron" (default — light, works in most environments),
                      "OpenStreetMap", "CartoDB dark_matter".
                      Pass None for no background tiles.

    Returns
    -------
    folium.Figure  (display with IPython.display.display())
    """
    try:
        import folium
    except ImportError as exc:
        raise ImportError(
            "folium is required for route maps. Install with: pip install folium"
        ) from exc

    import os as _os

    active_colors = colors if colors is not None else _ROUTE_COLOR

    if points_lookup is None:
        if services is None:
            raise ValueError("Either services or points_lookup must be provided.")
        _, points_lookup, _ = build_coord_lookups(services)

    driver_rows = sorted(
        [
            r for r in rows
            if r["driver_id"] == str(driver_id) and r["actual_start"] is not None
        ],
        key=lambda r: r["actual_start"],
    )
    if not driver_rows:
        raise ValueError(f"No assigned labors found for driver {driver_id!r}")

    # ── Build legs (from, to, color, is_osrm_eligible) ──────────────────────
    legs: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
    # waypoints: wkt → (latlon, popup_html) for circle markers
    waypoints: Dict[str, Tuple[Tuple[float, float], str]] = {}

    prev_end_wkt: Optional[str] = driver_home_wkt

    if driver_home_wkt:
        home_ll = _wkt_to_latlon(driver_home_wkt)
        if home_ll:
            waypoints[driver_home_wkt] = (
                home_ll,
                f"<b>Home</b><br>Driver: {driver_id}",
            )

    for row in driver_rows:
        lid = row["labor_id"]
        start_wkt, end_wkt = points_lookup.get(lid, (None, None))
        if start_wkt is None and end_wkt is None:
            continue

        start_ll = _wkt_to_latlon(start_wkt)
        end_ll = _wkt_to_latlon(end_wkt)

        # ── Driver-move leg (prev position → this labor's start) ────────────
        if prev_end_wkt and start_wkt and prev_end_wkt != start_wkt:
            prev_ll = _wkt_to_latlon(prev_end_wkt)
            if prev_ll and start_ll:
                legs.append((prev_ll, start_ll, active_colors["DRIVER_MOVE"]))

        # ── Labor leg (start → end) ─────────────────────────────────────────
        is_vt = row.get("labor_type") in _VT_LABOR_TYPES
        seg_color = active_colors["VEHICLE_TRANSPORTATION"] if is_vt else active_colors["DRIVER_MOVE"]

        if start_ll and end_ll and start_wkt != end_wkt:
            legs.append((start_ll, end_ll, seg_color))

        # ── Waypoint markers ────────────────────────────────────────────────
        t_start = row["actual_start"].strftime("%H:%M") if row["actual_start"] else "—"
        t_end   = row["actual_end"].strftime("%H:%M")   if row["actual_end"]   else "—"
        popup_html = (
            f"<b>Labor {lid}</b><br>"
            f"Service: {row['service_id']}<br>"
            f"Type: {row.get('labor_type', '—')}<br>"
            f"Start: {t_start} &nbsp; End: {t_end}"
        )
        if start_ll and start_wkt:
            waypoints.setdefault(start_wkt, (start_ll, popup_html))
        if end_ll and end_wkt:
            waypoints[end_wkt] = (end_ll, popup_html)  # overwrite with latest

        prev_end_wkt = end_wkt or prev_end_wkt

    if not legs:
        raise ValueError(
            f"No coordinate data available for driver {driver_id!r}. "
            "Ensure the payload carries address coordinates."
        )

    # ── Build map ────────────────────────────────────────────────────────────
    all_latlons = [ll for (from_ll, to_ll, _) in legs for ll in (from_ll, to_ll)]
    center = [
        sum(ll[0] for ll in all_latlons) / len(all_latlons),
        sum(ll[1] for ll in all_latlons) / len(all_latlons),
    ]

    # Size the Figure (the Jupyter-display wrapper), not the Map itself.
    # Sizing the Map directly produces invalid CSS when integers are used and
    # causes the Leaflet tile layer to never load (grey background).
    _w = f"{map_width}px" if isinstance(map_width, int) else map_width
    _h = f"{map_height}px" if isinstance(map_height, int) else map_height
    fig = folium.Figure(width=_w, height=_h)
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    fig.add_child(m)

    osrm_url = _os.environ.get("OSRM_URL")

    for from_ll, to_ll, color in legs:
        route_coords = None
        if use_osrm:
            try:
                route_coords = _osrm_route(from_ll, to_ll, osrm_url)
            except Exception:
                pass
        if route_coords:
            folium.PolyLine(route_coords, color=color, weight=4, opacity=0.8).add_to(m)
        else:
            folium.PolyLine(
                [from_ll, to_ll], color=color, weight=2, dash_array="5,5"
            ).add_to(m)

    for wkt, (latlon, popup_html) in waypoints.items():
        if wkt == driver_home_wkt:
            folium.Marker(
                location=latlon,
                popup=folium.Popup(popup_html, max_width=220),
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=latlon,
                radius=6,
                color="black",
                fill=True,
                fill_color="white",
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

    return fig
