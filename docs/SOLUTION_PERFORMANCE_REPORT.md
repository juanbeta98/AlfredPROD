# Solution Performance Report Guide

This document explains how to interpret the local artifact:

- `data/model_output/run-*/solution_evaluation_report__ts-*.json`

The report is generated in `main.py` during the `evaluate_solution` step and built by:

- `src/optimization/evaluation/solution_evaluator.py`

## Scope

The report is for **solution performance analysis** (local artifact), not for solver feasibility.

- Feasibility checks are covered by `solution_validation_report__*.json`.
- Payload KPIs (`output_payload__*.json`) remain **labor-level** in `addData`:
  - `labor_distance_km`
  - `driver_move_distance_km`

## Report Structure

Top-level keys:

- `summary`
- `assignment`
- `distance`
- `time_allocation`
- `punctuality`
- `utilization`
- `quality_checks`

`summary` is a compact view built from the detailed sections.

---

## 1) `summary`

High-level KPIs for quick monitoring.

- `services_total`: unique number of services in the solution.
- `labors_total`: total number of labor rows in the solution.
- `drivers_used`: unique assigned drivers in the solution.
- `services_with_vt_labors`: services that contain at least one `VEHICLE_TRANSPORTATION` labor.
- `services_successfully_assigned`: services where all VT labors have `assigned_driver`.
- `services_unassigned_or_failed`: `services_with_vt_labors - services_successfully_assigned`.
- `failed_services_total`: services with at least one VT labor in `FAILED` status.
- `vt_labors_total`: total VT labor count.
- `vt_labors_assigned`: VT labors with non-null `assigned_driver`.
- `vt_labors_unassigned`: VT labors without `assigned_driver`.
- `failed_labors_total`: all labor rows with `actual_status == FAILED`.
- `unassigned_labors_total`: labor rows with missing `assigned_driver`.
- `total_labor_distance_km`: sum of VT labor distances (`labor_distance_km`).
- `total_driver_move_distance_km`: sum of driver displacement distances (`driver_move_distance_km`).
- `service_assignment_rate_pct`: `services_successfully_assigned / services_with_vt_labors * 100`.
- `vt_assignment_rate_pct`: `vt_labors_assigned / vt_labors_total * 100`.
- `utilization_without_moves_pct`: system-level utilization considering only labor work time.
- `utilization_with_moves_pct`: system-level utilization considering labor work + driver moves.
- `driver_move_utilization_pct`: share of total reference window spent on driver moves.

---

## 2) `assignment`

Assignment and failure metrics (same core values used in `summary`):

- service-level assignment success/failure
- VT labor assignment coverage
- failed/unassigned counts
- assignment rates

Use this section to track coverage quality independently from timing/distance.

---

## 3) `distance`

Distance-focused KPIs:

- `total_labor_distance_km`: total VT labor kilometers.
- `total_driver_move_distance_km`: total kilometers drivers moved between labors.
- `avg_vt_labor_distance_km`: average VT labor distance.
- `avg_driver_move_distance_km_per_labor`: average driver move distance per labor row.

Interpretation:

- High `total_driver_move_distance_km` with normal `total_labor_distance_km` can indicate poor route continuity.
- Track ratios over time (example: `driver_move / labor_distance`).

---

## 4) `time_allocation`

Built from `moves_df` durations:

- `timeline_total_min`: total time in movement timeline.
- `free_time_min`: sum of `FREE_TIME` duration.
- `driver_move_min`: sum of `DRIVER_MOVE` duration.
- `labor_work_min`: remaining timeline time for actual labor rows.
- `%` fields are each component divided by `timeline_total_min`.

Interpretation:

- High `free_time_pct`: idle capacity.
- High `driver_move_pct`: excessive deadheading/mobility overhead.
- Higher `labor_work_pct`: more time spent executing work.

---

## 5) `punctuality`

Service punctuality based on first VT labor per service:

- `grace_minutes`: tolerance window from model params.
- `services_considered`: services with valid first-VT schedule and actual start.
- `late_services_count`: first VT starts after `schedule_date + grace_minutes`.
- `late_services_pct`: late services percentage.
- `total_lateness_min`: total positive lateness minutes.
- `avg_lateness_min_late_only`: lateness average over late services only.
- `avg_lateness_min_all_considered`: lateness average over all considered services.
- `normalized_tardiness_pct`: normalized lateness relative to grace capacity.

Interpretation:

- `late_services_pct` shows frequency.
- `total_lateness_min` and averages show severity.

---

## 6) `utilization`

This section uses the corrected reference window logic with driver shift times from directory:

- shift start/end come from driver directory (`start_time`, `end_time`).
- for each driver/day:
  - `start_working`: first active move (excluding `FREE_TIME`)
  - `end_working`: last active move
  - `reference_window = [min(start_working, shift_start), max(end_working, shift_end)]`

### `utilization.system`

- `drivers_considered`: unique drivers included.
- `driver_day_rows`: number of driver-day records.
- `total_reference_window_min`: sum of reference windows.
- `total_labor_work_min`: sum of labor work minutes.
- `total_driver_move_min`: sum of move minutes.
- `total_active_work_min`: labor + move minutes.
- `utilization_without_moves_pct`: `total_labor_work_min / total_reference_window_min * 100`.
- `utilization_with_moves_pct`: `total_active_work_min / total_reference_window_min * 100`.
- `driver_move_utilization_pct`: `total_driver_move_min / total_reference_window_min * 100`.
- `driver_move_share_of_active_pct`: `total_driver_move_min / total_active_work_min * 100`.

### `utilization.driver_distribution`

Distribution stats (avg, p50, p90) for:

- `utilization_without_moves_pct`
- `utilization_with_moves_pct`

### `utilization.drivers`

Per driver/day breakdown:

- `shift_window_found`: whether shift times were available.
- `reference_window_min`, `labor_work_min`, `driver_move_min`, `active_work_min`
- utilization percentages for each driver/day.

Practical interpretation for productivity:

- Usually desirable:
  - higher `utilization_without_moves_pct`
  - lower `driver_move_utilization_pct`
  - lower `driver_move_share_of_active_pct`
- `utilization_with_moves_pct` should be interpreted together with move-share metrics.

---

## 7) `quality_checks`

Data quality and structural sanity checks:

- `moves_rows`: row count in moves timeline.
- `missing_driver_move_distance_rows`: `DRIVER_MOVE` rows without usable distance.
- `negative_duration_rows`: rows with negative duration.
- `assigned_vt_labors_without_driver_move_row`: assigned VT labors missing corresponding move row.
- `assigned_drivers_missing_from_directory`: assigned drivers not found in directory.

These checks help validate metric reliability.

---

## Precision and Units

- Distances: kilometers (`km`)
- Durations: minutes (`min`)
- Percentages: `0-100`
- KPI values are rounded to 2 decimals.

---

## Known Current Caveat

There is temporary date coercion in solver (`src/optimization/solver.py`) currently kept on purpose.
This can affect cross-day interpretations. For now, interpret KPIs as run-level diagnostics under current solver behavior.
