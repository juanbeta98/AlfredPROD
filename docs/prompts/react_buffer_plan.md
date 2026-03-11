# BUFFER_REACT Implementation Plan

## Overview

BUFFER_REACT takes an existing schedule (preassigned), freezes "active" labors (those at or near their start time), and re-optimizes everything else using the same OFFLINE algorithm logic. New unassigned labors are also re-optimized in the same pass.

**Freeze rule (user-confirmed):** `freeze_cutoff = decision_time + time_previous_freeze`
Labors with `schedule_date <= freeze_cutoff` are frozen (starting within the next N minutes or already started).
**Decision time source:** Request timestamp from `request.json`, fallback to `now_colombia()`.

---

## Files to Create/Modify

### 1. `src/optimization/algorithms/buffer_react/buffer_react_algorithms.py` (NEW)

Three helper functions:

**`split_labors_by_freeze_cutoff(preassigned_labors, preassigned_moves, freeze_cutoff)`**
- Splits preassigned_labors into frozen (schedule_date ≤ freeze_cutoff AND has assigned_driver) and reassignable (rest)
- Labors in the freeze window but with no assignment go to reassignable
- Returns `(frozen_labors, reassignable_labors, frozen_moves)` where frozen_moves are moves rows whose labor_id is in the frozen set

**`build_post_freeze_driver_states(frozen_labors)`**
- For each driver in frozen_labors, finds their last frozen labor (max actual_end)
- Returns `Dict[driver_key, {"position": map_end_point_wkt, "available": actual_end_timestamp}]`
- These are partial state overrides (only position and available are changed; work_start stays from directorio)

**`strip_assignment_columns(df)`**
- Clears: `assigned_driver, actual_start, actual_end, dist_km, overtime_minutes, actual_status`
- Returns a copy of df ready for re-optimization

---

### 2. `src/optimization/algorithms/buffer_react/algorithm.py` (MAJOR REWRITE)

**`BufferReactAlgoConfig`** — adds `time_previous_freeze: int = 0` to the same fields as `OfflineAlgoConfig`.

**`BufferReactAlgorithm(OfflineAlgorithm)`** — inherits from OfflineAlgorithm to reuse:
- `_run_iterations_parallel()` — parallel/sequential iteration runner
- `_select_best_iteration()` — picks best iteration by labors covered + min distance
- `_concat_run_results()` — concat per-city results
- `_get_max_iter()` — city-specific iteration limits
- `_city_keys()` — extract location/department keys from df

Custom `__init__`: calls `OptimizationAlgorithm.__init__` (skips OfflineAlgorithm's) and builds `BufferReactAlgoConfig`.

**`solve()` — full override:**

```
1. Get decision_time from params["decision_time"] or now_colombia()
2. freeze_cutoff = decision_time + timedelta(minutes=config.time_previous_freeze)
3. Get preassigned labors/moves from params["preassigned"]
4. split_labors_by_freeze_cutoff → (frozen_labors, reassignable, frozen_moves)
5. build_post_freeze_driver_states(frozen_labors) → post_freeze_states
6. combined_df = strip_assignment_columns(reassignable) + input_df (new labors)
7. If combined_df is empty → return frozen state as-is
8. For each city_key in combined_df:
   a. Optional OSRM precompute (same as OfflineAlgorithm)
   b. Build iter_args with initial_drivers=post_freeze_states (NEW key)
   c. _run_iterations_parallel(iter_args)
   d. _select_best_iteration → run_results.append(...)
9. new_results, new_moves = _concat_run_results(run_results)
10. results_df = concat(frozen_labors, new_results)
    moves_df   = concat(frozen_moves, new_moves)
11. Return (results_df, metrics, {"moves_df": moves_df, "distance_method": ...})
```

Metrics include: frozen_labors_count, reassigned_labors_count, new_labors_count.

---

### 3. `src/optimization/algorithms/offline/offline_algorithms.py` (SMALL CHANGE)

In `run_assignment_algorithm()`, after the existing kwargs pops:

```python
initial_drivers = kwargs.pop("initial_drivers", None)
```

Change the driver initialization line from:
```python
drivers = init_drivers(df_sorted, directorio_df=directorio_df, city=city_key, **kwargs)
```
To:
```python
drivers = init_drivers(df_sorted, directorio_df=directorio_df, city=city_key, **kwargs)
if initial_drivers:
    for drv_key, override in initial_drivers.items():
        if drv_key in drivers:
            drivers[drv_key] = {**drivers[drv_key], **override}
        else:
            drivers[drv_key] = override
```

This is purely additive — OFFLINE algorithm behavior is unchanged (initial_drivers defaults to None).

---

### 4. `src/optimization/algorithms/offline/algorithm.py` (SMALL CHANGE)

**`_SHARED_KEYS`** — add `"initial_drivers"` so it's efficiently shared across parallel workers (not pickled per-task):
```python
_SHARED_KEYS = frozenset({
    "labors_df", "dist_dict", "time_dict", "directorio_df", "duraciones_df",
    "distance_method", "time_method", "alpha", "model_params", "master_data",
    "initial_drivers",  # NEW
})
```

**`_run_single_iteration()`** — pass it through to run_assignment_algorithm:
```python
results_df, moves_df, postponed_labors, dist_dict_out = run_assignment_algorithm(
    labors_df=labors_df,
    ...
    initial_drivers=args.get("initial_drivers"),  # NEW
)
```

---

### 5. `src/optimization/solver.py` (1 LINE)

After `algo_params["preassigned"] = ...`, add:
```python
algo_params["decision_time"] = self.context.get("decision_time")
```

---

### 6. `main.py` (3 LINES in context building)

In the context building block (around line 641-652), extract decision_time from request:
```python
ts_raw = request_payload.raw.get("timestamp") if request_payload else None
if ts_raw:
    try:
        context["decision_time"] = pd.Timestamp(ts_raw)
    except Exception:
        pass
```

---

## request.json Example for BUFFER_REACT

```json
{
  "request_id": "react-0001",
  "timestamp": "2026-02-18T09:00:00-05:00",
  "filters": { "department": "25", "start_date": "...", "end_date": "..." },
  "algorithm": {
    "name": "BUFFER_REACT",
    "params": {
      "time_previous_freeze": 30,
      "n_processes": -1
    }
  }
}
```

---

## Key Design Decisions

- **Inheritance from OfflineAlgorithm** — avoids copy-pasting ~150 lines of iteration machinery
- **`initial_drivers` as partial override** — only position/available are overridden; work_start stays from directorio, preserving driver schedule constraints
- **No changes to OptimizationSettings** — `time_previous_freeze` flows via the existing `overrides` mechanism (request.json → `algorithm.params` → `overrides[BUFFER_REACT]` → `for_algorithm()` → `algo_params`)
- **Frozen moves split by labor_id** — moves rows have a `labor_id` column; frozen moves = rows whose labor_id is in frozen set
- **Fallback when no preassigned** — if preassigned is empty, BUFFER_REACT behaves exactly like OFFLINE (no freezing)
