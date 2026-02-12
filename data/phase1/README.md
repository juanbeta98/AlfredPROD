# Phase 1 Artifacts

This directory contains API-vs-local parity artifacts for the experimentation run.

## Top-Level Files

- `api_snapshot.csv`
  - Flattened API payload snapshot across all departments included in the request.
  - One row per labor.
- `input.csv`
  - Local CSV used for comparison (your provided full local services snapshot).
  - One row per labor.
- `comparison_union_api_vs_local.csv`
  - Full outer join by `labor_id` between `api_snapshot.csv` and `input.csv`.
  - Includes presence flags (`both`, `api_only`, `local_only`), side-by-side fields (`*_api`, `*_local`), and per-field mismatch flags (`mismatch__*`).
  - Datetime values are normalized to Colombia timezone (`America/Bogota`, `-05:00`).
  - Datetime mismatch checks are done at minute precision.
- `comparison_summary_api_vs_local.csv`
  - Aggregate counts and percentages for labor/service overlap.
- `comparison_field_mismatches_api_vs_local.csv`
  - Per-field mismatch counts over shared labors only.

## Per-Department Folders

Each department has:

- `data/phase1/{department_code}/api`
- `data/phase1/{department_code}/local`

Each execution-mode folder stores:

- `payload_snapshot.json`: payload snapshot used for that department run.
- `input_full.csv`: department/date-window local baseline.
- `input.csv`: filtered local input aligned to payload labor IDs.
- `request.json`: request used in that run.
- `run.log`: execution log for that mode.
- `intermediate_exports/`: copied intermediate reports from the run.
- `model_output/`: copied output artifacts from the run.
- `run_error.txt` (only when run fails).

Each department root (`data/phase1/{department_code}`) has a `README.md` with removed/missing labor analysis.

## Regenerating Comparison Files

From repository root:

```bash
python3 scripts/analysis/generate_phase1_api_local_comparison.py
```

Or with explicit paths:

```bash
python3 scripts/analysis/generate_phase1_api_local_comparison.py \
  --api-csv data/phase1/api_snapshot.csv \
  --local-csv data/phase1/input.csv \
  --out-dir data/phase1
```

## Interpretation Notes

- `api_only` labor rows: present in API snapshot but absent in local CSV.
- `local_only` labor rows: present in local CSV but absent in API snapshot.
- `mismatch__<field> = True`: field differs for a labor present in both sources.
- For date/datetime fields, second/millisecond differences are ignored (minute precision).
