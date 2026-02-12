# Alfred Optimization Pipeline

End-to-end pipeline for ingesting service/labor data, validating it, running an optimization algorithm, and delivering results either to an API or to local files.

This repo is structured to support both:
- API mode (fetch input from ALFRED API and send results back).
- Local mode (read CSV input and write CSV/JSON outputs for development).

---

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use micromamba (recommended in this repo), run everything through `scripts/mrun`:
```bash
export ALFRED_MAMBA_ENV=AlfredEnv   # optional (default is AlfredEnv)
scripts/mrun python -V
scripts/mrun python -m pip install -r requirements.txt
```

### 2) Run locally with CSV input
```bash
export USE_API=false
export LOCAL_INPUT_FILE=./data/model_input/input.csv
export LOCAL_OUTPUT_DIR=./data/model_output
export WRITE_MODEL_SOLUTION=true
python main.py
```

Equivalent with micromamba:
```bash
scripts/mrun python main.py
```

### 3) Run against API
```bash
export USE_API=true
export API_ENDPOINT="https://your.api/endpoint"
export API_TOKEN="your-token"
export DEPARTMENT=25
export START_DATE="2024-01-01T00:00:00"
export END_DATE="2025-12-31T23:59:59"
python main.py
```

Equivalent with micromamba:
```bash
scripts/mrun python main.py
```

### API Snapshot Scripts
These helpers fetch raw API responses and write JSON snapshots under `data/api_snapshots`.
Edit the filter constants at the top of each script as needed.

```bash
python scripts/api/fetch_optimization_input.py
python scripts/api/fetch_driver_directory.py
```
or
```bash
scripts/mrun python scripts/api/fetch_optimization_input.py
scripts/mrun python scripts/api/fetch_driver_directory.py
```

Optional: provide a request file (default `request.json`) to override mode and parameters.
```bash
export REQUEST_PATH=./request.json
python main.py
```
or
```bash
export REQUEST_PATH=./request.json
scripts/mrun python main.py
```

---

## How the Pipeline Works

The main entrypoint is `main.py` and the pipeline runs in these stages:

1) **Config + logging bootstrap** (`src/config.py`, `src/logging_utils.py`)
2) **Request payload** (optional) (`src/io/request_loader.py`)
3) **Input acquisition**
   - API mode: `src/integration/client.py`
   - Local CSV: `src/io/input_loader.py`
4) **Parsing** API-style JSON -> DataFrame (`src/data/parsing/input_parser.py`)
5) **Validation** with rules (`src/data/validation`)
6) **Preassigned reconstruction** if `assigned_driver` exists (`src/optimization/common/preassigned.py`)
7) **Optimization solve** (`src/optimization/solver.py`)
8) **Format output payload** (`src/data/formatting/output_formatter.py`)
9) **Delivery / persistence**
   - Optional local artifacts: `src/io/output_writer.py`
   - Optional API POST: `src/integration/sender.py`

---

## Request Payload (Optional)

If `REQUEST_PATH` is set (defaults to `request.json`), the file can override input mode, filters, algorithm, and output path.

Example `request.json`:
```json
{
  "request_id": "sim-0001",
  "filters": {
    "department": "25",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2025-12-31T23:59:59",
    "city": "11"
  },
  "algorithm": { "name": "OFFLINE", "params": {} },
  "output": { "path": "./data/model_output" }
}
```

Behavior:
- Runtime mode (`API` vs `LOCAL`) is controlled only by `USE_API` (env var).
- `filters`: optional (department, start_date, end_date, city).
- `algorithm`: name + params passed to solver.
- `output.path`: overrides output directory or file.

---

## Local Input Format (CSV)

`src/io/input_loader.py` expects a CSV with these required columns:
```
service_id
schedule_date
start_address_id
start_address_point
end_address_id
end_address_point
labor_id
labor_type
labor_name
labor_category
```

Location columns:
- either `city` (legacy), or
- canonical code fields: `city_code`, `department_code`
- canonical name fields: `city_name`, `department_name`

Notes:
- `start_address_point` and `end_address_point` must be WKT `POINT (lon lat)`.
- Each row represents a service + labor pairing; rows are grouped by `service_id`.
- The loader emits an API-style JSON payload for the parser.

### Build Local Inputs From Raw Files
Use these scripts to generate local inputs before running `main.py` in `USE_API=false` mode:

```bash
python scripts/data_prep/build_input_csv_from_raw.py
python scripts/data_prep/build_driver_directory_csv.py
```

Defaults:
- Raw source directory: `data/model_input/raw_files`
- Generated service input: `data/model_input/input.csv`
- Generated driver directory: `data/model_input/driver_directory.csv`

---

## API Input Contract (JSON)

The parser (`src/data/parsing/input_parser.py`) expects an API-style payload like:
```json
{
  "data": [
    {
      "service_id": 1,
      "state": "scheduled",
      "created_at": "2024-02-01T12:00:00Z",
      "start_address": { "id": 10, "point": { "x": -74.1, "y": 4.6 }, "city": "149" },
      "end_address": { "id": 20, "point": { "x": -74.2, "y": 4.7 } },
      "serviceLabors": [
        {
          "id": 100,
          "labor_id": 501,
          "labor_name": "Alice",
          "labor_category": "driver",
          "schedule_date": "2024-02-02T08:00:00Z",
          "labor_sequence": 1,
          "alfred": null
        }
      ]
    }
  ]
}
```

The parser flattens this into a DataFrame with columns like:
`service_id`, `created_at`, `start_address_point`, `labor_id`, `schedule_date`, `assigned_driver`, etc.

---

## Output Formats

### API output
`src/data/formatting/output_formatter.py` builds:
```json
{
  "request_id": "...",
  "timestamp": "2025-01-01T12:00:00.000000",
  "status": "completed",
  "data": [ /* service + serviceLabors */ ],
  "metadata": { /* metrics + validation */ }
}
```

### Local output
Local artifacts are controlled by flags (independent of `USE_API`):
- `WRITE_MODEL_SOLUTION=true`
  - `data/model_output/run-*/output__ts-*.csv`
  - `data/model_output/run-*/output_payload__ts-*.json`
- `WRITE_VALIDATION_REPORTS=true`
  - `data/model_output/run-*/data_validation_invalid_rows__ts-*.csv`
  - `data/model_output/run-*/data_validation_report__ts-*.json`
  - `data/model_output/run-*/solution_validation_issues__ts-*.csv`
  - `data/model_output/run-*/solution_validation_report__ts-*.json`
- `WRITE_MODEL_SOLUTION=true`
  - `data/model_output/run-*/solution_evaluation_report__ts-*.json`

How to interpret solution performance metrics:
- `docs/SOLUTION_PERFORMANCE_REPORT.md`

Enable validation output with:
```bash
export WRITE_VALIDATION_REPORTS=true
```

Optional intermediate debug exports:
```bash
export WRITE_INTERMEDIATE_DATAFRAMES=true
export INTERMEDIATE_EXPORT_BASE_DIR=./data/intermediate_exports
```
When enabled, each run writes:
- `<base_dir>/run-<run_id>/input_df__ts-<timestamp>[__req-<request_id>].csv`
- `<base_dir>/run-<run_id>/preassigned_df__ts-<timestamp>[__req-<request_id>].csv`
- `<base_dir>/run-<run_id>/driver_directory__ts-<timestamp>[__req-<request_id>].csv`

`input_df` export reflects the dataframe right before solver creation (after validation and after preassigned split).

---

## Validation Rules

Validation happens after parsing. Current default rules include:
- Required fields: `service_id`, `labor_id`, `created_at`, `schedule_date`,
  `start_address_point`, `labor_name`, `end_address_point`
- Non-empty rows
- Unique `labor_id`
- `created_at` at least 2 hours before `schedule_date`
- location key (`department_name-city_name`) must be in allowed list (hardcoded in `main.py`)

When validation fails:
- Invalid rows are separated out.
- A summary report is generated.
- Execution aborts if all rows are invalid.

---

## Algorithms

Registered algorithms are defined in `src/optimization/algorithms/registry.py`:

| Name         | Status            | Notes |
|--------------|-------------------|-------|
| OFFLINE      | Implemented       | Baseline algorithm; supports multi-city iteration. |
| INSERT       | Stub              | Not implemented (raises `NotImplementedError`). |
| BUFFER_FIXED | Stub              | Not implemented (raises `NotImplementedError`). |
| BUFFER_REACT | Stub              | Not implemented (raises `NotImplementedError`). |
| ALFRED       | Stub              | Not implemented (raises `NotImplementedError`). |

Algorithm selection comes from `OptimizationSettings` or request payload:
- Default: `OFFLINE`
- Override per algorithm via `request.json` or settings overrides.

---

## Master Data

Algorithms load master data once per run:
- `data/master_data/directorio.{parquet|csv}`
- `data/master_data/duraciones.{parquet|csv}`
- `data/master_data/dist_dict.{parquet|pkl}`

`src/data/loading/master_data_loader.py` prefers parquet when available.

---

## Configuration

Environment variables (from `.env` or shell):

Core:
- `USE_API` (true/false)
- `API_ENDPOINT`
- `API_TOKEN`
- `REQUEST_PATH` (default `request.json`)

Filters:
- `DEPARTMENT`
- `START_DATE` (ISO datetime)
- `END_DATE` (ISO datetime)

Local mode:
- `LOCAL_INPUT_DIR`
- `LOCAL_INPUT_FILE`
- `LOCAL_DRIVER_DIRECTORY_FILE`
- `LOCAL_OUTPUT_DIR`
- `LOCAL_OUTPUT_FILE`

Execution:
- `REQUEST_TIMEOUT`
- `API_MAX_RETRIES`
- `LOG_LEVEL`
- `WRITE_VALIDATION_REPORTS`
- `WRITE_INTERMEDIATE_DATAFRAMES` (true/false)
- `INTERMEDIATE_EXPORT_BASE_DIR` (default `./data/intermediate_exports`)
- `WRITE_MODEL_SOLUTION` (true/false; writes solver CSV + formatted payload JSON)

Backward compatibility:
- `EXPORT_INTERMEDIATE_DATAFRAMES` is still accepted as an alias of `WRITE_INTERMEDIATE_DATAFRAMES`.

Model params:
- `OSRM_URL` (optional; used by model params)

---

## Known Behaviors and Caveats

- The solver currently normalizes all `schedule_date` values to the first row's date
  (see `src/optimization/solver.py`), which enforces a single planning day.
- Several algorithms are placeholders and will raise `NotImplementedError`.
- In API mode, failures attempt a best-effort error report back to the API.

---

## Repo Layout

```
.
├── main.py
├── src/
│   ├── config.py
│   ├── logging_utils.py
│   ├── integration/         # API client + sender
│   ├── io/                  # request/input/output loaders
│   ├── data/                # parsing, formatting, validation
│   └── optimization/        # solver + algorithms + settings
├── data/
│   ├── model_input/
│   ├── model_output/
│   └── master_data/
└── scripts/
```

---

## Development Tips

- Use `request.json` to experiment with algorithm params and filters quickly.
- Write validation reports to inspect bad rows.
- Keep master data files in `data/master_data` or override in settings if needed.

---

## License

See `LICENSE`.
