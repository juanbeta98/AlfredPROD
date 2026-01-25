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

### 2) Run locally with CSV input
```bash
export USE_API=false
export LOCAL_INPUT_FILE=./data/model_input/input.csv
export LOCAL_OUTPUT_DIR=./data/model_output
python main.py
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

Optional: provide a request file (default `request.json`) to override mode and parameters.
```bash
export REQUEST_PATH=./request.json
python main.py
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
8) **Format output** (API payload) (`src/data/formatting/output_formatter.py`)
9) **Delivery**
   - API mode: `src/integration/sender.py`
   - Local mode: `src/io/output_writer.py`

---

## Request Payload (Optional)

If `REQUEST_PATH` is set (defaults to `request.json`), the file can override input mode, filters, algorithm, and output path.

Example `request.json`:
```json
{
  "request_id": "sim-0001",
  "input": { "source": "api", "path": null },
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
- `input.source`: `"api"` or `"local"`, overrides `USE_API`.
- `input.path`: local CSV path when `source=local`.
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
city
labor_id
labor_type
labor_name
labor_category
```

Notes:
- `start_address_point` and `end_address_point` must be WKT `POINT (lon lat)`.
- Each row represents a service + labor pairing; rows are grouped by `service_id`.
- The loader emits an API-style JSON payload for the parser.

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
Local mode writes:
- `data/model_output/output_<timestamp>.csv`
- Optional validation artifacts:
  - `data/model_output/invalid_rows.csv`
  - `data/model_output/validation_report.json`

Enable validation output with:
```bash
export WRITE_VALIDATION_REPORTS=true
```

---

## Validation Rules

Validation happens after parsing. Current default rules include:
- Required fields: `service_id`, `labor_id`, `created_at`, `schedule_date`,
  `start_address_point`, `labor_name`, `end_address_point`
- Non-empty rows
- Unique `labor_id`
- `created_at` at least 2 hours before `schedule_date`
- `city` must be in allowed list (hardcoded in `main.py`)

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

`src/data/master_data_loader.py` prefers parquet when available.

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
- `LOCAL_OUTPUT_DIR`
- `LOCAL_OUTPUT_FILE`

Execution:
- `REQUEST_TIMEOUT`
- `API_MAX_RETRIES`
- `LOG_LEVEL`
- `WRITE_VALIDATION_REPORTS`

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
