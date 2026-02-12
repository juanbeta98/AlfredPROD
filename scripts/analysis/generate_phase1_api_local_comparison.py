"""
Generate API vs local comparison artifacts for Phase 1.

Outputs under data/phase1:
  - comparison_union_api_vs_local.csv
  - comparison_summary_api_vs_local.csv
  - comparison_field_mismatches_api_vs_local.csv

Rules:
  - Outer join by labor_id.
  - Common-field values are emitted side by side (<field>_api, <field>_local).
  - Datetime-like fields are normalized to Colombia timezone (America/Bogota).
  - Datetime mismatch checks are done at minute precision (ignore seconds/millis).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo


BOGOTA_TZ = ZoneInfo("America/Bogota")


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
PHASE1_DIR = ROOT / "data" / "phase1"
DEFAULT_API_CSV = PHASE1_DIR / "api_snapshot.csv"
DEFAULT_LOCAL_CSV = PHASE1_DIR / "input.csv"


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            normalized = {k: ("" if v is None else str(v).strip()) for k, v in row.items()}
            rows.append(normalized)
        header = list(reader.fieldnames or [])
    return rows, header


def _is_datetime_field(field: str) -> bool:
    if "date" in field:
        return True
    if field.endswith("_at"):
        return True
    return field in {"actual_start", "actual_end", "labor_created_at"}


def _parse_dt(value: str) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=BOGOTA_TZ)
    return dt.astimezone(BOGOTA_TZ)


def _normalize_dt_string(value: str) -> str:
    dt = _parse_dt(value)
    if dt is None:
        return (value or "").strip()
    return dt.isoformat()


def _minute_bucket(value: str) -> datetime | None:
    dt = _parse_dt(value)
    if dt is None:
        return None
    return dt.replace(second=0, microsecond=0)


def _normalize_scalar(value: str) -> str:
    return (value or "").strip()


@dataclass(frozen=True)
class CompareResult:
    equal: bool
    api_value_for_output: str
    local_value_for_output: str


def _compare_field(field: str, api_value: str, local_value: str) -> CompareResult:
    api_raw = _normalize_scalar(api_value)
    local_raw = _normalize_scalar(local_value)

    if _is_datetime_field(field):
        api_out = _normalize_dt_string(api_raw)
        local_out = _normalize_dt_string(local_raw)

        api_min = _minute_bucket(api_raw)
        local_min = _minute_bucket(local_raw)

        if api_min is not None and local_min is not None:
            equal = api_min == local_min
        elif api_out == "" and local_out == "":
            equal = True
        else:
            equal = api_out == local_out
        return CompareResult(equal=equal, api_value_for_output=api_out, local_value_for_output=local_out)

    return CompareResult(
        equal=api_raw == local_raw,
        api_value_for_output=api_raw,
        local_value_for_output=local_raw,
    )


def _ordered_union(values_a: Iterable[str], values_b: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in list(values_a) + list(values_b):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def generate_comparison(api_csv: Path, local_csv: Path, out_dir: Path) -> None:
    api_rows, api_header = _read_csv(api_csv)
    local_rows, local_header = _read_csv(local_csv)

    if "labor_id" not in api_header or "labor_id" not in local_header:
        raise ValueError("Both input files must contain labor_id.")

    api_by_labor = {row["labor_id"]: row for row in api_rows if row.get("labor_id")}
    local_by_labor = {row["labor_id"]: row for row in local_rows if row.get("labor_id")}

    labor_ids = sorted(_ordered_union(api_by_labor.keys(), local_by_labor.keys()))

    api_fields = [c for c in api_header if c != "labor_id"]
    local_fields = [c for c in local_header if c != "labor_id"]
    common_fields = [c for c in api_fields if c in set(local_fields)]
    api_only_fields = [c for c in api_fields if c not in set(local_fields)]
    local_only_fields = [c for c in local_fields if c not in set(api_fields)]

    union_fieldnames = [
        "labor_id",
        "presence",
        "present_api_flag",
        "present_local_flag",
        "any_field_mismatch",
        "mismatch_field_count",
    ]
    for field in common_fields:
        union_fieldnames.extend([f"{field}_api", f"{field}_local", f"mismatch__{field}"])
    for field in api_only_fields:
        union_fieldnames.append(f"{field}_api")
    for field in local_only_fields:
        union_fieldnames.append(f"{field}_local")

    union_rows: List[Dict[str, str]] = []
    shared_rows = 0
    shared_with_mismatch = 0

    field_shared_counts = {field: 0 for field in common_fields}
    field_mismatch_counts = {field: 0 for field in common_fields}

    for labor_id in labor_ids:
        api_row = api_by_labor.get(labor_id)
        local_row = local_by_labor.get(labor_id)
        in_api = api_row is not None
        in_local = local_row is not None

        if in_api and in_local:
            presence = "both"
            shared_rows += 1
        elif in_api:
            presence = "api_only"
        else:
            presence = "local_only"

        out_row: Dict[str, str] = {
            "labor_id": labor_id,
            "presence": presence,
            "present_api_flag": "1" if in_api else "0",
            "present_local_flag": "1" if in_local else "0",
            "any_field_mismatch": "False",
            "mismatch_field_count": "0",
        }

        mismatch_count = 0
        for field in common_fields:
            api_value = api_row.get(field, "") if api_row else ""
            local_value = local_row.get(field, "") if local_row else ""
            compared = _compare_field(field, api_value, local_value)
            out_row[f"{field}_api"] = compared.api_value_for_output
            out_row[f"{field}_local"] = compared.local_value_for_output

            mismatch = in_api and in_local and not compared.equal
            out_row[f"mismatch__{field}"] = "True" if mismatch else "False"

            if in_api and in_local:
                field_shared_counts[field] += 1
                if mismatch:
                    field_mismatch_counts[field] += 1
                    mismatch_count += 1

        for field in api_only_fields:
            value = api_row.get(field, "") if api_row else ""
            if _is_datetime_field(field):
                value = _normalize_dt_string(value)
            out_row[f"{field}_api"] = _normalize_scalar(value)
        for field in local_only_fields:
            value = local_row.get(field, "") if local_row else ""
            if _is_datetime_field(field):
                value = _normalize_dt_string(value)
            out_row[f"{field}_local"] = _normalize_scalar(value)

        if in_api and in_local and mismatch_count > 0:
            shared_with_mismatch += 1
            out_row["any_field_mismatch"] = "True"
            out_row["mismatch_field_count"] = str(mismatch_count)

        union_rows.append(out_row)

    union_path = out_dir / "comparison_union_api_vs_local.csv"
    _write_csv(union_path, union_fieldnames, union_rows)

    api_labor_ids = set(api_by_labor)
    local_labor_ids = set(local_by_labor)
    shared_labor_ids = api_labor_ids & local_labor_ids
    api_only_labor_ids = api_labor_ids - local_labor_ids
    local_only_labor_ids = local_labor_ids - api_labor_ids

    api_service_ids = {api_by_labor[k].get("service_id", "") for k in api_labor_ids if api_by_labor[k].get("service_id")}
    local_service_ids = {
        local_by_labor[k].get("service_id", "") for k in local_labor_ids if local_by_labor[k].get("service_id")
    }

    summary_rows = [
        {"category": "labors", "metric": "api_total", "value": str(float(len(api_labor_ids)))},
        {"category": "labors", "metric": "local_total", "value": str(float(len(local_labor_ids)))},
        {"category": "labors", "metric": "shared", "value": str(float(len(shared_labor_ids)))},
        {"category": "labors", "metric": "api_only", "value": str(float(len(api_only_labor_ids)))},
        {"category": "labors", "metric": "local_only", "value": str(float(len(local_only_labor_ids)))},
        {
            "category": "labors",
            "metric": "shared_pct_of_api",
            "value": f"{(100.0 * len(shared_labor_ids) / len(api_labor_ids)) if api_labor_ids else 0.0:.2f}",
        },
        {
            "category": "labors",
            "metric": "shared_pct_of_local",
            "value": f"{(100.0 * len(shared_labor_ids) / len(local_labor_ids)) if local_labor_ids else 0.0:.2f}",
        },
        {
            "category": "labors",
            "metric": "shared_with_any_field_mismatch",
            "value": str(float(shared_with_mismatch)),
        },
        {"category": "services", "metric": "api_total", "value": str(float(len(api_service_ids)))},
        {"category": "services", "metric": "local_total", "value": str(float(len(local_service_ids)))},
        {"category": "services", "metric": "shared", "value": str(float(len(api_service_ids & local_service_ids)))},
        {"category": "services", "metric": "api_only", "value": str(float(len(api_service_ids - local_service_ids)))},
        {"category": "services", "metric": "local_only", "value": str(float(len(local_service_ids - api_service_ids)))},
    ]

    summary_path = out_dir / "comparison_summary_api_vs_local.csv"
    _write_csv(summary_path, ["category", "metric", "value"], summary_rows)

    field_rows: List[Dict[str, str]] = []
    for field in common_fields:
        shared = field_shared_counts[field]
        mismatches = field_mismatch_counts[field]
        pct = (100.0 * mismatches / shared) if shared else 0.0
        field_rows.append(
            {
                "field": field,
                "shared_rows": str(shared),
                "mismatch_rows": str(mismatches),
                "mismatch_pct": f"{pct:.2f}",
            }
        )
    field_rows.sort(key=lambda r: (-int(r["mismatch_rows"]), r["field"]))

    field_mismatch_path = out_dir / "comparison_field_mismatches_api_vs_local.csv"
    _write_csv(
        field_mismatch_path,
        ["field", "shared_rows", "mismatch_rows", "mismatch_pct"],
        field_rows,
    )

    print(f"Wrote {union_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {field_mismatch_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate comparison_union/comparison_summary/comparison_field_mismatches for phase1."
    )
    parser.add_argument(
        "--api-csv",
        type=Path,
        default=DEFAULT_API_CSV,
        help="Path to API snapshot CSV.",
    )
    parser.add_argument(
        "--local-csv",
        type=Path,
        default=DEFAULT_LOCAL_CSV,
        help="Path to local input CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PHASE1_DIR,
        help="Output directory for generated CSVs.",
    )
    args = parser.parse_args()

    api_csv = args.api_csv if args.api_csv.is_absolute() else (ROOT / args.api_csv)
    local_csv = args.local_csv if args.local_csv.is_absolute() else (ROOT / args.local_csv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)

    if not api_csv.exists():
        raise FileNotFoundError(f"API CSV not found: {api_csv}")
    if not local_csv.exists():
        raise FileNotFoundError(f"Local CSV not found: {local_csv}")

    generate_comparison(api_csv=api_csv, local_csv=local_csv, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
