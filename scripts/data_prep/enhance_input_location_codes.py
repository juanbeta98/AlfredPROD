"""
Enhance local model input CSV with normalized location code columns.

Behavior:
1) Rename input column `city` to `city_code` (if needed).
2) Add/refresh `department_code` by mapping from data/master_data/cities.csv.
3) Backup the original input file and write the enhanced CSV as `input.csv`.

Usage:
    python3 scripts/data_prep/enhance_input_location_codes.py --apply
    python3 scripts/data_prep/enhance_input_location_codes.py --apply \
        --input-path data/model_input/input.csv \
        --cities-path data/master_data/cities.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
DEFAULT_INPUT_PATH = ROOT / "data" / "model_input" / "input.csv"
DEFAULT_CITIES_PATH = ROOT / "data" / "master_data" / "cities.csv"


@dataclass
class MappingStats:
    total_rows: int
    mapped_rows: int
    unmatched_rows: int
    unmatched_city_codes: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename input 'city' to 'city_code', add 'department_code' from cities.csv, "
            "backup old input.csv, and write enhanced input.csv."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to input CSV to enhance.",
    )
    parser.add_argument(
        "--cities-path",
        type=Path,
        default=DEFAULT_CITIES_PATH,
        help="Path to cities master CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only print what would happen.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _load_department_mapping(cities_path: Path) -> dict[str, str]:
    cities_df = pd.read_csv(cities_path, dtype=str, keep_default_na=False)

    required_columns = {"city", "data"}
    missing = required_columns - set(cities_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in cities CSV: {sorted(missing)}"
        )

    mapping: dict[str, str] = {}
    for _, row in cities_df.iterrows():
        city_key = str(row["city"]).strip()
        if city_key == "":
            continue
        payload_raw = row["data"]
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in cities.csv data column for city='{city_key}'"
            ) from exc

        department_code = payload.get("department_code")
        mapping[city_key] = "" if department_code is None else str(department_code)

    return mapping


def _enhance_input_df(
    input_df: pd.DataFrame,
    *,
    department_mapping: dict[str, str],
) -> tuple[pd.DataFrame, MappingStats]:
    df = input_df.copy()

    if "city" in df.columns and "city_code" not in df.columns:
        df = df.rename(columns={"city": "city_code"})
    elif "city" in df.columns and "city_code" in df.columns:
        # Keep canonical column and drop legacy alias.
        df = df.drop(columns=["city"])

    if "city_code" not in df.columns:
        raise ValueError("Input CSV must contain either 'city' or 'city_code' column.")

    city_codes = df["city_code"].astype("string").fillna("").str.strip()
    mapped_department = city_codes.map(department_mapping).fillna("")

    if "department_code" in df.columns:
        existing_department = df["department_code"].astype("string").fillna("")
        df["department_code"] = mapped_department.where(
            mapped_department.str.strip().ne(""),
            existing_department,
        )
    else:
        insert_at = df.columns.get_loc("city_code") + 1
        df.insert(insert_at, "department_code", mapped_department)

    unmatched_mask = city_codes.ne("") & ~city_codes.isin(pd.Index(department_mapping.keys()))
    unmatched_city_codes = (
        city_codes.loc[unmatched_mask]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    stats = MappingStats(
        total_rows=len(df),
        mapped_rows=int((city_codes.ne("") & mapped_department.ne("")).sum()),
        unmatched_rows=int(unmatched_mask.sum()),
        unmatched_city_codes=[str(v) for v in unmatched_city_codes],
    )

    return df, stats


def _build_backup_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    backup_name = f"{stem}__backup_{timestamp}{suffix}"
    backup_path = input_path.with_name(backup_name)

    idx = 1
    while backup_path.exists():
        backup_name = f"{stem}__backup_{timestamp}_{idx}{suffix}"
        backup_path = input_path.with_name(backup_name)
        idx += 1

    return backup_path


def main() -> int:
    args = _parse_args()

    input_path = _resolve_path(args.input_path)
    cities_path = _resolve_path(args.cities_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not cities_path.exists():
        raise FileNotFoundError(f"Cities CSV not found: {cities_path}")

    input_df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    department_mapping = _load_department_mapping(cities_path)
    enhanced_df, stats = _enhance_input_df(
        input_df,
        department_mapping=department_mapping,
    )

    backup_path = _build_backup_path(input_path)

    print(f"input_path: {input_path}")
    print(f"cities_path: {cities_path}")
    print(f"rows: {stats.total_rows}")
    print(f"mapped_department_rows: {stats.mapped_rows}")
    print(f"unmatched_rows: {stats.unmatched_rows}")
    print(f"unmatched_city_codes: {stats.unmatched_city_codes}")
    print(f"backup_path: {backup_path}")

    if not args.apply:
        print("dry_run: true (re-run with --apply to write files)")
        return 0

    original_renamed = False
    try:
        input_path.rename(backup_path)
        original_renamed = True
        enhanced_df.to_csv(input_path, index=False)
    except Exception:
        # Best-effort rollback if writing the new file fails after renaming.
        if original_renamed and backup_path.exists() and not input_path.exists():
            backup_path.rename(input_path)
        raise

    print("dry_run: false")
    print(f"old_file_renamed_to: {backup_path}")
    print(f"enhanced_file_written_to: {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
