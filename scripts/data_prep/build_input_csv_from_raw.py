"""
Build local `data/model_input/input.csv` from raw service CSV files.

This script is intentionally independent from runtime (`main.py`):
run it only when you need to refresh local input from raw extracts.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
DEFAULT_RAW_DIR = ROOT / "data" / "model_input" / "raw_files"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "model_input" / "input.csv"
BOGOTA_TZ = "America/Bogota"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_bogota_datetime(value: Any) -> Any:
    if _is_missing(value):
        return pd.NA

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return value

    if getattr(parsed, "tzinfo", None) is None:
        parsed = parsed.tz_localize(BOGOTA_TZ)
    else:
        parsed = parsed.tz_convert(BOGOTA_TZ)

    return parsed.isoformat()


def _require_columns(df: pd.DataFrame, *, name: str, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _build_address_lookups(address_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shop_lookup = (
        address_df.loc[address_df["shop"].notna(), ["shop", "address_id", "point"]]
        .drop_duplicates(subset=["shop"], keep="first")
        .rename(columns={"address_id": "shop_address_id", "point": "shop_address_point"})
    )
    alfred_lookup = (
        address_df.loc[address_df["alfred"].notna(), ["alfred", "address_id", "point"]]
        .drop_duplicates(subset=["alfred"], keep="first")
        .rename(columns={"address_id": "alfred_address_id", "point": "alfred_address_point"})
    )
    return shop_lookup, alfred_lookup


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def build_input_csv(
    *,
    raw_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    services_path = raw_dir / "services.csv"
    service_labors_path = raw_dir / "service_labors.csv"
    address_path = raw_dir / "address.csv"

    if not services_path.exists():
        raise FileNotFoundError(f"Missing file: {services_path}")
    if not service_labors_path.exists():
        raise FileNotFoundError(f"Missing file: {service_labors_path}")
    if not address_path.exists():
        raise FileNotFoundError(f"Missing file: {address_path}")

    services_df = pd.read_csv(services_path, low_memory=False)
    service_labors_df = pd.read_csv(service_labors_path, low_memory=False)
    address_df = pd.read_csv(address_path, low_memory=False)

    _require_columns(
        services_df,
        name="services.csv",
        required={
            "service_id",
            "created_at",
            "schedule_date",
            "start_address_id",
            "start_address_point",
            "end_address_id",
            "end_address_point",
            "city",
            "department_code",
        },
    )
    _require_columns(
        service_labors_df,
        name="service_labors.csv",
        required={
            "service_id",
            "labor_id",
            "labor_type",
            "labor_name",
            "labor_category",
            "labor_created_at",
            "labor_start_date",
            "labor_end_date",
            "alfred",
            "shop",
        },
    )
    _require_columns(
        address_df,
        name="address.csv",
        required={"shop", "alfred", "address_id", "point"},
    )

    unified = service_labors_df.merge(
        services_df,
        on="service_id",
        how="left",
        suffixes=("_labor", "_service"),
    )

    shop_lookup, alfred_lookup = _build_address_lookups(address_df)
    unified = unified.merge(shop_lookup, on="shop", how="left")
    unified = unified.merge(alfred_lookup, on="alfred", how="left")

    has_shop = unified["shop"].notna()
    has_alfred = unified["alfred"].notna()
    row_conflict = has_shop & has_alfred
    use_shop = has_shop & ~has_alfred
    use_alfred = has_alfred & ~has_shop

    unified["address_id"] = pd.NA
    unified["address_point"] = pd.NA

    unified.loc[use_shop, "address_id"] = unified.loc[use_shop, "shop_address_id"]
    unified.loc[use_shop, "address_point"] = unified.loc[use_shop, "shop_address_point"]

    unified.loc[use_alfred, "address_id"] = unified.loc[use_alfred, "alfred_address_id"]
    unified.loc[use_alfred, "address_point"] = unified.loc[use_alfred, "alfred_address_point"]

    datetime_cols = [
        "created_at",
        "schedule_date",
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
    ]
    for col in datetime_cols:
        if col in unified.columns:
            unified[col] = unified[col].map(_normalize_bogota_datetime)

    unified["labor_sequence"] = (
        unified.groupby("service_id", sort=False).cumcount() + 1
    )

    unified["city_code"] = unified["city"]

    output_columns = [
        "service_id",
        "created_at",
        "schedule_date",
        "start_address_id",
        "start_address_point",
        "end_address_id",
        "end_address_point",
        "city_code",
        "department_code",
        "labor_id",
        "labor_type",
        "labor_name",
        "labor_category",
        "labor_sequence",
        "alfred",
        "shop",
        "address_id",
        "address_point",
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
    ]

    output_df = unified.loc[:, output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(
        "input_csv_built rows=%s services=%s address_conflicts=%s output=%s",
        len(output_df),
        output_df["service_id"].nunique(dropna=True),
        int(row_conflict.sum()),
        output_path.as_posix(),
    )

    return output_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build input.csv from raw services files")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory with services.csv, service_labors.csv and address.csv (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_parser().parse_args()
    build_input_csv(
        raw_dir=_resolve_path(args.raw_dir),
        output_path=_resolve_path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
