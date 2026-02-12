"""
Build local `data/model_input/driver_directory.csv` from raw driver directory CSV.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
DEFAULT_SOURCE_PATH = ROOT / "data" / "model_input" / "raw_files" / "directorio_df.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "model_input" / "driver_directory.csv"


def _require_columns(df: pd.DataFrame, *, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Driver directory CSV missing required columns: {missing}")


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def build_driver_directory_csv(*, source_path: Path, output_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing file: {source_path}")

    df = pd.read_csv(source_path, low_memory=False)
    _require_columns(
        df,
        required={
            "driver_id",
            "ubicacion",
            "department_code",
            "start_time",
            "end_time",
        },
    )

    output_df = pd.DataFrame(
        {
            "driver_id": df["driver_id"],
            "point": df["ubicacion"],
            "department_code": df["department_code"],
            "start_time": df["start_time"],
            "end_time": df["end_time"],
            "city_id": df["city_id"] if "city_id" in df.columns else pd.NA,
            "city_name": df["city_name"] if "city_name" in df.columns else pd.NA,
            "department_name": df["department_name"] if "department_name" in df.columns else pd.NA,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(
        "driver_directory_csv_built rows=%s output=%s",
        len(output_df),
        output_path.as_posix(),
    )
    return output_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build driver_directory.csv from raw directory CSV")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=f"Raw directory CSV path (default: {DEFAULT_SOURCE_PATH})",
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
    build_driver_directory_csv(
        source_path=_resolve_path(args.source),
        output_path=_resolve_path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
