import argparse
from pathlib import Path

import pandas as pd


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
DEFAULT_BASE_DIR = ROOT / "data" / "master_data"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _convert_csv(path: Path) -> Path:
    df = pd.read_csv(path)
    out_path = path.with_suffix(".parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def _convert_dist_dict(path: Path) -> Path:
    dist_obj = pd.read_pickle(path)
    if isinstance(dist_obj, pd.DataFrame):
        df = dist_obj
    elif isinstance(dist_obj, dict):
        rows = []
        for city, city_map in dist_obj.items():
            for key, value in city_map.items():
                p1, p2 = key
                rows.append(
                    {
                        "city": str(city),
                        "p1": str(p1),
                        "p2": str(p2),
                        "distance_km": value,
                    }
                )
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported dist_dict object type: {type(dist_obj)}")

    out_path = path.with_suffix(".parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert master data CSV files to Parquet."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing master CSV files.",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=["directorio", "duraciones", "dist_dict"],
        help="File stems to convert (without extension).",
    )
    args = parser.parse_args()

    base_dir = _resolve_path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    for stem in args.stems:
        if stem == "dist_dict":
            pkl_path = base_dir / f"{stem}.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Pickle not found: {pkl_path}")
            out_path = _convert_dist_dict(pkl_path)
            print(f"Converted {pkl_path} -> {out_path}")
            continue

        csv_path = base_dir / f"{stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        out_path = _convert_csv(csv_path)
        print(f"Converted {csv_path} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
