"""
Phase 1 API vs LOCAL parity experiment.

Creates traceable artifacts under:
    data/phase1/{department_code}/{execution_mode}

For each department present in the fetched API payload:
  - payload snapshot (department-scoped)
  - input_full.csv (local baseline for the window)
  - input.csv (labor-filtered to payload labor IDs)
  - run request.json used
  - run.log
  - copied intermediate exports
  - copied model outputs
  - README.md with removed/missing labor analysis and reasons

No permanent runtime logic changes are required; this script only orchestrates
environment overrides and copies artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data.parsing.input_parser import InputParser
from src.integration.client import ALFREDAPIClient
from src.io.request_loader import load_request


PHASE1_DIR = ROOT / "data" / "phase1"
MODEL_OUTPUT_DIR = ROOT / "data" / "model_output"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate_exports"
FULL_INPUT_PATH = ROOT / "data" / "model_input" / "input.csv"
RAW_SERVICES_PATH = ROOT / "data" / "model_input" / "raw_files" / "services.csv"
MRUN = ROOT / "scripts" / "mrun"


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir_model_output: Path
    run_dir_intermediate: Path
    log_path: Path


def _resolve_request(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _to_ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _extract_run_id(log_text: str) -> str:
    # run_id is logged as "run_id=xxxxxxxx"
    matches = re.findall(r"run_id=([0-9a-f]{8})", log_text)
    if not matches:
        raise RuntimeError("Could not extract run_id from run log")
    return matches[0]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _filter_full_input_for_window(
    full_df: pd.DataFrame,
    *,
    department_code: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    df = full_df.copy()
    if "department_code" in df.columns:
        dep = df["department_code"].astype("string").str.strip()
        df = df.loc[dep.eq(department_code).fillna(False)].copy()

    if "schedule_date" in df.columns and (start_date is not None or end_date is not None):
        schedule = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
        mask = pd.Series(True, index=df.index)
        if start_date is not None:
            mask &= schedule >= start_date
        if end_date is not None:
            mask &= schedule <= end_date
        df = df.loc[mask].copy()

    return df


def _subset_payload_by_services(payload: Dict[str, Any], service_ids: set[str]) -> Dict[str, Any]:
    services = payload.get("data", [])
    if not isinstance(services, list):
        return {"count": 0, "next": None, "previous": None, "data": []}

    selected: List[Dict[str, Any]] = []
    for service in services:
        if not isinstance(service, dict):
            continue
        sid = service.get("service_id")
        sid_txt = str(sid) if sid is not None else ""
        if sid_txt in service_ids:
            selected.append(service)

    scoped = dict(payload)
    scoped["data"] = selected
    scoped["count"] = len(selected)
    return scoped


def _payload_labor_rows(payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    services = payload.get("data", [])
    if not isinstance(services, list):
        return pd.DataFrame()

    for service in services:
        if not isinstance(service, dict):
            continue
        sid = service.get("service_id")
        state = service.get("state")
        service_schedule = service.get("schedule_date")
        labors = service.get("serviceLabors", [])
        if not isinstance(labors, list):
            continue
        for labor in labors:
            if not isinstance(labor, dict):
                continue
            rows.append(
                {
                    "service_id": str(sid) if sid is not None else None,
                    "labor_id": str(labor.get("id")) if labor.get("id") is not None else None,
                    "labor_name": labor.get("labor_name"),
                    "labor_category": labor.get("labor_category"),
                    "state": state,
                    "service_schedule_date": service_schedule,
                    "labor_schedule_date": labor.get("schedule_date"),
                }
            )

    return pd.DataFrame(rows)


def _read_service_state_map() -> Dict[str, str]:
    if not RAW_SERVICES_PATH.exists():
        return {}
    df = pd.read_csv(RAW_SERVICES_PATH, low_memory=False)
    if "service_id" not in df.columns or "state_service" not in df.columns:
        return {}
    key = df["service_id"].astype("string").str.strip()
    value = df["state_service"].astype("string").str.strip()
    state_map: Dict[str, str] = {}
    for k, v in zip(key.tolist(), value.tolist()):
        if k and k != "<NA>" and v and v != "<NA>":
            state_map[str(k)] = str(v)
    return state_map


def _reason_removed(
    *,
    service_id: str,
    payload_service_ids: set[str],
    state_map: Dict[str, str],
) -> str:
    state = state_map.get(service_id)
    if state in {"CANCELED", "CANCELLED"}:
        return "Service state is CANCELED in local raw services extract."
    if service_id not in payload_service_ids:
        return "Service absent from API payload for the same request window (likely stale/local-only extract)."
    return "Labor absent in payload while service exists (payload labor set differs from CSV labor set)."


def _run_main(mode: str, env: Dict[str, str], log_path: Path) -> RunArtifacts:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(MRUN), "python", "main.py"]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} run failed with code {proc.returncode}. See {log_path}")

    run_id = _extract_run_id(proc.stdout)
    run_tag = f"run-{mode}-{run_id}"
    output_dir = MODEL_OUTPUT_DIR / run_tag
    intermediate_dir = INTERMEDIATE_DIR / run_tag
    if not output_dir.exists():
        raise RuntimeError(f"Expected output directory not found: {output_dir}")
    if not intermediate_dir.exists():
        raise RuntimeError(f"Expected intermediate directory not found: {intermediate_dir}")

    return RunArtifacts(
        run_id=run_id,
        run_dir_model_output=output_dir,
        run_dir_intermediate=intermediate_dir,
        log_path=log_path,
    )


def _write_readme(
    readme_path: Path,
    *,
    department_code: str,
    city_values: Iterable[str],
    payload_rows: pd.DataFrame,
    input_full_df: pd.DataFrame,
    input_filtered_df: pd.DataFrame,
    removed_rows: pd.DataFrame,
    missing_rows: pd.DataFrame,
    removed_reason_map: Dict[str, str],
    request_payload: Dict[str, Any],
) -> None:
    cities = sorted({c for c in city_values if isinstance(c, str) and c.strip()})
    lines: List[str] = []
    lines.append(f"# Phase 1 Report - Department {department_code}")
    lines.append("")
    lines.append("## Request Filters")
    lines.append("```json")
    lines.append(json.dumps(request_payload.get("filters", {}), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Coverage Summary")
    lines.append(f"- Cities found in API payload: {', '.join(cities) if cities else '(none)'}")
    lines.append(f"- Payload labors: {len(payload_rows)}")
    lines.append(f"- input_full.csv labors (department/date window): {len(input_full_df)}")
    lines.append(f"- input.csv labors (filtered to payload labor IDs): {len(input_filtered_df)}")
    lines.append(f"- Removed labors (input_full - payload): {len(removed_rows)}")
    lines.append(f"- Missing labors (payload - input_full): {len(missing_rows)}")
    lines.append("")

    lines.append("## Removed Labors")
    if removed_rows.empty:
        lines.append("No removed labors.")
    else:
        lines.append("| labor_id | service_id | labor_name | labor_category | reason |")
        lines.append("|---|---|---|---|---|")
        for _, row in removed_rows.sort_values(["service_id", "labor_id"]).iterrows():
            labor_id = str(row.get("labor_id"))
            service_id = str(row.get("service_id"))
            reason = removed_reason_map.get(labor_id, "Unknown")
            lines.append(
                f"| {labor_id} | {service_id} | {row.get('labor_name', '')} | "
                f"{row.get('labor_category', '')} | {reason} |"
            )
    lines.append("")

    lines.append("## Missing From Local CSV")
    if missing_rows.empty:
        lines.append("No missing payload labors.")
    else:
        lines.append("| labor_id | service_id | labor_name | labor_category | reason |")
        lines.append("|---|---|---|---|---|")
        for _, row in missing_rows.sort_values(["service_id", "labor_id"]).iterrows():
            lines.append(
                f"| {row.get('labor_id')} | {row.get('service_id')} | {row.get('labor_name', '')} | "
                f"{row.get('labor_category', '')} | Not present in local CSV extract for the same window. |"
            )
    lines.append("")

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def _write_run_error(mode_dir: Path, mode: str, error: Exception) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    path = mode_dir / "run_error.txt"
    path.write_text(
        f"{mode} run failed.\n{type(error).__name__}: {error}\n",
        encoding="utf-8",
    )


def _clear_run_error(mode_dir: Path) -> None:
    err = mode_dir / "run_error.txt"
    if err.exists():
        err.unlink()


def run_phase1(request_path: Path) -> None:
    Config.validate()

    request = load_request(request_path)
    request_raw = _read_json(request_path)
    filters = request.filters

    endpoint = Config.SERVICES_ENDPOINT
    token = Config.API_TOKEN
    if not endpoint or not token:
        raise RuntimeError("SERVICES_ENDPOINT and API_TOKEN must be configured.")

    client = ALFREDAPIClient(
        endpoint_url=endpoint,
        api_token=token,
        timeout=Config.REQUEST_TIMEOUT,
        max_retries=Config.API_MAX_RETRIES,
    )

    payload = client.get_optimization_data(
        department=filters.department,
        start_date=filters.start_date,
        end_date=filters.end_date,
    )

    payload_df, _ = InputParser.parse(payload)
    if payload_df.empty:
        raise RuntimeError("API payload parsed as empty. Nothing to process.")

    departments = (
        payload_df["department_code"].astype("string").str.strip().dropna().unique().tolist()
    )
    departments = sorted(str(x) for x in departments if str(x).strip())
    if not departments:
        raise RuntimeError("No department_code found in payload.")

    if not FULL_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing local full CSV: {FULL_INPUT_PATH}")
    full_df = pd.read_csv(FULL_INPUT_PATH, low_memory=False)
    state_map = _read_service_state_map()

    start_ts = _to_ts(filters.start_date.isoformat() if filters.start_date else None)
    end_ts = _to_ts(filters.end_date.isoformat() if filters.end_date else None)

    for department_code in departments:
        dept_root = PHASE1_DIR / department_code
        api_dir = dept_root / "api"
        local_dir = dept_root / "local"
        api_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)

        dept_df = payload_df.loc[
            payload_df["department_code"].astype("string").str.strip().eq(department_code)
        ].copy()
        service_ids = set(dept_df["service_id"].astype("string").str.strip().dropna().tolist())
        city_values = dept_df["city_name"].astype("string").dropna().tolist()

        scoped_payload = _subset_payload_by_services(payload, service_ids)
        payload_rows = _payload_labor_rows(scoped_payload)
        payload_labor_ids = set(payload_rows["labor_id"].dropna().astype(str).tolist())
        payload_service_ids = set(payload_rows["service_id"].dropna().astype(str).tolist())

        input_full_df = _filter_full_input_for_window(
            full_df,
            department_code=department_code,
            start_date=start_ts,
            end_date=end_ts,
        )
        full_rows = input_full_df.copy()
        full_rows["service_id"] = full_rows["service_id"].astype("string")
        full_rows["labor_id"] = full_rows["labor_id"].astype("string")

        input_filtered_df = full_rows.loc[
            full_rows["labor_id"].isin(payload_labor_ids)
        ].copy()

        input_full_path_local = local_dir / "input_full.csv"
        input_path_local = local_dir / "input.csv"
        input_full_df.to_csv(input_full_path_local, index=False)
        input_filtered_df.to_csv(input_path_local, index=False)

        # Mirror into API folder for traceability.
        shutil.copy2(input_full_path_local, api_dir / "input_full.csv")
        shutil.copy2(input_path_local, api_dir / "input.csv")

        payload_path_api = api_dir / "payload_snapshot.json"
        payload_path_local = local_dir / "payload_snapshot.json"
        _write_json(payload_path_api, scoped_payload)
        _write_json(payload_path_local, scoped_payload)

        csv_labor_ids = set(full_rows["labor_id"].dropna().astype(str).tolist())
        removed_ids = sorted(csv_labor_ids - payload_labor_ids)
        missing_ids = sorted(payload_labor_ids - csv_labor_ids)

        removed_rows = full_rows.loc[full_rows["labor_id"].isin(removed_ids)].copy()
        missing_rows = payload_rows.loc[payload_rows["labor_id"].isin(missing_ids)].copy()

        removed_reason_map: Dict[str, str] = {}
        for _, row in removed_rows.drop_duplicates(subset=["labor_id"]).iterrows():
            labor_id = str(row["labor_id"])
            service_id = str(row["service_id"])
            removed_reason_map[labor_id] = _reason_removed(
                service_id=service_id,
                payload_service_ids=payload_service_ids,
                state_map=state_map,
            )

        _write_readme(
            dept_root / "README.md",
            department_code=department_code,
            city_values=city_values,
            payload_rows=payload_rows,
            input_full_df=input_full_df,
            input_filtered_df=input_filtered_df,
            removed_rows=removed_rows,
            missing_rows=missing_rows,
            removed_reason_map=removed_reason_map,
            request_payload=request_raw,
        )

        run_request = dict(request_raw)
        run_request["filters"] = dict(run_request.get("filters", {}))
        run_request["filters"]["department"] = department_code
        if filters.start_date is not None:
            run_request["filters"]["start_date"] = filters.start_date.isoformat()
        if filters.end_date is not None:
            run_request["filters"]["end_date"] = filters.end_date.isoformat()

        run_request_path = dept_root / "request.json"
        _write_json(run_request_path, run_request)
        shutil.copy2(run_request_path, api_dir / "request.json")
        shutil.copy2(run_request_path, local_dir / "request.json")

        base_env = os.environ.copy()

        local_env = dict(base_env)
        local_env["USE_API"] = "false"
        local_env["REQUEST_PATH"] = str(run_request_path)
        local_env["LOCAL_INPUT_FILE"] = str(input_path_local)
        local_env["LOCAL_INPUT_DIR"] = str(local_dir)

        local_ok = True
        try:
            local_run = _run_main(
                mode="local",
                env=local_env,
                log_path=local_dir / "run.log",
            )
            _copy_tree(local_run.run_dir_model_output, local_dir / "model_output")
            _copy_tree(local_run.run_dir_intermediate, local_dir / "intermediate_exports")
            _clear_run_error(local_dir)
        except Exception as exc:
            local_ok = False
            _write_run_error(local_dir, "local", exc)

        api_env = dict(base_env)
        api_env["USE_API"] = "true"
        api_env["REQUEST_PATH"] = str(run_request_path)

        api_ok = True
        try:
            api_run = _run_main(
                mode="api",
                env=api_env,
                log_path=api_dir / "run.log",
            )
            _copy_tree(api_run.run_dir_model_output, api_dir / "model_output")
            _copy_tree(api_run.run_dir_intermediate, api_dir / "intermediate_exports")
            _clear_run_error(api_dir)
        except Exception as exc:
            api_ok = False
            _write_run_error(api_dir, "api", exc)

        if not local_ok or not api_ok:
            readme_path = dept_root / "README.md"
            with readme_path.open("a", encoding="utf-8") as f:
                f.write("\n## Run Status\n")
                f.write(f"- local: {'ok' if local_ok else 'failed'}\n")
                f.write(f"- api: {'ok' if api_ok else 'failed'}\n")
                if not local_ok:
                    f.write("- local failure details: `local/run_error.txt`\n")
                if not api_ok:
                    f.write("- api failure details: `api/run_error.txt`\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run phase1 API/local parity experiment and archive artifacts under data/phase1."
    )
    parser.add_argument(
        "--request-path",
        type=Path,
        default=Path("request.json"),
        help="Request JSON path (default: request.json).",
    )
    args = parser.parse_args()

    request_path = _resolve_request(args.request_path)
    run_phase1(request_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
