import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config
from src.data.id_normalization import normalize_id_value
from src.io.artifact_naming import build_run_subdir

logger = logging.getLogger(__name__)

# ======================================================
# Public API
# ======================================================

def load_local_input(
    local_path: str | Path,
    *,
    write_debug_json: bool = False,
    debug_output_path: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load local input data from a CSV file (transformed) or a JSON payload
    snapshot (returned as-is).

    The payload structure intentionally matches the API input shape
    consumed by InputParser so the downstream parse/validate/solve flow
    is identical for USE_API=true and USE_API=false.
    """
    input_path = Path(local_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Local input file not found: {input_path}")

    payload: Dict[str, Any]
    if input_path.suffix.lower() == ".json":
        payload = _load_local_json_payload(input_path)
    else:
        payload = _load_local_csv_as_payload(input_path)

    if write_debug_json:
        _write_debug_payload(payload, debug_output_path, run_id=run_id)

    return payload


def _load_local_json_payload(input_path: Path) -> Dict[str, Any]:
    logger.debug("Loading local input payload from JSON", extra={"path": str(input_path)})
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object payload in {input_path}")

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Expected key 'data' as list in {input_path}")

    labors_total = 0
    for service in data:
        if not isinstance(service, dict):
            continue
        labors = service.get("serviceLabors")
        if isinstance(labors, list):
            labors_total += len(labors)

    logger.info(
        "local_input_loaded_from_json services=%s labors=%s",
        len(data),
        labors_total,
    )
    return payload


def _load_local_csv_as_payload(csv_path: Path) -> Dict[str, Any]:
    logger.debug("Loading local input from CSV", extra={"path": str(csv_path)})

    df = pd.read_csv(csv_path)

    _validate_required_columns(df)

    services_by_key: Dict[str, Dict[str, Any]] = {}

    for row_index, row in df.iterrows():
        row_dict = row.to_dict()

        service_id = normalize_id_value(row_dict.get("service_id"))
        # Keep rows with missing service_id instead of silently dropping them.
        service_key = service_id if service_id is not None else f"__missing_service_{row_index}"

        service_record = services_by_key.get(service_key)
        if service_record is None:
            service_record = _build_service_record(row_dict, service_id=service_id)
            services_by_key[service_key] = service_record
        else:
            _merge_service_level_fields(service_record, row_dict)

        service_record["serviceLabors"].append(_build_labor_record(row_dict))

    payload: Dict[str, Any] = {
        "count": len(services_by_key),
        "next": None,
        "previous": None,
        "data": list(services_by_key.values()),
    }

    logger.info(
        "local_input_loaded services=%s labors=%s",
        payload["count"],
        len(df),
    )
    return payload


# ======================================================
# Helpers
# ======================================================

def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure required columns exist in the CSV input.
    """
    required_columns = {
        "service_id",
        "schedule_date",
        "start_address_id",
        "start_address_point",
        "end_address_id",
        "end_address_point",
        "labor_id",
        "labor_type",
        "labor_name",
        "labor_category",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    location_columns = {"city", "city_code", "city_name", "department_code", "department_name"}
    if not (location_columns & set(df.columns)):
        raise ValueError(
            "Missing location columns in CSV: expected at least one of "
            "'city', 'city_code', 'city_name', 'department_code', 'department_name'"
        )


def _build_service_record(row: Dict[str, Any], *, service_id: Optional[str]) -> Dict[str, Any]:
    """
    Build one service object using API payload keys expected by InputParser.
    """
    return {
        "service_id": service_id,
        "state": _clean_nullable_text(row.get("state") or row.get("state_service")),
        "created_at": _normalize_datetime_value(row.get("created_at")),
        "schedule_date": _normalize_datetime_value(row.get("schedule_date")),
        "start_address": {
            "id": normalize_id_value(row.get("start_address_id")),
            "point": _parse_point(row.get("start_address_point")),
            "city": _resolve_city_value(row),
        },
        "end_address": {
            "id": normalize_id_value(row.get("end_address_id")),
            "point": _parse_point(row.get("end_address_point")),
        },
        "serviceLabors": [],
    }


def _merge_service_level_fields(service_record: Dict[str, Any], row: Dict[str, Any]) -> None:
    """
    Fill missing service-level values from additional rows of the same service.
    """
    if service_record.get("created_at") is None:
        service_record["created_at"] = _normalize_datetime_value(row.get("created_at"))

    if service_record.get("state") is None:
        service_record["state"] = _clean_nullable_text(row.get("state") or row.get("state_service"))

    if service_record.get("schedule_date") is None:
        service_record["schedule_date"] = _normalize_datetime_value(row.get("schedule_date"))

    start_address = service_record.get("start_address", {})
    if start_address.get("id") is None:
        start_address["id"] = normalize_id_value(row.get("start_address_id"))
    if start_address.get("point") is None:
        start_address["point"] = _parse_point(row.get("start_address_point"))
    if start_address.get("city") is None:
        start_address["city"] = _resolve_city_value(row)
    service_record["start_address"] = start_address

    end_address = service_record.get("end_address", {})
    if end_address.get("id") is None:
        end_address["id"] = normalize_id_value(row.get("end_address_id"))
    if end_address.get("point") is None:
        end_address["point"] = _parse_point(row.get("end_address_point"))
    service_record["end_address"] = end_address


def _build_labor_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build one labor object using API payload keys expected by InputParser.
    """
    assigned_driver = normalize_id_value(row.get("alfred") or row.get("assigned_driver"))
    shop_id = normalize_id_value(row.get("shop") or row.get("shop_id"))

    shop_address_id = row.get("shop_address_id")
    if _is_missing(shop_address_id):
        shop_address_id = row.get("address_id")

    shop_address_point = row.get("shop_address_point")
    if _is_missing(shop_address_point):
        shop_address_point = row.get("address_point")

    shop_address: Dict[str, Any] | None = None
    if not _is_missing(shop_address_id) or not _is_missing(shop_address_point):
        shop_address = {
            "id": normalize_id_value(shop_address_id),
            "point": _parse_point(shop_address_point),
        }

    shop: Dict[str, Any] | None = None
    if shop_id is not None or shop_address is not None:
        shop = {}
        if shop_id is not None:
            shop["id"] = shop_id

    return {
        "id": normalize_id_value(row.get("labor_id")),
        "labor_id": normalize_id_value(row.get("labor_type")),
        "labor_name": _clean_nullable_text(row.get("labor_name")),
        "labor_category": _clean_nullable_text(row.get("labor_category")),
        "schedule_date": _normalize_datetime_value(row.get("schedule_date")),
        "labor_sequence": _normalize_labor_sequence(row.get("labor_sequence")),
        "alfred": {"id": assigned_driver} if assigned_driver is not None else None,
        "actual_start": _normalize_datetime_value(row.get("actual_start") or row.get("labor_start_date")),
        "actual_end": _normalize_datetime_value(row.get("actual_end") or row.get("labor_end_date")),
        "shop": shop,
        "shop_address": shop_address,
    }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return isinstance(value, str) and value.strip() == ""


def _clean_nullable_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value).strip() or None


def _normalize_datetime_value(value: Any) -> str | None:
    if _is_missing(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        txt = str(value).strip()
        return txt or None
    return parsed.isoformat()


def _normalize_labor_sequence(value: Any) -> int:
    if _is_missing(value):
        return 1

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 1


def _resolve_city_value(row: Dict[str, Any]) -> Dict[str, Any] | str | None:
    legacy_city_txt = str(row.get("city")).strip() if not _is_missing(row.get("city")) else ""
    legacy_city_code = _normalize_city_code_candidate(row.get("city"))

    city_code = row.get("city_code")
    department_code = row.get("department_code")
    city_name = row.get("city_name")
    department_name = row.get("department_name")

    city_code_txt = str(city_code).strip() if not _is_missing(city_code) else ""
    if not city_code_txt and legacy_city_code:
        city_code_txt = legacy_city_code
    department_code_txt = str(department_code).strip() if not _is_missing(department_code) else ""
    city_name_txt = str(city_name).strip() if not _is_missing(city_name) else ""
    department_name_txt = str(department_name).strip() if not _is_missing(department_name) else ""

    if legacy_city_txt and not legacy_city_txt.isdigit():
        dep_guess, city_guess = _split_department_city_text(legacy_city_txt)
        if not department_name_txt and dep_guess:
            department_name_txt = dep_guess
        if not city_name_txt and city_guess:
            city_name_txt = city_guess
        if not city_name_txt and not dep_guess:
            city_name_txt = legacy_city_txt

    if city_code_txt or department_code_txt or city_name_txt or department_name_txt:
        city_payload: Dict[str, Any] = {}
        if city_code_txt:
            city_payload["id"] = city_code_txt

        location_data: Dict[str, Any] = {}
        if city_code_txt:
            location_data["city_code"] = city_code_txt
        if city_name_txt:
            location_data["city_name"] = city_name_txt
        if department_code_txt:
            location_data["department_code"] = department_code_txt
        if department_name_txt:
            location_data["department_name"] = department_name_txt

        if location_data:
            city_payload["data"] = location_data

        if department_name_txt and city_name_txt:
            city_payload["name"] = f"{department_name_txt}-{city_name_txt}"
        elif city_name_txt:
            city_payload["name"] = city_name_txt
        elif department_name_txt:
            city_payload["name"] = department_name_txt

        return city_payload if city_payload else None

    if legacy_city_txt:
        return legacy_city_txt

    return None


def _split_department_city_text(value: str) -> tuple[str | None, str | None]:
    left, sep, right = value.partition("-")
    if not sep:
        return None, value.strip() or None
    return left.strip() or None, right.strip() or None


def _normalize_city_code_candidate(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric = float(value)
            if numeric.is_integer():
                return str(int(numeric))
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return None


def _parse_point(point_value: Any) -> Dict[str, float] | None:
    """
    Convert a WKT POINT string or API-like dict to a point payload.

    Expected WKT format:
        POINT (longitude latitude)

    Output format matches API:
        {"x": lon, "y": lat}
    """
    if _is_missing(point_value):
        return None

    if isinstance(point_value, dict):
        x = point_value.get("x")
        y = point_value.get("y")
        if _is_missing(x) or _is_missing(y):
            return None
        return {"x": float(x), "y": float(y)}

    if not isinstance(point_value, str):
        raise ValueError(f"Invalid POINT value: {point_value!r}")

    try:
        cleaned = (
            point_value.replace("POINT", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )
        lon_str, lat_str = cleaned.split()

        return {
            "x": float(lon_str),
            "y": float(lat_str),
        }

    except Exception as exc:
        raise ValueError(f"Failed to parse POINT string: {point_value}") from exc


def _write_debug_payload(
    payload: Dict[str, Any],
    output_path: Optional[str | Path],
    *,
    run_id: Optional[str] = None,
) -> None:
    """
    Persist the generated payload for debugging purposes.
    """
    path = (
        Path(output_path)
        if output_path
        else Path(Config.RUNS_DIR) / build_run_subdir(run_id) / "input" / "processed_input.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Writing debug payload", extra={"path": str(path)})

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
