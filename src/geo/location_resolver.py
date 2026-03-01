from __future__ import annotations

import csv
import json
import logging
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)

STATUS_RESOLVED = "resolved"
STATUS_RESOLVED_DEPARTMENT_ONLY = "resolved_department_only"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_MISSING = "missing"


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_label(value: Any) -> str:
    text = _clean_text(value)
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = " ".join(normalized.upper().split())
    return normalized


def _split_department_city(value: Any) -> tuple[str | None, str | None]:
    text = _clean_text(value)
    if text is None:
        return None, None

    left, sep, right = text.partition("-")
    if not sep:
        return None, text
    return _clean_text(left), _clean_text(right)


def _iter_city_aliases(city_name: str | None, display_city_name: str | None) -> Iterable[str]:
    for value in (city_name, display_city_name):
        label = _normalize_label(value)
        if label:
            yield label


@dataclass(frozen=True)
class CanonicalLocation:
    city_code: str
    city_name: str | None
    department_code: str | None
    department_name: str | None


class LocationResolver:
    def __init__(
        self,
        *,
        city_code_to_location: Dict[str, CanonicalLocation],
        city_by_department_code_and_name: Dict[Tuple[str, str], str],
        city_by_department_name_and_name: Dict[Tuple[str, str], str],
        city_alias_to_city_codes: Dict[str, set[str]],
        department_name_to_code: Dict[str, str],
    ):
        self._city_code_to_location = city_code_to_location
        self._city_by_department_code_and_name = city_by_department_code_and_name
        self._city_by_department_name_and_name = city_by_department_name_and_name
        self._city_alias_to_city_codes = city_alias_to_city_codes
        self._department_name_to_code = department_name_to_code

    @classmethod
    def from_files(
        cls,
        *,
        cities_path: str | Path | None = None,
        address_path: str | Path | None = None,
    ) -> "LocationResolver":
        city_code_to_location: Dict[str, CanonicalLocation] = {}
        city_by_department_code_and_name: Dict[Tuple[str, str], str] = {}
        city_by_department_name_and_name: Dict[Tuple[str, str], str] = {}
        city_alias_to_city_codes: Dict[str, set[str]] = {}
        department_name_to_codes: Dict[str, set[str]] = {}

        if cities_path:
            cities_file = Path(cities_path)
            if cities_file.exists():
                cls._load_cities_csv(
                    cities_file,
                    city_code_to_location=city_code_to_location,
                    city_by_department_code_and_name=city_by_department_code_and_name,
                    city_by_department_name_and_name=city_by_department_name_and_name,
                    city_alias_to_city_codes=city_alias_to_city_codes,
                    department_name_to_codes=department_name_to_codes,
                )
            else:
                logger.warning("cities_crosswalk_not_found path=%s", cities_file)

        if address_path:
            address_file = Path(address_path)
            if address_file.exists():
                cls._load_address_csv_fallback(
                    address_file,
                    city_code_to_location=city_code_to_location,
                )
            else:
                logger.warning("address_crosswalk_not_found path=%s", address_file)

        department_name_to_code: Dict[str, str] = {}
        for label, codes in department_name_to_codes.items():
            if len(codes) == 1:
                department_name_to_code[label] = next(iter(codes))

        return cls(
            city_code_to_location=city_code_to_location,
            city_by_department_code_and_name=city_by_department_code_and_name,
            city_by_department_name_and_name=city_by_department_name_and_name,
            city_alias_to_city_codes=city_alias_to_city_codes,
            department_name_to_code=department_name_to_code,
        )

    @staticmethod
    def _load_cities_csv(
        path: Path,
        *,
        city_code_to_location: Dict[str, CanonicalLocation],
        city_by_department_code_and_name: Dict[Tuple[str, str], str],
        city_by_department_name_and_name: Dict[Tuple[str, str], str],
        city_alias_to_city_codes: Dict[str, set[str]],
        department_name_to_codes: Dict[str, set[str]],
    ) -> None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                city_code = _clean_text(row.get("city"))
                if city_code is None:
                    continue

                payload_raw = row.get("data")
                payload: Dict[str, Any] = {}
                if payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        payload = {}

                display_name = _clean_text(row.get("name"))
                display_department, display_city = _split_department_city(display_name)

                department_code = _clean_text(payload.get("department_code"))
                department_name = _clean_text(payload.get("department_name")) or display_department
                city_name = _clean_text(payload.get("city_name")) or display_city

                city_code_to_location[city_code] = CanonicalLocation(
                    city_code=city_code,
                    city_name=city_name,
                    department_code=department_code,
                    department_name=department_name,
                )

                if department_code is not None:
                    for alias in _iter_city_aliases(city_name, display_city):
                        city_by_department_code_and_name[(department_code, alias)] = city_code
                    if department_name is not None:
                        department_label = _normalize_label(department_name)
                        if department_label:
                            department_name_to_codes.setdefault(department_label, set()).add(department_code)

                if department_name is not None:
                    dept_label = _normalize_label(department_name)
                    if dept_label:
                        for alias in _iter_city_aliases(city_name, display_city):
                            city_by_department_name_and_name[(dept_label, alias)] = city_code

                for alias in _iter_city_aliases(city_name, display_city):
                    city_alias_to_city_codes.setdefault(alias, set()).add(city_code)

    @staticmethod
    def _load_address_csv_fallback(
        path: Path,
        *,
        city_code_to_location: Dict[str, CanonicalLocation],
    ) -> None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                city_code = _clean_text(row.get("city"))
                department_code = _clean_text(row.get("department_code"))
                if city_code is None or department_code is None:
                    continue

                current = city_code_to_location.get(city_code)
                if current is None:
                    city_code_to_location[city_code] = CanonicalLocation(
                        city_code=city_code,
                        city_name=None,
                        department_code=department_code,
                        department_name=None,
                    )
                    continue

                if current.department_code is None:
                    city_code_to_location[city_code] = CanonicalLocation(
                        city_code=current.city_code,
                        city_name=current.city_name,
                        department_code=department_code,
                        department_name=current.department_name,
                    )

    def resolve(
        self,
        *,
        city_code: Any = None,
        city_name: Any = None,
        department_code: Any = None,
        department_name: Any = None,
        department_city_text: Any = None,
    ) -> dict[str, str | None]:
        city_code_txt = _clean_text(city_code)
        city_name_txt = _clean_text(city_name)
        department_code_txt = _clean_text(department_code)
        department_name_txt = _clean_text(department_name)

        from_text_department, from_text_city = _split_department_city(department_city_text)
        if department_name_txt is None:
            department_name_txt = from_text_department
        if city_name_txt is None:
            city_name_txt = from_text_city

        if department_code_txt is None and department_name_txt is not None:
            department_code_txt = self._department_name_to_code.get(_normalize_label(department_name_txt))

        if city_code_txt is not None:
            mapped = self._city_code_to_location.get(city_code_txt)
            if mapped is not None:
                department_code_txt = department_code_txt or mapped.department_code
                department_name_txt = department_name_txt or mapped.department_name
                city_name_txt = city_name_txt or mapped.city_name
                return self._result(
                    city_code=city_code_txt,
                    city_name=city_name_txt,
                    department_code=department_code_txt,
                    department_name=department_name_txt,
                    status=STATUS_RESOLVED,
                )

            return self._result(
                city_code=city_code_txt,
                city_name=city_name_txt,
                department_code=department_code_txt,
                department_name=department_name_txt,
                status=self._status_when_partial(
                    city_code=city_code_txt,
                    city_name=city_name_txt,
                    department_code=department_code_txt,
                    department_name=department_name_txt,
                ),
            )

        city_label = _normalize_label(city_name_txt)
        if city_label and (department_code_txt or department_name_txt):
            city_code_match = None
            if department_code_txt:
                city_code_match = self._city_by_department_code_and_name.get(
                    (department_code_txt, city_label)
                )
            if city_code_match is None and department_name_txt:
                city_code_match = self._city_by_department_name_and_name.get(
                    (_normalize_label(department_name_txt), city_label)
                )

            if city_code_match is not None:
                mapped = self._city_code_to_location.get(city_code_match)
                if mapped is not None:
                    department_code_txt = department_code_txt or mapped.department_code
                    department_name_txt = department_name_txt or mapped.department_name
                    city_name_txt = city_name_txt or mapped.city_name

                return self._result(
                    city_code=city_code_match,
                    city_name=city_name_txt,
                    department_code=department_code_txt,
                    department_name=department_name_txt,
                    status=STATUS_RESOLVED,
                )

        if city_label and not department_code_txt and not department_name_txt:
            candidates = self._city_alias_to_city_codes.get(city_label, set())
            if len(candidates) == 1:
                only_city = next(iter(candidates))
                mapped = self._city_code_to_location.get(only_city)
                if mapped is not None:
                    return self._result(
                        city_code=mapped.city_code,
                        city_name=city_name_txt or mapped.city_name,
                        department_code=mapped.department_code,
                        department_name=mapped.department_name,
                        status=STATUS_RESOLVED,
                    )
                return self._result(
                    city_code=only_city,
                    city_name=city_name_txt,
                    department_code=None,
                    department_name=None,
                    status=STATUS_MISSING,
                )

            if len(candidates) > 1:
                return self._result(
                    city_code=None,
                    city_name=city_name_txt,
                    department_code=None,
                    department_name=None,
                    status=STATUS_AMBIGUOUS,
                )

        if department_code_txt or department_name_txt:
            return self._result(
                city_code=None,
                city_name=city_name_txt,
                department_code=department_code_txt,
                department_name=department_name_txt,
                status=STATUS_RESOLVED_DEPARTMENT_ONLY,
            )

        return self._result(
            city_code=None,
            city_name=None,
            department_code=None,
            department_name=None,
            status=STATUS_MISSING,
        )

    @staticmethod
    def _status_when_partial(
        *,
        city_code: str | None,
        city_name: str | None,
        department_code: str | None,
        department_name: str | None,
    ) -> str:
        if department_code or department_name:
            return STATUS_RESOLVED
        if city_code or city_name:
            return STATUS_MISSING
        return STATUS_MISSING

    @staticmethod
    def _result(
        *,
        city_code: str | None,
        city_name: str | None,
        department_code: str | None,
        department_name: str | None,
        status: str,
    ) -> dict[str, str | None]:
        return {
            "city_code": city_code,
            "city_name": city_name,
            "department_code": department_code,
            "department_name": department_name,
            "location_resolution_status": status,
        }


@lru_cache(maxsize=4)
def get_location_resolver(
    cities_path: str | None = None,
    address_path: str | None = None,
) -> LocationResolver:
    repo_root = Path(__file__).resolve().parent.parent.parent
    resolved_cities = cities_path or str(repo_root / "data" / "master_data" / "cities.csv")
    resolved_address = address_path or str(
        repo_root / "data" / "model_input" / "raw_files" / "address.csv"
    )
    return LocationResolver.from_files(
        cities_path=resolved_cities,
        address_path=resolved_address,
    )
