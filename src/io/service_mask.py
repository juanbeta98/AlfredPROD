"""
Service mask — scope a live API payload to a known reference snapshot.

Purpose
-------
When running the solver against the live API (USE_API=true), the API may
return more services than were present in a specific production snapshot used
as a baseline.  The mask narrows the payload to only the labor instance IDs
that appear in that snapshot, making the solver's input directly comparable
to the baseline.

This is a testing/benchmarking aid.  It is NOT needed for local runs that
already load a pre-scoped JSON file (e.g. input_mirror.json or the output of
scripts/data/production_to_solver_input.py).

Lifecycle
---------
- Enabled : set SERVICE_MASK_PATH in .env to the reference snapshot path.
- Disabled: leave SERVICE_MASK_PATH empty (or unset).  The block in main.py
            is a no-op when the env var is falsy.
- Remove  : delete this file and the SERVICE MASK block in main.py, then
            unset SERVICE_MASK_PATH from .env.

Relation to production_to_solver_input.py
------------------------------------------
scripts/data/production_to_solver_input.py converts a production output into
a solver-ready input file (strips alfred assignments, filters off-date labors,
etc.).  Use that script to build the local input file.  Use this mask only
when you need to apply the same scoping dynamically at runtime against the API.

Usage:
    labor_ids = load_labor_ids_from_snapshot("experiments/phase2/cases/file_snapshots/bogota-services-20260218-solucion.json")
    raw_input = apply_service_mask(raw_input, labor_ids)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


def load_labor_ids_from_snapshot(path: str | Path) -> Set[int]:
    """Return the set of labor instance IDs (serviceLabors[].id) from a snapshot file."""
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Service mask snapshot not found: {snapshot_path}")

    with snapshot_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(
            f"Invalid snapshot format in {snapshot_path}: expected {{\"data\": [...]}}"
        )

    labor_ids: Set[int] = set()
    for service in data["data"]:
        for labor in service.get("serviceLabors", []):
            lid = labor.get("id")
            if lid is not None:
                labor_ids.add(int(lid))

    logger.info(
        "service_mask_loaded snapshot=%s labor_ids=%s",
        snapshot_path.name,
        len(labor_ids),
    )
    return labor_ids


def apply_service_mask(
    raw_input: Dict[str, Any],
    labor_ids: Set[int],
) -> Dict[str, Any]:
    """
    Filter raw_input["data"] to only services/labors whose labor instance ID
    appears in labor_ids. Services with no remaining labors are dropped entirely.
    """
    services = raw_input.get("data", [])
    if not isinstance(services, list):
        return raw_input

    filtered: list[Dict[str, Any]] = []
    dropped_services = 0
    kept_labors = 0
    dropped_labors = 0

    for service in services:
        original = service.get("serviceLabors", [])
        kept = [lb for lb in original if int(lb.get("id", -1)) in labor_ids]
        dropped_labors += len(original) - len(kept)

        if kept:
            filtered.append({**service, "serviceLabors": kept})
            kept_labors += len(kept)
        else:
            dropped_services += 1

    logger.info(
        "service_mask_applied services_in=%s services_out=%s services_dropped=%s "
        "labors_kept=%s labors_dropped=%s",
        len(services),
        len(filtered),
        dropped_services,
        kept_labors,
        dropped_labors,
    )

    return {**raw_input, "data": filtered}
