"""
Scope a production baseline payload to the same services/labors as the solver input.

Purpose
-------
The baseline (output_payload.json) covers all services in the production snapshot,
while the solver only receives a filtered subset (input_solver.json, 36 services).
To make comparison metrics comparable apples-to-apples, this script produces a
scoped version of the baseline that contains exactly the same services and labors
as the solver input — with alfred (driver) assignments preserved.

The original file is NOT modified; a new file is written to OUTPUT_FILE.

Usage (from repo root):
    python scripts/data/scope_baseline_to_solver_input.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------
SOLVER_INPUT_FILE = Path("experiments/phase2/cases/input_solver.json")
BASELINE_FILE     = Path("experiments/phase2/cases/file_snapshots/output_payload.json")
OUTPUT_FILE       = Path("experiments/phase2/cases/file_snapshots/output_payload_scoped.json")
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _summary(services: list[dict]) -> str:
    labors = sum(len(s.get("serviceLabors", [])) for s in services)
    return f"services={len(services)} labors={labors}"


def scope_baseline(
    baseline: dict[str, Any],
    solver_input: dict[str, Any],
) -> dict[str, Any]:
    """Filter baseline['data'] to only services/labors present in solver_input['data']."""
    solver_services = solver_input.get("data", [])
    solver_service_ids: set[int] = {int(s["service_id"]) for s in solver_services if "service_id" in s}
    solver_labor_ids:   set[int] = {
        int(lb["id"])
        for s in solver_services
        for lb in s.get("serviceLabors", [])
        if "id" in lb
    }

    logger.info("solver scope: service_ids=%s  labor_ids=%s", len(solver_service_ids), len(solver_labor_ids))

    baseline_services = baseline.get("data", [])
    result: list[dict[str, Any]] = []
    dropped_services = dropped_labors = 0

    for svc in baseline_services:
        if int(svc.get("service_id", -1)) not in solver_service_ids:
            dropped_services += 1
            logger.info("  drop service=%s — not in solver scope", svc.get("service_id"))
            continue

        original_labors = svc.get("serviceLabors", [])
        kept = [lb for lb in original_labors if int(lb.get("id", -1)) in solver_labor_ids]
        n_dropped = len(original_labors) - len(kept)
        dropped_labors += n_dropped

        if n_dropped:
            logger.info(
                "  service=%s: pruned %s labor(s) not in solver scope",
                svc.get("service_id"), n_dropped,
            )

        if kept:
            result.append({**svc, "serviceLabors": kept})
        else:
            dropped_services += 1
            logger.info("  drop service=%s — no labors remaining after scope filter", svc.get("service_id"))

    logger.info(
        "scope_baseline: services_in=%s services_out=%s services_dropped=%s labors_dropped=%s",
        len(baseline_services), len(result), dropped_services, dropped_labors,
    )
    return {**baseline, "data": result}


def main() -> int:
    _abs = lambda p: p if p.is_absolute() else _PROJECT_ROOT / p
    solver_path  = _abs(SOLVER_INPUT_FILE)
    baseline_path = _abs(BASELINE_FILE)
    output_path  = _abs(OUTPUT_FILE)

    for path, name in [(solver_path, "SOLVER_INPUT_FILE"), (baseline_path, "BASELINE_FILE")]:
        if not path.exists():
            logger.error("%s not found: %s", name, path)
            return 1

    logger.info("solver_input = %s", solver_path)
    logger.info("baseline     = %s", baseline_path)
    logger.info("output       = %s", output_path)

    solver_input = json.loads(solver_path.read_text(encoding="utf-8"))
    baseline     = json.loads(baseline_path.read_text(encoding="utf-8"))

    logger.info("baseline input  → %s", _summary(baseline.get("data", [])))
    logger.info("solver input    → %s", _summary(solver_input.get("data", [])))

    result = scope_baseline(baseline, solver_input)

    logger.info("scoped output   → %s", _summary(result.get("data", [])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("written → %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
