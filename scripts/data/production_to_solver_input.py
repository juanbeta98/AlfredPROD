"""
Convert a production output payload to a solver-ready input.

Rules are applied in order. Each rule is a plain function that receives the
current service list plus keyword context and returns a new service list.
To add a rule: write a function matching the signature and append it to RULES.

Current rules
-------------
1. filter_services_with_alfred
       Keep only services where production assigned ≥1 driver ON THE PLANNING DATE.
       Services whose planning-date labors are all alfred=null were outside the
       scheduling scope and should not be presented to the solver as a benchmark.
       (Multi-day services with prior-day alfred assignments but no planning-date
       assignment are treated as unscheduled and dropped.)

2. strip_alfred_assignments
       Set alfred=null on serviceLabors scheduled on PLANNING_DATE so the
       solver starts from scratch on those labors.  Off-date labors (already-
       executed legs of multi-day services) keep their alfred values; the
       pipeline reads them to reconstruct the vehicle's current location via
       prior_stop_point before filtering them out of the solver input.

Usage (from repo root):
    python scripts/data/production_to_solver_input.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------
INPUT_FILE   = Path("experiments/phase2/cases/file_snapshots/bogota-services-20260218-solucion.json")
OUTPUT_FILE  = Path("experiments/phase2/cases/input_solver.json")
PLANNING_DATE: Optional[str] = "2026-02-18"   # ISO date "YYYY-MM-DD"; None = skip rule 2
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Type alias for rule functions
RuleFn = Callable[..., list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _labor_date(labor: dict[str, Any]) -> Optional[str]:
    """Return the YYYY-MM-DD portion of a labor's schedule_date, or None."""
    sd = labor.get("schedule_date")
    return str(sd)[:10] if sd else None


def _summary(services: list[dict]) -> str:
    labors = sum(len(s.get("serviceLabors", [])) for s in services)
    return f"services={len(services)} labors={labors}"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def rule_filter_services_with_alfred(
    services: list[dict[str, Any]],
    planning_date: Optional[str] = None,
    **_: Any,
) -> list[dict[str, Any]]:
    """
    Rule 1 — Keep only services where production assigned ≥1 driver ON THE PLANNING DATE.

    When planning_date is set, only labors matching that date are examined.
    A service whose planning-date labors are all alfred=null was outside
    production's scheduling scope for this run and should not be presented
    to the solver as a benchmark.

    When planning_date is None, all labors are examined (original behaviour).
    """
    kept, dropped = [], []
    for svc in services:
        labors_to_check = svc.get("serviceLabors", [])
        if planning_date:
            labors_to_check = [lb for lb in labors_to_check if _labor_date(lb) == planning_date]
        has_assignment = any(lb.get("alfred") is not None for lb in labors_to_check)
        (kept if has_assignment else dropped).append(svc)

    for svc in dropped:
        logger.info(
            "  [rule1] drop service=%s — no alfred assignment on %s",
            svc.get("service_id"), planning_date or "any date",
        )
    logger.info("[rule1] filter_services_with_alfred → kept=%s dropped=%s", len(kept), len(dropped))
    return kept


def rule_prune_off_date_labors(
    services: list[dict[str, Any]],
    planning_date: Optional[str] = None,
    **_: Any,
) -> list[dict[str, Any]]:
    """
    Rule 2 — Drop labors not scheduled on PLANNING_DATE.

    Services can carry labors from other dates (e.g. a multi-day service whose
    initial transport happened on a prior day).  Those cross-date labors should
    not enter the solver for a single-day planning run.  Services left with no
    remaining labors are dropped entirely.

    Skipped when planning_date is None.
    """
    if not planning_date:
        logger.info("[rule2] prune_off_date_labors → skipped (no PLANNING_DATE set)")
        return services

    result: list[dict[str, Any]] = []
    total_pruned = 0
    services_dropped = 0

    for svc in services:
        original = svc.get("serviceLabors", [])
        kept     = [lb for lb in original if _labor_date(lb) == planning_date]
        n_pruned = len(original) - len(kept)
        total_pruned += n_pruned

        if n_pruned:
            logger.info(
                "  [rule2] service=%s pruned %s labor(s) not on %s",
                svc.get("service_id"), n_pruned, planning_date,
            )

        if kept:
            result.append({**svc, "serviceLabors": kept})
        else:
            services_dropped += 1
            logger.info(
                "  [rule2] drop service=%s — no labors remaining on %s",
                svc.get("service_id"), planning_date,
            )

    logger.info(
        "[rule2] prune_off_date_labors → planning_date=%s labors_pruned=%s services_dropped=%s",
        planning_date, total_pruned, services_dropped,
    )
    return result


def rule_strip_alfred_assignments(
    services: list[dict[str, Any]],
    planning_date: Optional[str] = None,
    **_: Any,
) -> list[dict[str, Any]]:
    """
    Rule 3 — Set alfred=null on serviceLabors scheduled on PLANNING_DATE.

    Only labors on the planning date are stripped so the solver starts from
    scratch on those.  Off-date labors (already-executed legs of multi-day
    services) keep their original alfred values; they serve as location context
    for the pipeline and should accurately reflect what was executed.

    When planning_date is None, all labors are stripped (original behaviour).
    """
    result: list[dict[str, Any]] = []
    total_stripped = 0

    for svc in services:
        labors = []
        for lb in svc.get("serviceLabors", []):
            is_planning_labor = (planning_date is None) or (_labor_date(lb) == planning_date)
            if is_planning_labor and lb.get("alfred") is not None:
                lb = {**lb, "alfred": None}
                total_stripped += 1
            labors.append(lb)
        result.append({**svc, "serviceLabors": labors})

    logger.info("[rule3] strip_alfred_assignments → assignments_stripped=%s", total_stripped)
    return result


# ---------------------------------------------------------------------------
# Pipeline — append new rule functions here to extend
# ---------------------------------------------------------------------------
RULES: list[RuleFn] = [
    rule_filter_services_with_alfred,
    rule_strip_alfred_assignments,
]


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def transform(
    payload: dict[str, Any],
    *,
    planning_date: Optional[str],
) -> dict[str, Any]:
    """Apply all rules in sequence and return the transformed payload."""
    services = payload.get("data", [])
    logger.info("input  → %s", _summary(services))

    for rule in RULES:
        services = rule(services, planning_date=planning_date)

    logger.info("output → %s", _summary(services))
    return {**payload, "data": services}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    _abs = lambda p: p if p.is_absolute() else _PROJECT_ROOT / p
    input_path  = _abs(INPUT_FILE)
    output_path = _abs(OUTPUT_FILE)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    logger.info("input_file  = %s", input_path)
    logger.info("output_file = %s", output_path)
    logger.info("planning_date = %s", PLANNING_DATE or "(none — off-date labors stripped too)")

    with input_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    result = transform(payload, planning_date=PLANNING_DATE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("written → %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
