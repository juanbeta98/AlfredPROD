"""CLI entry point for two-solution comparison.

Edit the CONFIGURATION block below, then run from the project root:

    python scripts/analysis/run_comparison.py

All logic lives in src/analysis/compare_solutions.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.compare_solutions import compare
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------
SOL_A   = Path("experiments/phase2/cases/file_snapshots/output_payload_scoped.json")
SOL_B   = Path("experiments/phase2/results/output/output_payload.json")
LABEL_A = "baseline"
LABEL_B = "algorithm"
EVAL_A: Optional[Path] = None
EVAL_B: Optional[Path] = None
OUT_DIR = Path("experiments/phase2/comparisons")
INPUT_FILE: Optional[Path] = Path("experiments/phase2/cases/input_solver.json")
PLANNING_DATE: Optional[str] = "2026-02-18"
METRIC_WARN_THRESHOLD_PCT: float = 5.0
DISTANCE_METHOD: str = DEFAULT_DISTANCE_METHOD
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _abs     = lambda p: p if p.is_absolute() else _PROJECT_ROOT / p
    _abs_opt = lambda p: None if p is None else _abs(p)
    compare(
        path_a=_abs(SOL_A), path_b=_abs(SOL_B),
        label_a=LABEL_A, label_b=LABEL_B,
        eval_path_a=_abs_opt(EVAL_A), eval_path_b=_abs_opt(EVAL_B),
        out_dir=_abs(OUT_DIR),
        input_file=_abs_opt(INPUT_FILE),
        planning_date=PLANNING_DATE,
        metric_warn_threshold_pct=METRIC_WARN_THRESHOLD_PCT,
        distance_method=DISTANCE_METHOD,
    )
