import unittest

import pandas as pd

from main import (
    _build_assignment_diagnostics_metrics,
    _finalize_assignment_diagnostics,
    _stabilize_results_order,
)


class TestPhase3MainHelpers(unittest.TestCase):
    def test_finalize_assignment_diagnostics_marks_failed_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "service_id": 1,
                    "labor_id": 10,
                    "actual_status": "FAILED",
                    "reassignment_candidate": True,
                },
                {
                    "service_id": 2,
                    "labor_id": 20,
                    "actual_status": "FAILED",
                    "reassignment_candidate": False,
                },
                {
                    "service_id": 3,
                    "labor_id": 30,
                    "actual_status": "COMPLETED",
                    "reassignment_candidate": True,
                    "infeasibility_cause_code": "assignment_failed_unassigned",
                    "infeasibility_cause_detail": "temp",
                    "is_infeasible": True,
                },
            ]
        )
        out = _finalize_assignment_diagnostics(df)
        service1 = out.loc[out["service_id"] == 1].iloc[0]
        service2 = out.loc[out["service_id"] == 2].iloc[0]
        service3 = out.loc[out["service_id"] == 3].iloc[0]

        self.assertTrue(bool(service1["is_infeasible"]))
        self.assertEqual(service1["infeasibility_cause_code"], "reassignment_failed_unassigned")
        self.assertTrue(bool(service2["is_infeasible"]))
        self.assertEqual(service2["infeasibility_cause_code"], "assignment_failed_unassigned")
        self.assertFalse(bool(service3["is_infeasible"]))
        self.assertIsNone(service3["infeasibility_cause_code"])

    def test_build_assignment_diagnostics_metrics_counts(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "service_id": 1,
                    "labor_id": 10,
                    "actual_status": "FAILED",
                    "assigned_driver": pd.NA,
                    "is_infeasible": True,
                    "is_warning": False,
                    "reassignment_candidate": True,
                    "infeasibility_cause_code": "reassignment_failed_unassigned",
                },
                {
                    "service_id": 2,
                    "labor_id": 20,
                    "actual_status": "COMPLETED",
                    "assigned_driver": "123",
                    "is_infeasible": False,
                    "is_warning": True,
                    "warning_code": "intermediate_arrival_delay",
                    "reassignment_candidate": True,
                },
            ]
        )
        metrics = _build_assignment_diagnostics_metrics(df)
        self.assertEqual(metrics["rows_total"], 2)
        self.assertEqual(metrics["rows_infeasible"], 1)
        self.assertEqual(metrics["rows_warning"], 1)
        self.assertEqual(metrics["rows_reassignment_candidate"], 2)
        self.assertEqual(metrics["rows_reassignment_failed"], 1)
        self.assertEqual(metrics["rows_reassignment_success"], 1)
        self.assertEqual(metrics["infeasibility_causes"]["reassignment_failed_unassigned"], 1)
        self.assertEqual(metrics["warning_causes"]["intermediate_arrival_delay"], 1)

    def test_stabilize_results_order_sorts_by_service_sequence(self) -> None:
        df = pd.DataFrame(
            [
                {"service_id": 2, "labor_sequence": 1, "labor_id": 200, "schedule_date": "2026-02-12T10:00:00-05:00"},
                {"service_id": 1, "labor_sequence": 2, "labor_id": 101, "schedule_date": "2026-02-12T11:00:00-05:00"},
                {"service_id": 1, "labor_sequence": 1, "labor_id": 100, "schedule_date": "2026-02-12T09:00:00-05:00"},
            ]
        )
        ordered = _stabilize_results_order(df)
        self.assertEqual(ordered["service_id"].tolist(), [1, 1, 2])
        self.assertEqual(ordered["labor_sequence"].tolist(), [1, 2, 1])


if __name__ == "__main__":
    unittest.main()
