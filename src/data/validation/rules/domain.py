from datetime import timedelta
from typing import Optional

import pandas as pd

from .base import ValidationRule


# ======================================================
# Domain-specific validation rules
# ======================================================

class CreatedBeforeScheduleRule(ValidationRule):
    """
    Validate that 'created_at' occurs a minimum amount of time
    before 'schedule_date'.
    """

    name = "created_before_schedule"
    blocking = True

    def __init__(
        self,
        created_field: str = "created_at",
        schedule_field: str = "schedule_date",
        minimum_delta_hours: float = 2.0,
        timezone_aware: bool = False,
    ):
        """
        Args:
            created_field: Field containing creation timestamp
            schedule_field: Field containing scheduled timestamp
            minimum_delta_hours: Minimum hours between created and scheduled
            timezone_aware: Whether datetimes must be timezone-aware
        """
        super().__init__(
            created_field=created_field,
            schedule_field=schedule_field,
            minimum_delta_hours=minimum_delta_hours,
            timezone_aware=timezone_aware,
        )

        self.created_field = created_field
        self.schedule_field = schedule_field
        self.minimum_delta = timedelta(hours=minimum_delta_hours)
        self.timezone_aware = timezone_aware

    def validate(self, row: pd.Series):
        # --------------------------------------------------
        # Field existence checks
        # --------------------------------------------------
        if self.created_field not in row or self.schedule_field not in row:
            # Let RequiredFieldRule handle missing fields
            return True, None

        created_raw = row[self.created_field]
        schedule_raw = row[self.schedule_field]

        # --------------------------------------------------
        # Null checks
        # --------------------------------------------------
        if pd.isna(created_raw) or pd.isna(schedule_raw):
            return True, None  # Nullability handled elsewhere if needed

        # --------------------------------------------------
        # Parse datetimes safely
        # --------------------------------------------------
        created_at = self._parse_datetime(created_raw)
        schedule_at = self._parse_datetime(schedule_raw)

        if created_at is None or schedule_at is None:
            return (
                False,
                f"Invalid datetime format in '{self.created_field}' or "
                f"'{self.schedule_field}'",
            )

        # --------------------------------------------------
        # Timezone consistency check
        # --------------------------------------------------
        if self.timezone_aware:
            if created_at.tzinfo is None or schedule_at.tzinfo is None:
                return (
                    False,
                    "Datetime fields must be timezone-aware",
                )

        # --------------------------------------------------
        # Business logic check
        # --------------------------------------------------
        if schedule_at - created_at < self.minimum_delta:
            return (
                False,
                f"'{self.created_field}' must be at least "
                f"{self.minimum_delta} before '{self.schedule_field}'",
            )

        return True, None

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    @staticmethod
    def _parse_datetime(value) -> Optional[pd.Timestamp]:
        """
        Safely parse a datetime-like value.
        """
        try:
            return pd.to_datetime(value)
        except Exception:
            return None
