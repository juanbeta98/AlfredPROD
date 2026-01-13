import pandas as pd
import numpy as np

from .base import ValidationRule


# ======================================================
# Generic validation rules (schema-light, reusable)
# ======================================================

class RequiredFieldRule(ValidationRule):
    """
    Validate that a required field exists and is not null/empty.
    """

    name = "required_field"
    blocking = True

    def __init__(self, field: str):
        """
        Args:
            field: Name of the required field
        """
        super().__init__(field=field)
        self.field = field

    def validate(self, row: pd.Series):
        # Field missing entirely
        if self.field not in row:
            return False, f"Missing required field '{self.field}'"

        value = row[self.field]

        # Field present but null
        if pd.isna(value):
            return False, f"Required field '{self.field}' is null"

        # Field present but empty (string / container)
        if isinstance(value, str) and value.strip() == "":
            return False, f"Required field '{self.field}' is empty"

        return True, None


class NonEmptyRowRule(ValidationRule):
    """
    Validate that a row is not entirely empty.
    """

    name = "non_empty_row"
    blocking = True

    def validate(self, row: pd.Series):
        if row.isna().all():
            return False, "Row contains only null values"
        return True, None


class NumericFiniteRule(ValidationRule):
    """
    Validate that numeric values are finite (not NaN / inf).
    Applies to all numeric fields unless restricted.
    """

    name = "numeric_finite"
    blocking = True

    def __init__(self, fields: list[str] | None = None):
        """
        Args:
            fields: Optional list of fields to check.
                    If None, all numeric fields are checked.
        """
        super().__init__(fields=fields)
        self.fields = fields

    def validate(self, row: pd.Series):
        fields_to_check = self.fields or row.index

        for field in fields_to_check:
            if field not in row:
                continue

            value = row[field]
            if isinstance(value, (int, float)):
                if pd.isna(value) or not np.isfinite(value):
                    return False, f"Numeric field '{field}' is not finite"

        return True, None


class AllowedValuesRule(ValidationRule):
    """
    Validate that a field value belongs to an allowed set.
    """

    name = "allowed_values"
    blocking = True

    def __init__(self, field: str, allowed_values: set):
        """
        Args:
            field: Field name to validate
            allowed_values: Set of permitted values
        """
        super().__init__(field=field, allowed_values=allowed_values)
        self.field = field
        self.allowed_values = allowed_values

    def validate(self, row: pd.Series):
        if self.field not in row:
            return True, None  # Let RequiredFieldRule handle missing fields

        value = row[self.field]

        if pd.isna(value):
            return True, None  # Nullability handled elsewhere if needed

        if value not in self.allowed_values:
            return (
                False,
                f"Field '{self.field}' has invalid value '{value}' "
                f"(allowed: {sorted(self.allowed_values)})",
            )

        return True, None
