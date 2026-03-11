class AvailabilityError(Exception):
    """Base exception for the availability module."""


class ScheduleLoadError(AvailabilityError):
    """Raised when the live schedule cannot be loaded or parsed."""


class InvalidServiceRequestError(AvailabilityError):
    """Raised when the ServiceRequest is structurally invalid."""
