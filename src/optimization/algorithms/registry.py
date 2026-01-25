from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Type

from .base import OptimizationAlgorithm

logger = logging.getLogger(__name__)


class AlgorithmNotFoundError(ValueError):
    """Raised when an unknown algorithm name is requested."""


def normalize_algorithm_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise AlgorithmNotFoundError("Algorithm name must be a non-empty string")
    return name.strip().upper()


def _build_registry() -> Mapping[str, Type[OptimizationAlgorithm]]:
    """
    Registry builder.

    Lazy-imports algorithm classes to avoid importing heavy dependencies
    at module import time (helps CLI startup time and test isolation).
    """
    # Local imports to prevent import-time side effects
    from .insert.algorithm import InsertAlgorithm
    from .buffer_fixed.algorithm import BufferFixedAlgorithm
    from .buffer_react.algorithm import BufferReactAlgorithm
    from .alfred.algorithm import AlfredAlgorithm
    from .offline.algorithm import OfflineAlgorithm

    return {
        "INSERT": InsertAlgorithm,
        "BUFFER_FIXED": BufferFixedAlgorithm,
        "BUFFER_REACT": BufferReactAlgorithm,
        "ALFRED": AlfredAlgorithm,
        "OFFLINE": OfflineAlgorithm,
    }


# Build once (still uses lazy imports inside)
_REGISTRY: Mapping[str, Type[OptimizationAlgorithm]] = _build_registry()


def list_algorithms() -> list[str]:
    """Return available algorithm keys."""
    return sorted(_REGISTRY.keys())


def get_algorithm(
    name: str,
    params: Dict[str, Any] | None = None,
) -> OptimizationAlgorithm:
    """
    Create an algorithm instance by name.

    Args:
        name: Algorithm name (case-insensitive)
        params: Resolved parameters (shared + per-algorithm overrides)

    Returns:
        An instantiated OptimizationAlgorithm.

    Raises:
        AlgorithmNotFoundError: if name is not registered
    """
    key = normalize_algorithm_name(name)

    algo_cls = _REGISTRY.get(key)
    if algo_cls is None:
        allowed = ", ".join(list_algorithms())
        raise AlgorithmNotFoundError(
            f"Unknown algorithm '{name}'. Allowed values: {allowed}"
        )

    logger.debug(
        "Instantiating algorithm",
        extra={"algorithm": key, "params_keys": sorted((params or {}).keys())},
    )
    return algo_cls(params=params)
