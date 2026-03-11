from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelParams:
    """
    Shared model parameters used across all optimization algorithms.

    Units:
    - Speeds: km/h
    - Times: minutes
    """

    # Reproducibility
    seed: int = 10

    # Speeds
    # alfred_speed_kmh: float = 30.0            # driver to first point
    alfred_speed_kmh: float = 20.0            # driver to first point
    vehicle_transport_speed_kmh: float = 32 # vehicle transporsvt speed

    # Times (minutes)
    tiempo_previo_min: int = 0        # minutes before schedule_date
    # tiempo_previo_min: int = 30        # minutes before schedule_date
    tiempo_gracia_min: int = 15
    tiempo_alistar_min: int = 30
    tiempo_other_min: int = 30
    tiempo_finalizacion_min: int = 15
    workday_end_str: str = "19:00:00"
    osrm_url: Optional[str] = field(default_factory=lambda: os.environ.get("OSRM_URL"))

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        if self.alfred_speed_kmh <= 0:
            raise ValueError("alfred_speed_kmh must be > 0")

        if self.vehicle_transport_speed_kmh <= 0:
            raise ValueError("vehicle_transport_speed_kmh must be > 0")

        for name, value in {
            "tiempo_previo_min": self.tiempo_previo_min,
            "tiempo_gracia_min": self.tiempo_gracia_min,
            "tiempo_alistar_min": self.tiempo_alistar_min,
            "tiempo_other_min": self.tiempo_other_min,
            "tiempo_finalizacion_min": self.tiempo_finalizacion_min,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Optional: convenience conversions (often useful)
    def kmh_to_m_per_min(self, kmh: float) -> float:
        # 1 km = 1000 m, 1 h = 60 min
        return (kmh * 1000.0) / 60.0

    @property
    def alfred_speed_m_per_min(self) -> float:
        return self.kmh_to_m_per_min(self.alfred_speed_kmh)

    @property
    def vehicle_transport_speed_m_per_min(self) -> float:
        return self.kmh_to_m_per_min(self.vehicle_transport_speed_kmh)
