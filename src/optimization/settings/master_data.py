from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MasterDataParams:
    """
    File locations for master data used by algorithms.
    """
    base_dir: str = "data/master_data"
    directorio_stem: str = "directorio"
    duraciones_stem: str = "duraciones"
    dist_dict_stem: str = "dist_dict"
    prefer_parquet: bool = True

    def validate(self) -> None:
        if not self.base_dir:
            raise ValueError("base_dir must be non-empty")
        if not self.directorio_stem:
            raise ValueError("directorio_stem must be non-empty")
        if not self.duraciones_stem:
            raise ValueError("duraciones_stem must be non-empty")
        if not self.dist_dict_stem:
            raise ValueError("dist_dict_stem must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_dir": self.base_dir,
            "directorio_stem": self.directorio_stem,
            "duraciones_stem": self.duraciones_stem,
            "dist_dict_stem": self.dist_dict_stem,
            "prefer_parquet": self.prefer_parquet,
        }
