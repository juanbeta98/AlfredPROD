from typing import Any, Dict
from datetime import datetime

import dataclasses
import numpy as np
import pandas as pd


class OutputFormatter:
    """
    Formatter for optimization results into API-ready JSON.
    """

    @staticmethod
    def format(
        results: Any,
        metadata: Dict[str, Any] | None = None,
        request_id: str | None = None,
        status: str = "completed",
    ) -> Dict[str, Any]:
        """
        Formats optimization output into a generic JSON structure.

        Args:
            results: Optimization output (DataFrame, list, dict)
            metadata: Optional metadata from input
            request_id: Optional request identifier
            status: Execution status

        Returns:
            JSON-serializable dict
        """
        if isinstance(results, pd.DataFrame):
            results_payload = OutputFormatter._format_results_df(results)
        else:
            results_payload = results if results is not None else []

        return {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "data": results_payload,
            "metadata": OutputFormatter._sanitize_payload(metadata),
        }

    @staticmethod
    def _format_results_df(df: pd.DataFrame) -> list[Dict[str, Any]]:
        if df.empty:
            return []

        service_columns = {
            "service_id",
            "state",
            "created_at",
            "city",
            "start_address_id",
            "start_address_point",
            "end_address_id",
            "end_address_point",
        }
        labor_columns = {
            "labor_id",
            "labor_type",
            "labor_name",
            "labor_category",
            "schedule_date",
            "labor_sequence",
            "shop_id",
            "shop_address_id",
            "shop_address_point",
        }

        output: list[Dict[str, Any]] = []

        for service_id, group in df.groupby("service_id", sort=False):
            first = group.iloc[0]
            service: Dict[str, Any] = {
                "service_id": OutputFormatter._to_int(service_id),
            }

            if "state" in df.columns:
                service["state"] = OutputFormatter._clean_value(first.get("state"))
            if "created_at" in df.columns:
                service["created_at"] = OutputFormatter._format_dt(first.get("created_at"))

            start_address = OutputFormatter._build_address(
                first,
                id_col="start_address_id",
                point_col="start_address_point",
                city_col="city",
            )
            if start_address:
                service["start_address"] = start_address

            end_address = OutputFormatter._build_address(
                first,
                id_col="end_address_id",
                point_col="end_address_point",
            )
            if end_address:
                service["end_address"] = end_address

            service_labors: list[Dict[str, Any]] = []
            for _, row in group.iterrows():
                labor: Dict[str, Any] = {}

                if "labor_id" in df.columns:
                    labor["id"] = OutputFormatter._clean_value(row.get("labor_id"))
                if "labor_type" in df.columns:
                    labor["labor_id"] = OutputFormatter._clean_value(row.get("labor_type"))
                    labor["labor_type"] = OutputFormatter._clean_value(row.get("labor_type"))
                if "labor_name" in df.columns:
                    labor["labor_name"] = OutputFormatter._clean_value(row.get("labor_name"))
                if "labor_category" in df.columns:
                    labor["labor_category"] = OutputFormatter._clean_value(row.get("labor_category"))
                if "schedule_date" in df.columns:
                    labor["schedule_date"] = OutputFormatter._format_dt(row.get("schedule_date"))
                if "labor_sequence" in df.columns:
                    labor["labor_sequence"] = OutputFormatter._clean_value(row.get("labor_sequence"))

                shop = OutputFormatter._build_shop(row)
                if shop:
                    labor["shop"] = shop

                shop_address = OutputFormatter._build_address(
                    row,
                    id_col="shop_address_id",
                    point_col="shop_address_point",
                )
                if shop_address:
                    labor["shop_address"] = shop_address

                for col in df.columns:
                    if col in service_columns or col in labor_columns:
                        continue
                    if col in {"service_id", "city"}:
                        continue
                    if col in labor:
                        continue
                    labor[col] = OutputFormatter._clean_value(row.get(col))

                service_labors.append(labor)

            service["serviceLabors"] = service_labors
            output.append(service)

        return output

    @staticmethod
    def _build_address(row: pd.Series, *, id_col: str, point_col: str, city_col: str | None = None) -> Dict[str, Any] | None:
        address_id = row.get(id_col) if id_col in row.index else None
        point_value = row.get(point_col) if point_col in row.index else None
        city_value = row.get(city_col) if city_col and city_col in row.index else None

        if pd.isna(address_id) and pd.isna(point_value) and pd.isna(city_value):
            return None

        address: Dict[str, Any] = {
            "id": OutputFormatter._clean_value(address_id),
        }

        point = OutputFormatter._parse_point(point_value)
        if point:
            address["point"] = point

        if city_col:
            address["city"] = OutputFormatter._clean_value(city_value)

        return address

    @staticmethod
    def _build_shop(row: pd.Series) -> Dict[str, Any] | None:
        if "shop_id" not in row.index:
            return None
        shop_id = row.get("shop_id")
        if pd.isna(shop_id):
            return None
        return {"id": OutputFormatter._clean_value(shop_id)}

    @staticmethod
    def _parse_point(value: Any) -> Dict[str, Any] | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        if text.startswith("POINT"):
            cleaned = (
                text.replace("POINT", "")
                .replace("(", "")
                .replace(")", "")
                .strip()
            )
            parts = cleaned.split()
            if len(parts) != 2:
                return None
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                return None
            return {"x": lon, "y": lat, "srid": 4326}
        return None

    @staticmethod
    def _format_dt(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None or value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
            return None
        if value is pd.NA or (isinstance(value, (np.generic,)) and pd.isna(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "isoformat") and not isinstance(value, str):
            return value.isoformat()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    @staticmethod
    def _to_int(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _sanitize_payload(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {str(k): OutputFormatter._sanitize_payload(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OutputFormatter._sanitize_payload(v) for v in value]
        if dataclasses.is_dataclass(value):
            return OutputFormatter._sanitize_payload(dataclasses.asdict(value))
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            return OutputFormatter._sanitize_payload(value.to_dict())
        return OutputFormatter._clean_value(value)
