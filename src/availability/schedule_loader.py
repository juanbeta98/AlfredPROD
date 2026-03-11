"""
schedule_loader.py — Fetch and prepare the live preassigned schedule state.

Mirrors steps 3–7 of main.py (input acquisition, parsing, driver directory
loading, master data loading, preassigned reconstruction) but stripped of all
output writing and error-reporting side effects.

The resulting ScheduleState is frozen and reused across all slot probes in a
single availability check run.
"""

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.availability.exceptions import ScheduleLoadError
from src.availability.models import ScheduleState
from src.config import Config
from src.data.loading.master_data_loader import MasterData, load_master_data
from src.data.parsing.driver_directory_parser import DriverDirectoryParser
from src.data.parsing.input_parser import InputParser
from src.integration.client import ALFREDAPIClient
from src.optimization.settings.solver_settings import OptimizationSettings

logger = logging.getLogger(__name__)

COLOMBIA_TZ = "America/Bogota"


def load_schedule_state(
    department_code: str,
    schedule_date: date,
    settings: OptimizationSettings,
) -> ScheduleState:
    """
    Fetch the live schedule from the ALFRED API and reconstruct the preassigned
    base for availability probing.

    Args:
        department_code : Department code string ("25", "76", "5", ...).
        schedule_date   : The day to load the schedule for.
        settings        : Optimization settings (distance_method, model_params, etc.).

    Returns:
        ScheduleState with frozen base_labors_df and base_moves_df.

    Raises:
        ScheduleLoadError: If any mandatory step fails.
    """
    try:
        tz = ZoneInfo(COLOMBIA_TZ)
        day_start = datetime(
            schedule_date.year, schedule_date.month, schedule_date.day,
            0, 0, 0, tzinfo=tz,
        )
        day_end = day_start + timedelta(hours=23, minutes=59, seconds=59)
        day_str = schedule_date.isoformat()
        dept_int = int(department_code)

        logger.info(
            "availability_load_schedule department=%s date=%s",
            department_code, day_str,
        )

        # 1. Fetch services from API
        services_client = ALFREDAPIClient(
            endpoint_url=Config.SERVICES_ENDPOINT,
            api_token=Config.API_TOKEN,
            timeout=Config.REQUEST_TIMEOUT,
            max_retries=Config.API_MAX_RETRIES,
        )
        raw_input = services_client.get_optimization_data(
            department=dept_int,
            start_date=day_start,
            end_date=day_end,
        )

        # 2. Parse → DataFrame
        input_df, _ = InputParser.parse(raw_input)
        if input_df is None:
            input_df = pd.DataFrame()

        # 3. Filter to department+day
        if not input_df.empty and "department_code" in input_df.columns:
            dept_str = str(department_code).strip()
            input_df = input_df[
                input_df["department_code"].astype(str).str.strip() == dept_str
            ].copy()

        if not input_df.empty and "schedule_date" in input_df.columns:
            input_df = input_df[
                pd.to_datetime(input_df["schedule_date"], errors="coerce").dt.date
                == schedule_date
            ].copy()

        logger.info(
            "availability_schedule_parsed rows=%d department=%s date=%s",
            len(input_df), department_code, day_str,
        )

        # 4. Fetch driver directory
        driver_client = ALFREDAPIClient(
            endpoint_url=Config.ALFREDS_ENDPOINT,
            api_token=Config.API_TOKEN,
            timeout=Config.REQUEST_TIMEOUT,
            max_retries=Config.API_MAX_RETRIES,
        )
        raw_drivers = driver_client.get_driver_directory(
            active=True,
            schedule_date=schedule_date,
            department=dept_int,
        )
        driver_directory_df = DriverDirectoryParser.parse(raw_drivers)

        logger.info(
            "availability_drivers_loaded drivers=%d department=%s",
            len(driver_directory_df), department_code,
        )

        # 5. Load master data; override directorio_df with live driver directory
        master_data = load_master_data(settings.master_data)
        master_data = MasterData(
            directorio_df=driver_directory_df,
            duraciones_df=master_data.duraciones_df,
            dist_dict=master_data.dist_dict,
        )

        # 6. Reconstruct preassigned schedule (base for all probes)
        base_labors_df = pd.DataFrame()
        base_moves_df = pd.DataFrame()

        if not input_df.empty and "assigned_driver" in input_df.columns:
            has_preassigned = (
                input_df["assigned_driver"].notna()
                & input_df["assigned_driver"].astype(str).str.strip().ne("")
            )
            if has_preassigned.any():
                from src.optimization.common.preassigned import reconstruct_preassigned_state

                base_labors_df, _, base_moves_df, _ = reconstruct_preassigned_state(
                    input_df,
                    directorio_df=master_data.directorio_df,
                    duraciones_df=master_data.duraciones_df,
                    dist_method=settings.distance_method,
                    dist_dict=master_data.dist_dict,
                    model_params=settings.model_params,
                )
                logger.info(
                    "availability_preassigned_reconstructed labors=%d department=%s date=%s",
                    len(base_labors_df), department_code, day_str,
                )

        return ScheduleState(
            base_labors_df=base_labors_df,
            base_moves_df=base_moves_df,
            master_data=master_data,
            settings=settings,
            day_str=day_str,
            department_code=department_code,
        )

    except Exception as exc:
        raise ScheduleLoadError(
            f"Failed to load schedule state for department={department_code} "
            f"date={schedule_date}: {exc}"
        ) from exc
