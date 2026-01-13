import pandas as pd
from datetime import datetime

import os

import argparse

from src.data.data_load import load_tables
from src.config.config import *
from src.utils.inst_generation_utils import (
    top_service_days, 
    create_artificial_dynamic_week, 
    split_static_dynamic,
    create_hist_directory, 
    filter_invalid_services,
    add_labor_sequence,
    build_services_map_df
)

inst_configurations = {
    'instAD2':{
        'min_delay_minutes': 0,
        'only_unilabor_services': False
    },
    'instAD2b':{
        'min_delay_minutes': 0,
        'only_unilabor_services': True
    },
    'instAD3':{
        'min_delay_minutes': 30,
        'only_unilabor_services': False
    },
    'instAD4':{
        'min_delay_minutes': 120,
        'only_unilabor_services': False
    }
}

def AD_inst_generation(**kwargs):
    # ------ Load the data ------
    data_path = f'{REPO_PATH}/data'
    instance = kwargs['instance']
    directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

    # ------ Filter invalid services ------
    labors_filtered_df = filter_invalid_services(
        labors_raw_df, 
        min_delay_minutes=kwargs['min_delay_minutes'],
        only_unilabor_services=kwargs['only_unilabor_services'])
    
    # ------ Define labor sequence order ------
    labors_enriched_df = add_labor_sequence(labors_filtered_df)

    # ------ Find most busy days ------
    top7_df = top_service_days(
        labors_enriched_df, 
        city_col="city", 
        date_col="schedule_date", 
        top_n=7, 
        starting_year=2025
    )

    # ------ Create artificial instance ------
    labors_inst_df, mapping_df = create_artificial_dynamic_week(
        labors_enriched_df, 
        top7_df, 
        seed=52, 
        starting_date="2026-01-05"
    )

    service_location_map_df = build_services_map_df(labors_inst_df)
    labors_inst_df = labors_inst_df.merge(
        service_location_map_df, 
        on=['service_id', 'labor_id'], 
        how='left'
    )

    # ------ Create historic directory of drivers ------
    hist_directory = create_hist_directory(labors_inst_df)

    # ------ Split static and dynamic labors ------
    labors_inst_static_df, labors_inst_dynamic_df = split_static_dynamic(labors_inst_df)

    # ------ Ensure output directory exists ------
    output_dir = os.path.join(data_path, "instances", "artif_inst", instance)
    os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

    # ------ Saving instance ------
    labors_inst_static_df.to_csv(os.path.join(output_dir, f"labors_{instance}_static_df.csv"), index=False)
    labors_inst_dynamic_df.to_csv(os.path.join(output_dir, f"labors_{instance}_dynamic_df.csv"), index=False)
    labors_inst_df.to_csv(os.path.join(output_dir, f"labors_{instance}_df.csv"), index=False)
    hist_directory.to_csv(os.path.join(output_dir, "directorio_hist_df.csv"), index=False)
    mapping_df.to_csv(os.path.join(output_dir, f"mapping_{instance}.csv"), index=False)

    print(f'-------- Instance -{instance}- generated successfully --------\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Artificial Dynamic instance")
    parser.add_argument("--inst", help="Instance name (e.g. AD2b)")

    args = parser.parse_args()
    instance = f'inst{args.inst}'

    AD_inst_generation(instance=instance, **inst_configurations[instance])
