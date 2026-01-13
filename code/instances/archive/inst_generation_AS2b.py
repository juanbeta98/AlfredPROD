import pandas as pd

from datetime import datetime

import os

from src.data_load import load_tables
from src.config import *
from src.inst_generation_utils import top_service_days, create_artificial_week, create_hist_directory

# ------ Load the data ------
data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'instAS2'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

def filter_invalid_services(labors_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra servicios inválidos e inconsistentes.

    Reglas:
        1. Si algún labor no tiene 'address_id', se descarta TODO el servicio.
        2. Se eliminan labores cuyo 'labor_name' sea 'Trailer transport'.
        3. Se eliminan labores cuya fecha de programación ('schedule_date')
           sea anterior o igual a la fecha de creación ('created_at').

    Parámetros:
        labors_raw_df (pd.DataFrame): DataFrame con todas las labores.
            Debe incluir las columnas:
            'service_id', 'address_id', 'labor_name',
            'schedule_date' y 'created_at'.

    Retorna:
        pd.DataFrame: DataFrame filtrado, sin servicios incompletos
                      ni inconsistentes.
    """
    df = labors_raw_df.copy()

    # Identificar y eliminar servicios con address_id faltante
    invalid_services = df.loc[(
        df['address_id'].isna() or
        df['labor_name']=='Trailer transport'), 'service_id'].unique()

    df = df[~df['service_id'].isin(invalid_services)]

    # Eliminar labores de tipo 'Trailer transport'
    # df = df[df['labor_name'] != 'Trailer transport']

    #  Eliminar labores con schedule_date <= created_at
    df = df[df['schedule_date'] > df['created_at']]

    # Conservar solo servicios con una sola labor
    service_counts = df['service_id'].value_counts()
    single_labor_services = service_counts[service_counts == 1].index
    df = df[df['service_id'].isin(single_labor_services)]

    print(f"Labores totales removidos: {len(labors_raw_df) - len(df)}")

    # Devolver copia limpia
    return df.copy()

# ------ Filter invalid services ------
labors_filtered_df = filter_invalid_services(labors_raw_df)

top7_df = top_service_days(labors_filtered_df, city_col="city", date_col="labor_start_date", 
                           top_n=7, starting_year=2025)

labors_inst_df, mapping_df = create_artificial_week(labors_filtered_df, top7_df, 
                                                    seed=52, starting_date="2026-01-05")

# ------ Create historic directory of drivers ------
hist_directory = create_hist_directory(labors_inst_df)

# ------ Ensure output directory exists ------
output_dir = os.path.join(data_path, "instances", "artif_inst", instance)
os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

# ------ Saving instance ------
labors_inst_df.to_csv(os.path.join(output_dir, f"labors_{instance}_df.csv"), index=False)
hist_directory.to_csv(os.path.join(output_dir, "directorio_hist_df.csv"), index=False)
mapping_df.to_csv(os.path.join(output_dir, f"mapping_{instance}.csv"), index=False)

print(f'-------- Instance -{instance}- generated successfully --------\n')
