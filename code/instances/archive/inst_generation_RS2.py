import pandas as pd
from datetime import datetime, timedelta

import os

from src.data_load import load_tables
from src.config import *

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'instRS2'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(
    data_path, generate_labors=False
)

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
    invalid_services = df.loc[df['address_id'].isna(), 'service_id'].unique()
    print(f"Servicios inválidos encontrados: {len(invalid_services)}")

    df = df[~df['service_id'].isin(invalid_services)]

    # Eliminar labores de tipo 'Trailer transport'
    df = df[df['labor_name'] != 'Trailer transport']

    #  Eliminar labores con schedule_date <= created_at
    df = df[df['schedule_date'] > df['created_at']]

    # Devolver copia limpia
    return df.copy()

labors_filtered_df = filter_invalid_services(labors_raw_df)

# --- 📅 Identify the last full week available ---
labors_filtered_df['labor_start_date'] = pd.to_datetime(labors_filtered_df['labor_start_date'])
max_date = labors_filtered_df['labor_start_date'].max().normalize()

# Align to the last Sunday
last_sunday = max_date - pd.Timedelta(days=max_date.weekday() + 1) if max_date.weekday() != 6 else max_date
last_monday = last_sunday - pd.Timedelta(days=6)

print(f"Using week: {last_monday.date()} → {last_sunday.date()}")

# --- Slice data for that week ---
mask = (labors_filtered_df['labor_start_date'] >= last_monday) & \
       (labors_filtered_df['labor_start_date'] <= last_sunday)
labors_inst_df = labors_filtered_df.loc[mask].copy()

# --- Save ---
out_dir = f'{data_path}/instances/real_inst/{instance}'
os.makedirs(out_dir, exist_ok=True)
labors_inst_df.to_csv(f'{out_dir}/labors_{instance}_df.csv', index=False)

print(f'-------- Instance -{instance}- generated successfully --------\n')
