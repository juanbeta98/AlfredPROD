import pandas as pd

from datetime import datetime
# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_tables
from src.config import *
from src.inst_generation_utils import top_service_days, create_artificial_week

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'inst4b'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

def filter_invalid_services(labors_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina servicios incompletos: si algún labor NO es de tipo VEHICLE_TRANSPORTATION
    y no tiene 'address_id', se descarta TODO el servicio.

    Parámetros:
        labors_raw_df (pd.DataFrame): DataFrame con todas las labores,
                                      debe incluir columnas 'service_id',
                                      'labor_category' y 'address_id'.

    Retorna:
        pd.DataFrame: DataFrame filtrado, sin servicios incompletos.
    """
    df = labors_raw_df.copy()

    # Identificar servicios inválidos
    invalid_services = (
        df.loc[
            #(df['labor_category'] != 'VEHICLE_TRANSPORTATION') &
            (df['address_id'].isna())
        ]['service_id']
        .unique()
    )

    print(f"Servicios inválidos encontrados: {len(invalid_services)}")

    # Filtrar fuera todos esos servicios
    df_cleaned = df[~df['service_id'].isin(invalid_services)].copy()

    return df_cleaned

labors_filtered_df = filter_invalid_services(labors_raw_df)

top7_df = top_service_days(labors_filtered_df, city_col="city", date_col="labor_start_date", 
                           top_n=7, starting_year=2025)

# Extra filtering (remove labors of type trailer transport)
labors_filtered_df = labors_filtered_df[labors_filtered_df['labor_name'] != 'Trailer transport']

labors_inst_df, mapping_df = create_artificial_week(labors_filtered_df, top7_df, 
                                                    seed=52, starting_date="2026-01-05")

labors_inst_df.to_csv(f'{data_path}/instances/artif_inst/{instance}/labors_{instance}_df.csv',index=False)
mapping_df.to_csv(f'{data_path}/instances/artif_inst/{instance}/mapping_{instance}.csv',index=False)

print(f'-------- Instance -{instance}- generated successfully --------\n')
