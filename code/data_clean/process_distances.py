import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from time import perf_counter
from tqdm import tqdm

from src.distance_utils import distance
from src.data_load import load_tables, load_instance
from src.config import *
from src.experimentation_config import instance_map

data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
instance = 'instAD2'
instance_type = instance_map(instance)

directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)
labors_real_df = load_instance(data_path, instance, labors_raw_df)

def compute_transport_distances(labors_real_df: pd.DataFrame, method="osrm", timeout=5, checkpoint=True):
    """
    Compute distances grouped by city.
    Saves partial results safely per city to avoid corruption and enable resume.
    """

    dist_dict_by_city = {}
    grouped = labors_real_df.groupby("city")

    # Path to final and temporary files
    output_dir = f"{data_path}/instances/{instance_type}_inst/dist"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/osrm_dist_dict.pkl"

    # Load partial progress if checkpoint exists
    if checkpoint and os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                dist_dict_by_city = pickle.load(f)
            print(f"🟢 Resuming from checkpoint: {len(dist_dict_by_city)} cities loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint ({e}), starting fresh...")

    print(f"\n🌍 Generating OSRM distances for {len(grouped)} cities...\n")

    for city, df_sub in tqdm(grouped, desc="Processing cities"):
        if city in dist_dict_by_city:
            tqdm.write(f"⏩ Skipping {city} (already computed)")
            continue

        start = perf_counter()
        dist_dict = {}

        start_nodes = df_sub[["start_address_point"]].rename(
            columns={"start_address_point": "address_point"}
        )
        end_nodes = df_sub[["end_address_point"]].rename(
            columns={"end_address_point": "address_point"}
        )
        nodes = pd.concat([start_nodes, end_nodes], ignore_index=True).dropna().drop_duplicates()

        # Compute distances for actual transport pairs
        for _, row in tqdm(df_sub.iterrows(), total=len(df_sub), leave=False, desc=f"{city} pairs"):
            sp, dp = row["start_address_point"], row["end_address_point"]
            if pd.isna(sp) or pd.isna(dp):
                continue

            if (sp, dp) not in dist_dict:
                d = distance(sp, dp, method=method, timeout=timeout)
                dist_dict[(sp, dp)] = d
                dist_dict[(dp, sp)] = d

        dist_dict_by_city[city] = dist_dict

        # ✅ Safe write after each city (atomic save)
        tmp_path = output_file + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(dist_dict_by_city, f)
        os.replace(tmp_path, output_file)

        tqdm.write(f"✅ {city}: {round(perf_counter()-start, 2)}s | {len(nodes)} nodes | {len(dist_dict)} distances")

    print("\n------------------ ✅ All distances generated successfully ------------------")
    return dist_dict_by_city


# Run
if __name__ == "__main__":
    print(f'------------------ Generating distances for -{instance}- ------------------')
    real_distances = compute_transport_distances(labors_real_df, method='osrm')
    print('\n------------------ Distances generation complete ------------------')
