import pandas as pd
import math



class fn_dir():

    @staticmethod
    def parse_point(point_str):
        """
        Extrae latitud y longitud de un string 'POINT (lon lat)'.
        Retorna (lat, lon) como floats.
        """
        if pd.isna(point_str):
            return None, None
        # Quitar 'POINT (' y ')'
        coords = point_str.strip().lstrip('POINT').strip(' ()').split()
        lon, lat = map(float, coords)
        return lat, lon
    
    @staticmethod
    def haversine_distance(p1, p2):
        """
        Calcula la distancia Haversine en km entre dos puntos en formato 'POINT (lon lat)'.
        """
        lat1, lon1 = fn_dir.parse_point(p1)
        lat2, lon2 = fn_dir.parse_point(p2)
        if None in (lat1, lon1, lat2, lon2):
            return pd.NA
        # Convertir a radianes
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371 * c  # Radio de la Tierra en km
    

