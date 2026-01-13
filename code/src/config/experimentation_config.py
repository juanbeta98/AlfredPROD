import pandas as pd
import numpy as np

# Sarario mínimo: https://tickelia.com/co/blog/actualidad/salario-minimo-colombia/#:~:text=Esto%20significa%20que%2C%20aunque%20el,de%20prestaciones%20y%20seguridad%20social.
# Horas extras: https://actualicese.com/horas-extra-y-recargos/?srsltid=AfmBOoox01zXLaHcVGSO28a38fJRnOZ3zLVS9qOXpCZ3ZeGxS8gn8tyu

codificacion_ciudades = {
                            '149':'BOGOTA', 
                            '1':'MEDELLIN', 
                            '126':'BARRANQUILLA',
                            '150':'CARTAGENA',
                            '844':'BUCARAMANGA',
                            '830':'PEREIRA',
                            '1004':'CALI'
                        }

hyperparameter_selection = {
    'driver_distance':      0.3,
    'driver_extra_time':    0.3,
    'hybrid':               0.3
}

max_iterations = {
    '149': 1000,
        '1': 600,
        '1004': 600,
        '126': 150,
        '150': 150,
        '844': 150,
        '830': 150,
}

# max_iterations = {city:int(max_iter/10) for city,max_iter in max_iterations.items()}
# max_iterations = {city:2 for city, max_iter in max_iterations.items()}

iterations_nums = {city:[int(p * max_iterations[city]) for p in np.linspace(0.2, 1.0, 4)] for city in max_iterations}

algo_colors = {
    "Histórico": "#C0392B",
    'OFFLINE': "#2980B9",
    'INSERT': "#27AE60",
    'BUFFER_FIXED': "#16A085",
    'REACT': "#E67E22",
    'BUFFER_REACT': "#9B59B6",
    'ALFRED': "#FF1493"
}

algo_colors = {
    # unchanged
    "Histórico": "#C0392B",   # deep red
    'OFFLINE': "#2980B9",
    # spectrum (sorted by "coolness")
    "INSERT": "#8B5A2B",       # boring brown
    "BUFFER_FIXED": "#E67E22", # medium orange
    "REACT": "#F1C40F",        # yellow-ish transition (optional midpoint)
    "BUFFER_REACT": "#27AE60", # optimal green
    "ALFRED": "#9B59B6"        # purple
}

def instance_map(instance_name: str) -> str:
    """
    Returns the instance type ("artif" or "real") based on its naming convention.

    Rules:
        - Instances starting with 'instA' → 'artif' (artificial)
        - Instances starting with 'instR' → 'real' (real)
        - Raises ValueError if pattern not recognized.

    Examples:
        >>> get_instance_type("instAD1")
        'artif'
        >>> get_instance_type("instRS3")
        'real'
    """
    # if ((not instance_name.startswith("inst")) or (len(instance_name) < 5)) and (instance_name[0]!='N'):
    #     raise ValueError(f"Invalid instance name: {instance_name}")

    code_letter = instance_name[4].upper()

    if code_letter == "A":
        return "artif"
    elif code_letter == "R":
        return "real"
    elif (instance_name[0].upper() == 'N') or (instance_name[0].upper() == 'V'):
        return 'simu'
    else:
        raise ValueError(f"Unrecognized instance type in: {instance_name}")


def fechas_map(instance_name: str) -> str:
    """
    Returns the instance type ("artif" or "real") based on its naming convention.

    Rules:
        - Instances starting with 'instA' → 'artif' (artificial)
        - Instances starting with 'instR' → 'real' (real)
        - Raises ValueError if pattern not recognized.

    Examples:
        >>> get_instance_type("instAD1")
        'artif'
        >>> get_instance_type("instRS3")
        'real'
    """
    # if ((not instance_name.startswith("inst")) or (len(instance_name) < 5)) and (instance_name[0]!='N'):
    #     raise ValueError(f"Invalid instance name: {instance_name}")

    code_letter = instance_name[4].upper()

    if code_letter == "A":
        return pd.date_range("2026-01-05", "2026-01-11").strftime("%Y-%m-%d").tolist()
    elif code_letter == "R":
        return pd.date_range("2025-07-21", "2025-07-27").strftime("%Y-%m-%d").tolist()
    elif (instance_name[0] == 'N') or (instance_name[0] == 'v'):
        return ['2026-11-11']
    else:
        raise ValueError(f"Unrecognized instance type in: {instance_name}")


