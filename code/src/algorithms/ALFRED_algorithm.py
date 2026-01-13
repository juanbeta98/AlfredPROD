from datetime import datetime, timedelta

import json
import os

from src.config.config import REPO_PATH


def alfred_algorithm_assignment(
    alfred_parameters,
    run_params,
):

    #--------- Upload current solution
    assingment_state = upload_current_state(
        run_params['instance']
    )

    #--------- Filter according to alfred parameters
    



    algorithm = alfred_parameters['algorithm']
    if algorithm == 'OFFLINE':

        iter_args = [
                {
                    "city": city,
                    "fecha": fecha,
                    "iter_idx": i,
                    "df_day": df_day,
                    "dist_dict": dist_dict_city,
                    "directorio_hist_filtered_df": directorio_hist_filtered_df,
                    "duraciones_df": duraciones_df,
                    "distance_method": distance_method,
                    "assignment_type": assignment_type,
                    "alpha": alpha,
                    "instance": instance,
                    "driver_init_mode": driver_init_mode,
                }
                for i in range(1, max_iter + 1)
            ]
        pass

    elif algorithm == 'FIXED_BUFFER':
        pass

    elif algorithm == 'REACT_BUFFER':
        pass

    return 1


def generate_time_intervals(start_time: str, end_time: str, increment: int):
    intervals = [('00:00', start_time)]
    fmt = "%H:%M"

    current = datetime.strptime(start_time, fmt)
    end = datetime.strptime(end_time, fmt)

    while current + timedelta(minutes=increment) <= end:
        lower = current.strftime(fmt)
        upper = (current + timedelta(minutes=increment)).strftime(fmt)
        intervals.append((lower, upper))
        current += timedelta(minutes=increment)

    return intervals


def define_run_algorithm(decision_time):
    if decision_time[1] == '00:00':
        return 'OFFLINE'
    elif decision_time[1] == '11:00':
        return 'REACT_BUFFER'
    else:
        return 'BUFFER_FIXED'
    

def generate_alfred_parameters(
    decision_time,
    **kwargs):

    parameters = {
        'previous_run': decision_time[0],
        'current_time': decision_time[1],
        'algorithm': decision_time
    }

    if kwargs['return_params']:
        return parameters 
    
    else:
        output_dir = os.path.join(
            REPO_PATH,
            'resultados',
            kwargs['experiment_type'],
            kwargs['instance'],
            kwargs['dist_method'],
            'temp_alfred_algorithm'
        )
        
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'ALFRED_params.json'), 'w') as fp:
            json.dump(parameters, fp)


def load_alfred_parameters(**kwargs):

    load_path = os.path.join(
        REPO_PATH,
        'resultados',
        kwargs['experiment_type'],
        kwargs['instance'],
        kwargs['dist_method'],
        'temp_alfred_algorithm'
        )

    with open('strings.json') as f:
        parameters = json.load(f)

    return parameters
