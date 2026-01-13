import os

from src.utils.utils import build_algos_header, process_instance_selection, safe_run
from src.run.run_baselines import run_online_hist_baseline, run_online_algo_baseline, run_online_algo_baseline_parallel
from src.run.run_ONLINE_static import run_ONLINE_static
from src.run.run_INSERT import run_INSERT
from src.run.run_INSERT_BUFFER import run_INSERT_BUFFER
# # from src.run.run_REACT import run_REACT
from src.run.run_REACT_BUFFER import run_REACT_BUFFER
from src.run.run_ALFRED import run_ALFRED

def main():
    # ====== MANUAL DEBUG CONFIGURATION ======
    DEBUG_MODE = True  # ⬅ Set to False when running from the command line
    DEFAULT_INSTANCE = "instAD3"
    DEFAULT_DISTANCE_METHOD = 'haversine'
    DEFAULT_SAVE = False

    # ====== Run configuration ======
    optimization_obj = 'driver_distance'         # Options: ['hybrid', 'driver_distance', 'driver_extra_time']
    experiment_type = 'online_operation'
    multiprocessing = True
    n_processes = None

    run_historic_baseline = True
    run_algo_baseline = True
    run_online_static_algo = True
    run_INSERT_algo = True
    run_INSERT_BUFFER_algo = True
    run_REACT_algo = True
    run_REACT_BUFFER_algo = True
    
    run_ALFRED_algo = True

    # ====== Instance selection ======    
    instance, distance_method, save_results = process_instance_selection(
        DEBUG_MODE,
        DEFAULT_INSTANCE,
        DEFAULT_DISTANCE_METHOD,
        DEFAULT_SAVE
    )

    # ====== Run header ======
    build_algos_header(
        run_historic_baseline,
        run_algo_baseline,
        run_online_static_algo,
        run_INSERT_algo,
        run_INSERT_BUFFER_algo,
        run_REACT_algo,
        run_REACT_BUFFER_algo,
        run_ALFRED_algo,
        instance,
        optimization_obj,
        distance_method,
        save_results,
        multiprocessing
    )

    # ====== Algorithm running ======
    if run_historic_baseline:
        safe_run(
            'historic baseline',
            run_online_hist_baseline,
            instance,
            distance_method=distance_method,
            save_results=save_results,
            experiment_type=experiment_type
        )
    
    if run_algo_baseline:
        safe_run(
            "algorithm offline baseline",
            run_online_algo_baseline,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type
        )


    if run_online_static_algo:
        safe_run(
            'ONLINE static',
            run_ONLINE_static,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type
        )
    
    if run_INSERT_algo:
        safe_run(
            'INSERT',
            run_INSERT,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results,
            experiment_type=experiment_type
        )
    
    if run_INSERT_BUFFER_algo:
        safe_run(
            'INSERT_BUFFER',
            run_INSERT_BUFFER,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results,
            batch_interval_minutes=30,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type
        )
        
    if run_REACT_algo:
        safe_run(
            'REACT',
            run_REACT_BUFFER,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            time_previous_freeze=0,
            batch_interval_minutes=0,
            save_results=save_results,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type
        )
    
    if run_REACT_BUFFER_algo:
        safe_run(
            'REACT_BUFFER',
            run_REACT_BUFFER,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            time_previous_freeze=0,
            batch_interval_minutes=30,
            save_results=save_results,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type

        )
    
    if run_ALFRED_algo:
        safe_run(
            'ALFRED',
            run_ALFRED,
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            time_previous_freeze=0,
            save_results=save_results,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            experiment_type=experiment_type,
        )

    print(f"\n \n✅ Full pipeline for {instance} completed successfully.\n")
    print("\n" + "=" * 120 + '\n \n')



if __name__ == "__main__":
    main()
