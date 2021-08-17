# Define the simulation function
def run_sim(simind):

    # Wrap in try-except so that we can report failures. Nothing should fail.
    try:
        
        # Now pull the appropriate cross validation and training inds
        sim_cv_inds = CV_INDS[simind]
        sim_training_inds = TRAINING_INDS[simind]
        
        # Get the training data
        sim_training_x = DESIGN_SPACE[sim_training_inds]
        sim_training_y = FITNESS[sim_training_inds]
        
        # Package training data
        sim_training_data = (sim_training_inds, sim_training_x, sim_training_y)

        # Instantiate the models we will be training
        default_models, _ = prep_input_data(PARAM_DF.copy(), DESIGN_SPACE.shape)

        # Get the model seeds
        model_seeds = build_seeds(default_models, simind)
        assert len(model_seeds) == len(default_models)
        
        # Pass data into the run_mlde function.
        sim_results = run_mlde(default_models,
                                sim_training_data,
                                DESIGN_SPACE,
                                COMBO_TO_IND,
                                n_to_average = 1, 
                                train_test_inds = sim_cv_inds, 
                                progress_pos = None,
                                _return_processed = True,
                                _reshape_x = True,
                                _seeds = model_seeds)

        # Save the simulation results
        save_sim_results(sim_results, ARGS["saveloc"], simind)
    
    except Exception as e:
        
        # Warn user
        print(f"Unexpected error in simulation {simind}: {e}.")
    
# Define a function for running all simulations
def run_all_sims():
    
    # Define the simulations to test
    sim_range = list(range(ARGS["sim_low"], ARGS["sim_high"]))
    
    # If GPU, run in series. Otherwise, run in parallel
    if ARGS["models_used"] == "GPU":
        for ind in tqdm(sim_range, desc = "Sims complete"):
            _ = run_sim(ind)
            
    else:
        with Pool(ARGS["n_jobs"]) as p:
            _ = list(tqdm(p.imap_unordered(run_sim, sim_range),
                          desc = "Sims complete", total = len(sim_range)))
        
# Execute if running as main
if __name__ == "__main__":
    
    # Set environment variables
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Logging level
    os.environ['PYTHONHASHSEED'] = "0" # Seed for more reproducible Keras
    os.environ['MKL_NUM_THREADS'] = "1" # Stop oversubscription
    
    # Import support functions
    from code.simulate.simulation_utils import (parse_arguments, build_seeds,
                                                load_param_df, load_training_inds,
                                                save_sim_results)
    
    # Parse arguments
    ARGS = parse_arguments()
    
    # Load all other 3rd party modules
    import numpy as np
    import pickle
    from multiprocessing import Pool
    from tqdm import tqdm
    
    # Load custom mlde functions/modules
    from code.run_mlde.run_funcs import run_mlde, prep_input_data
    
    # Confirm that the simulation input folder exists. If not, the user has not
    # downloaded the information yet
    SIM_INPUT_DIR = "./SimulationTrainingData/"
    if not os.path.isdir(SIM_INPUT_DIR):
        raise AssertionError("You must download the simulation inputs from "
                             "CaltechData before running this script.")

    # Load the design space
    DESIGN_SPACE = np.load(os.path.join(SIM_INPUT_DIR, 
                                        f"Encodings/{ARGS['encoding']}.npy"))

    # Load the fitness array
    FITNESS = np.load(os.path.join(SIM_INPUT_DIR, "Fitness.npy"))

    # Load the parameter dataframe
    PARAM_DF = load_param_df(ARGS["models_used"], SIM_INPUT_DIR)

    # Load the combo to ind dictionary
    with open(os.path.join(SIM_INPUT_DIR, "FilteredComboToInd.pkl"), "rb") as f:
        COMBO_TO_IND = pickle.load(f)   

    # Load the training indices
    TRAINING_INDS = load_training_inds(ARGS["training_type"],
                                    ARGS["training_specifics"],
                                    ARGS["training_samples"],
                                    SIM_INPUT_DIR)

    # Load the cross validation indices
    with open(os.path.join(SIM_INPUT_DIR, f"CrossValIndices/{ARGS['training_samples']}/cv1.pkl"), "rb") as f:
        CV_INDS = pickle.load(f)
        
    # Run all simulations
    run_all_sims()
