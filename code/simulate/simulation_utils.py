# Import required modules
import numpy as np
import pandas as pd
import argparse
import os
from time import strftime
from multiprocessing import cpu_count
from itertools import chain

# Import custom objects
from ..params.defaults import CPU_MODELS, GPU_MODELS

# Write a model to index dictionary
MODEL_TO_IND = {f"{major}-{minor}": i for i, (major, minor) 
                in enumerate(chain(CPU_MODELS, GPU_MODELS))}

# Define a maximum seed value (maximum possbile unsigned 32-bit int)
MAX_SEED = 2**32 - 1

# Write a function that determines where the training data is
def load_training_inds(training_type, training_specs, training_size, sim_input_dir):

    # Map location to training inds
    file_loc_dict = {"random": os.path.join(sim_input_dir, "TrainingIndices/Random/AllSampleInds.npy"),
                    "triad": os.path.join(sim_input_dir, f"TrainingIndices/Triad/{training_specs}.npy"),
                    "sim": os.path.join(sim_input_dir, f"TrainingIndices/Simulated/SimulatedData_{training_specs}.npy"),
                    "evmutation": os.path.join(sim_input_dir, f"TrainingIndices/EVmutation/{training_specs}.npy"),
                    "msatransformer": os.path.join(sim_input_dir, f"TrainingIndices/MsaTransformer/{training_specs}.npy")}
    
    # Load the appropriate data
    training_inds = np.load(file_loc_dict[training_type])
        
    # Get the reduced training inds
    reduced_training_inds = training_inds[:, :training_size].copy()
        
    return reduced_training_inds

# Write a function that determines what model dataframe should be used
def load_param_df(models_used, sim_input_dir):
    
    # Get the names of the parameter tables
    param_filenames = {"CPU": "CPU_MldeParameters.csv",
                      "GPU": "GPU_MldeParameters.csv",
                      "LimitedSmall": "LimitedSmall_MldeParameters.csv",
                      "LimitedLarge": "LimitedLarge_MldeParameters.csv",
                      "XGB": "XGB_MldeParameters.csv"}
    desired_file = param_filenames[models_used]
    
    # Load the appropriate data
    return pd.read_csv(os.path.join(sim_input_dir, f"ParamTables/{desired_file}"))

# Define a function for generating random seeds for a simulation
def build_seeds(models, simulation_ind):

    # Build a random number generator for the simulation
    sim_rng = np.random.RandomState(simulation_ind)

    # Get a seed for each of the 22 inbuilt models in the simulation
    all_model_seeds = sim_rng.randint(0, MAX_SEED, size = 22)

    # Get the requested models for this set of simulations
    requested_model_names = [f"{model.major_model}-{model.specific_model}" 
                             for model in models]

    # Associate seeds with the requested models
    requested_model_seeds = [all_model_seeds[MODEL_TO_IND[requested_model]]
                             for requested_model in requested_model_names]
    
    return requested_model_seeds

# Write a function for saving simulation results
def save_sim_results(sim_results, saveloc, simulation_ind):
    
    # Make a folder for the simulation
    sim_folder = os.path.join(saveloc, str(simulation_ind))
    os.mkdir(sim_folder)
    
    # Unpack the results and save
    results_df, summary_df, _, sorted_preds, _ = sim_results
    results_df.to_csv(os.path.join(sim_folder, "PredictionResults.csv"), index = False)
    summary_df.to_csv(os.path.join(sim_folder, "SummaryResults.csv"), index = False)
    np.save(os.path.join(sim_folder, "SortedIndividualPreds.npy"), sorted_preds)

def log_args(args, start_time):
        
    # Log all commands
    all_commands = "\n".join(f"{key}: {val}" for key, val in args.items())
    with open(os.path.join(args["saveloc"], f"{start_time}_Log.txt"), "w") as f:
        f.write(all_commands)

# Define a function for parsing arguments
def parse_arguments():
    
    # Create the parser and add all required arguments
    parser = argparse.ArgumentParser(description = "Run simulated MLDE.")
    parser.add_argument("encoding", type = str, help = "Name of the encoding to use")
    parser.add_argument("training_type", type = str, help = "Training indices to use. 'random', 'triad', 'evmutation', 'msatransformer', or 'sim'")
    parser.add_argument("training_samples", type = int, help = "Number of training points used")
    parser.add_argument("models_used", type = str, 
                        help = "Name of the parameter df to use. Must be 'CPU', 'GPU', 'LimitedSmall', 'LimitedLarge' or 'XGB'")
    parser.add_argument("saveloc", type = str, help = "Directory for storing results.")
    parser.add_argument("--training_specifics", type = str, required = False, default = "random",
                        help = "Training dataset to use when `training_type` is 'triad' or 'sim'")
    parser.add_argument("--sim_low", type = int, required = False, default = 0)
    parser.add_argument("--sim_high", type = int, required = False, default = 2000)
    parser.add_argument("--n_jobs", type = int, required = False, default = cpu_count())
    parser.add_argument("--device", type = str, required = False, default = "0")
                                 
    # Parse arguments
    args = vars(parser.parse_args())
     
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args["device"]
     
    # Make sure the training specifics have been set for non-random simulations
    if args["training_type"] in {"triad", "sim"}:
        assert args["training_specifics"] != "random", "Must set training specifics"
     
    # Get the start time
    start_time = strftime("%Y%m%d-%H%M%S")
    
    # Define the nesting of output folders
    folder_order = (start_time, args["training_type"], args["training_specifics"], 
                    args["encoding"], args["models_used"], str(args["training_samples"]))

    # Create the output directory
    output_dir = args["saveloc"]
    for folder in folder_order:
        
        # Add to the existing path
        output_dir = os.path.join(output_dir, folder)
        os.mkdir(output_dir)
     
    # Record the final output directory
    args["saveloc"] = output_dir
    
    # Save arguments
    log_args(args, start_time)
    
    # Return the parsed dictionary
    return args