"""
Run MLDE training and predictions.
"""
# Define the main function
def main():
    
    # Turn off extensive tensorflow readout and restrict sklearn to using 1
    # processor only
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    # Import necessary functions for command line execution
    import argparse
    from time import strftime

    # Import relevant functions from MLDE
    from code.run_mlde.run_funcs import run_mlde_cl, process_args
   
    # Get the directory of this file and define the default parameter location.
    filedir = os.path.dirname(os.path.abspath(__file__))
    default_param_loc = os.path.join(filedir, "Support", "Params", 
                                     "MldeParameters.csv")
            
    # Instantiate argparser
    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument("training_data", help = "Path to csv file containing combinations with fitness")
    parser.add_argument("encoding_data", help = "Path to normalized design space from 'GenereateEncodings.py'")
    parser.add_argument("combo_to_ind_dict", help = "Path to dictionary translating combo to ind from 'GenereateEncodings.py'")
    parser.add_argument("--model_params", help = "Path to MLDE parameters file",
                        required = False, default = default_param_loc)
    parser.add_argument("--output", help = "Location to save MLDE outputs",
                        required = False, default = os.getcwd())
    parser.add_argument("--n_averaged", help = "Number of top models to average when making predictions",
                        required = False, default = 3, type = int)
    parser.add_argument("--n_cv", help = "Number of rounds of cross validation to perform in training",
                        required = False, default = 5, type = int)
    parser.add_argument("--no_shuffle", help = "Set flag to not shuffle cross validation indices",
                        required = False, action = "store_false")
    parser.add_argument("--hyperopt", help = "Set flag to include hyperparameter optimization in MLDE",
                        required = False, action = "store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the arguments
    processed_args = process_args(args)
        
    # Finally, run mlde
    run_mlde_cl(*processed_args, n_to_average = args.n_averaged,
                n_cv = args.n_cv, hyperopt = args.hyperopt, 
                shuffle = args.no_shuffle)
    
# Only execute if we are running as main
if __name__ == "__main__":
    main()