"""
This module contains functions needed for running MLDE. It includes both the
functions made publicly available and the functions used for running MLDE from
the command line.
"""
# Import third party modules
import os
import csv
import numpy as np
import pandas as pd
import warnings
import pickle
from time import time, strftime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from itertools import chain
from sklearn.model_selection import KFold

# Import custom modules
from Support.RunMlde.MldeClasses import MldeModel
from Support.RunMlde.MldeHyperopt import run_hyperopt
from Support.RunMlde.LossFuncs import mse
from Support.RunMlde.CompleteKerasParams import process_val
from Support.RunMlde.FinalizeX import finalize_x
from Support.RunMlde.TrainAndPredict import train_and_predict
from Support.Params.Defaults import default_training_params, default_model_params

# Write a function that processes results from mlde
def process_results(unprocessed_results, model_names, n_to_average, 
                    combo_to_ind, training_inds, _debug = False):
    """
    Processes the results from MLDE.Support.RunMlde.TrainAndPredict.train_and_predict()
    to (1) sort all results by cross-validation testing error, (2) build compound
    predictions by averaging a given number of top-models together, and (3) returning
    predictions and summary stats in a more human-readable format.
    
    Parameters
    ----------
    unprocessed_results: iterable of tuples
        Iterable containing output from train_and_predict() on multiple models
    model_names: 1d numpy array of str
        The names of the models in the order they were passed in to train_and_predict()
    n_to_average: int
        The number of top models whose predictions should be averaged together
        to generate final predictions
    combo_to_ind: dict
        Dictionary linking combo name to combo index in the full design space. 
        This dictionary is output from 'GenerateEncodings.py'
    training_inds: numpy array
        List of indices giving the locations of training data in the design space
        
    Returns
    -------
    results_df: pd.DataFrame
        Dataframe containing all possible combinations in the design space and their
        predicted values. Also incuded is a flag denoting whether or not the
        given combo was in the training set.
    summary_df: pd.DataFrame
        Dataframe detailing the cross-validation training error achieved by
        each model in the results
    compound_preds: 2D numpy array, shape (N-models x N-preds)
        Rolling average over predictions made by each model, rolling from model
        with lowest cv test error to model with highest cv test error
    sorted_preds: 2D numpy array, shape (N-models x N-preds)
        Predictions made by each model, sorted from model with the lowest cv
        test error to model with the highest cv test error
    sorted_stds:
        Standard deviations on cross validation predictions made by each model,
        sorted from model with the lowest cv test error to model with the
        highest cv test error
    """
    # Unpack the unprocessed results
    all_train_loss, all_test_loss, all_preds, all_stds = zip(*unprocessed_results)
    
    # Convert to numpy arrays
    all_train_loss = np.array(all_train_loss)
    all_test_loss = np.array(all_test_loss)
    all_preds = np.stack(all_preds)
    all_stds = np.stack(all_stds)
    
    # Get order models from best to worst test loss
    sorted_inds = np.argsort(all_test_loss)
        
    # Sort all other arrays accordingly
    sorted_train_loss = all_train_loss[sorted_inds]
    sorted_test_loss = all_test_loss[sorted_inds]
    sorted_preds = all_preds[sorted_inds]
    sorted_stds = all_stds[sorted_inds]
    sorted_model_names = model_names[sorted_inds]
    
    # Build a dataframe which summarizes results
    summary_list = [[model, train_loss, test_loss] for model, train_loss, test_loss in 
                    zip(sorted_model_names, sorted_train_loss, sorted_test_loss)]
    summary_df = pd.DataFrame(summary_list, 
                              columns = ["ModelName", "cvTrainingError", "cvTestingError"])
    
    # Generate compound predictions
    cumulative_preds = np.cumsum(sorted_preds, axis = 0)
    compound_preds = np.empty_like(sorted_preds)
    for i in range(len(compound_preds)):
        compound_preds[i] = cumulative_preds[i] / (i + 1)
    
    # Pull the requested averaged value
    mlde_preds = compound_preds[n_to_average - 1]
    
    # Convert the training inds array to a set. Reverse direction of combo_to_ind
    # dictionary
    training_ind_set = set(training_inds)
    ind_to_combo = {ind: combo for combo, ind in combo_to_ind.items()}
    
    # Construct a dataframe which will contain all predictions for all possible
    # combos
    df_list = [[ind_to_combo[i], pred_fitness] for 
               i, pred_fitness in enumerate(mlde_preds)]
    results_df = pd.DataFrame(df_list, columns = ["AACombo", "PredictedFitness"])
    
    # Add a column giving whether this combo was in the training data or not
    detail_col = ["YES" if i in training_ind_set else "NO"
                  for i in range(len(results_df))]
    results_df["InTrainingData?"] = detail_col
    
    # Sort results based on predicted fitness
    results_df.sort_values(by = "PredictedFitness", ascending = False,
                           inplace = True)
    
    # Return all results
    if _debug:
        return (all_train_loss, all_test_loss, all_preds, all_stds,
                sorted_train_loss, sorted_test_loss, sorted_preds, sorted_stds,
                sorted_model_names, compound_preds, results_df, summary_df)
    else:
        return (results_df, summary_df, compound_preds, sorted_preds, sorted_stds)

# Write a function that extracts training data from the design space
def get_training_from_design(training_data, combo_to_ind, normalized_design_space):
    """
    Given a pandas DataFrame containing the training combos and measured fitness
    values, return the indices of these combos, the embeddings at those indices,
    and the measured fitness, each as separate numpy arrays
    
    Parameters
    ----------
    training_data: pd.DataFrame
        Dataframe containing the training combos and measured fitness values
    combo_to_ind: dict
        Dictionary linking a combination to its index in the full design space.
        This is generated during GenerateEncodings.py
    normalized_design_space: 3D numpy array
        The full design space, mean-centered and unit-scaled. This is generated
        during GenerateEncodings.py
        
    Returns
    -------
    training_inds: 1D numpy array
        Numpy array giving the indices of the amino acid combinations in the 
        full design space
    embedded_combos: 3D numpy array
        Numpy array containing the embeddings for the training data combos
    training_fitness: 1D numpy array
        Numpy array containing the labels (fitness) for training combinations
    """
    # Extract the training data from the design space
    training_combos = training_data.AACombo.values
    training_inds = np.array([combo_to_ind[combo] for combo in training_combos])
    embedded_combos = normalized_design_space[training_inds]
    training_fitness = training_data.Fitness.values
    
    # Return relevant extracted information
    return training_inds, embedded_combos, training_fitness

# Write a function that runs MLDE starting from a list of sampled combinations
# (with associated fitness), a full set of normalized embeddings, the 
# appropriate combo-to-ind dict, and a list of MldeModel classes
def run_mlde(mlde_models, training_data, normalized_design_space,
             combo_to_ind, n_to_average = 3, train_test_inds = None,
             progress_pos = 0, _return_processed = True, 
             _debug = False, _reshape_x = False):
    """
    Runs MLDE starting from a list of gpu- and cpu-bound MldeModel instances
    and reported training data. Designed as a method to enable users to perform
    MLDE using a custom model.
    
    Parameters
    ----------
    mlde_models: iterable of MldeModel instances
        An iterable of pre-instantiated MldeModels.
        These models will be run in series.
    training_data: pd.DataFrame
        Dataframe containing the columns 'AACombo' and 'Fitness'. The combo given
        by 'AACombo' will be embedded using the space given by normalized_design_space.
        This embedding will be the features used for MLDE, while the associated
        fitness will be the labels.
    normalized_design_space: Numpy array
        This is the full design space output from GenerateEncodings.py. Before use,
        this design space should be reshaped to fit whatever model is being trained.
        For instance, the design space from GenerateEncodings.py is 3D, while
        sklearn models inbuilt in MLDE require a 2D input; the last two dimensions
        should be flattened before training an sklearn model.
    combo_to_ind: dict
        The dictionary output by GenerateEncodings.py that links a combination
        name to its index in the normalized_design_space
    n_to_average: int: default = 3
        Number of top models from which to average the predictions when generating
        predictions on the design space
    train_test_inds: list of lists: 
        Cross-validation indices for the run
    progress_pos: int
        Passed to tqdm. Gives the location of the process bar.
        
    Returns
    -------
    results_df: pd.DataFrame
        Dataframe containing all possible combinations in the design space and their
        predicted values. Also incuded is a flag denoting whether or not the
        given combo was in the training set.
    summary_df: pd.DataFrame
        Dataframe detailing the cross-validation training error achieved by
        each model in the results
    compound_preds: 2D numpy array, shape (N-models x N-preds)
        Rolling average over predictions made by each model, rolling from model
        with lowest cv test error to model with highest cv test error
    sorted_preds: 2D numpy array, shape (N-models x N-preds)
        Predictions made by each model, sorted from model with the lowest cv
        test error to model with the highest cv test error
    sorted_stds:
        Standard deviations on cross validation predictions made by each model,
        sorted from model with the lowest cv test error to model with the
        highest cv test error
    """
    # If progress pos is None, disable tqdm
    if progress_pos is None:
        disable = True
        progress_pos = 0
    else:
        disable = False
    
    # Extract training data from the design space
    if isinstance(training_data, tuple):
        training_inds, embedded_combos, training_fitness = training_data
    else:
        (training_inds, embedded_combos,
        training_fitness) = get_training_from_design(training_data, combo_to_ind,
                                                    normalized_design_space)
        
    # Make sure the training ids, training fitness, and embedded combos are all
    # the same length
    assert len(training_inds) == len(embedded_combos)
    assert len(training_inds) == len(training_fitness)
           
    # Get the model names
    model_names = np.array([f"{model.major_model}-{model.specific_model}" 
                            for model in mlde_models])
            
    # Package args for both GPU and CPU
    model_args = [[model, train_test_inds] for model in mlde_models]

    # Now run all models
    n_models = len(model_names)
    results = [None for _ in range(n_models)]
    for i, (model, train_test_inds) in tqdm(enumerate(model_args), 
                                          desc = "Default Training",
                                          position = progress_pos, total = n_models,
                                          leave = False, disable = disable):
        results[i] = train_and_predict(model, 
                                       sampled_x = embedded_combos,
                                       sampled_y = training_fitness, 
                                       x_to_predict = normalized_design_space,
                                       train_test_inds = train_test_inds,
                                       _reshape_x = _reshape_x)
    
    # If we are debugging, return a lot of information
    if _debug:
        processed_results = process_results(results, model_names, n_to_average, 
                                            combo_to_ind, training_inds)
        return (training_inds, embedded_combos, training_fitness, model_names,
                results, processed_results, model_args)
    
    # Process the results and return
    if _return_processed:
        processed_results = process_results(results, model_names, n_to_average, 
                                            combo_to_ind, training_inds)
        return processed_results
    
    # Otherwise, we are going to be moving along to hyperopt, so no processing yet
    else:
        return results, model_names
    
# Write a function that saves processed results
def save_results(processed_results, output_location):
    """
    Saves results from process_results() to a given output location.
    
    Parameters
    ----------
    processed_results: tuple
        Output from process_results()
    output_location: str
        Directory in which to save the results
        
    Returns
    -------
    None. Savenames associated with process_results() outputs are below:
        - results_df --> PredictedFitness.csv
        - summary_df --> LossSummaries.csv
        - compound_preds --> CompoundPreds.npy
        - individual_preds --> IndividualPreds.npy
        - sorted_stds --> PredictionStandardDeviation.npy
        
    Example
    --------
    save_results(process_results(*args), save_directory)
    """
    # Unpack processed results
    (results_df, summary_df, compound_preds, 
     sorted_preds, sorted_stds) = processed_results
    
    # Save csvs
    results_df.to_csv(os.path.join(output_location, "PredictedFitness.csv"), index = False)
    summary_df.to_csv(os.path.join(output_location, "LossSummaries.csv"), index = False)
    
    # Save numpy arrays
    np.save(os.path.join(output_location, "CompoundPreds.npy"), compound_preds)
    np.save(os.path.join(output_location, "IndividualPreds.npy"), sorted_preds)
    np.save(os.path.join(output_location, "PredictionStandardDeviation.npy"), sorted_stds)    
     
# Write a function that performs hyperparameter optimization in parallel
def run_hyperopt_mlde(model_data, embedded_combos, training_fitness, 
                      normalized_design_space, train_test_inds, progress_pos = 1):
    """
    Function for performing hyperparameter optimization on inbuilt MLDE models.
    
    Parameters
    ----------
    model_data: iterable
        The input data for models that are not
    embedded_combos: 3D numpy array
        Numpy array containing the embeddings for the training data combos
    training_fitness: 1D numpy array
        Numpy array containing the labels (fitness) for training combinations
    normalized_design_space: Numpy array
        This is the full design space output from GenerateEncodings.py. Before use,
        this design space should be reshaped to fit whatever model is being trained.
        For instance, the design space from GenerateEncodings.py is 3D, while
        sklearn models inbuilt in MLDE require a 2D input; the last two dimensions
        should be flattened before training an sklearn model.
    train_test_inds: list of lists: 
        Cross-validation indices for the run
    progress_pos: int
        Passed to tqdm. Gives the location of the process bar.
        
    Returns
    -------
    results: tuple of tuples
        Results from the function MLDE.Support.RunMlde.MldeHyperopt.run_hyperopt()
        for all input models
    """    
    # Run hyperopt in series for GPU-based models
    n_models = len(model_data)
    results = [None for _ in range(n_models)]
    for i, (major_model, specific_model, hyperopt_rounds) in tqdm(enumerate(model_data),
                                                                  desc = "Hyperopt",
                                                                  position = progress_pos,
                                                                  total = n_models, 
                                                                  leave = False):
        
        # Finalize x shape
        sampled_x = finalize_x(major_model, specific_model, embedded_combos)
        x_to_predict = finalize_x(major_model, specific_model, normalized_design_space)
        
        # If we do not have >0 hyperopt rounds, return filler info
        if hyperopt_rounds <= 0:
            
            # Define the dud dataframe
            columns = ["MajorModel", "SpecificModel", "HyperRound", "RunTime",
                       "TrainErr", "TestErr", "Hyper", "HyperVal"]
            dud_df = pd.DataFrame([[major_model, specific_model, hyperopt_rounds,
                                   0, np.inf, np.inf, np.nan, "NoHyperoptPerformed"]],
                                  columns = columns)
            
            # Record the dud results from train and predict
            n_to_predict = len(x_to_predict)
            dud_tp = (np.inf, np.inf, np.zeros(n_to_predict), np.zeros(n_to_predict))
            results[i] = (dud_df, dud_tp)
            continue
        
        # Pull the default training params for the model
        training_params = default_training_params[major_model].copy()
        
        # Run hyperparameter optimization
        results[i] = run_hyperopt(major_model, specific_model, 
                                        training_params, 
                                        sampled_x = sampled_x,
                                        sampled_y = training_fitness,
                                        x_to_predict = x_to_predict,
                                        eval_metric = mse, 
                                        train_test_inds = train_test_inds, 
                                        hyperopt_rounds = hyperopt_rounds)
                          
    return results
     
# Write a function that instantiates a set of MldeModels based on input parameters
def prep_input_data(parameter_df, x_shape): 
    """
    Given the data input to run_mlde_cl, build all args needed for running both
    default and hyperparameter optimization functions. This means instantiating
    a number of model instances with inbuilt default parameters (for passage 
    into run_mlde) as well as packaging args needed for run_hyperopt_mlde()
    
    Parameters
    ----------
    parameter_df: pd.DataFrame
        Dataframe derived from MLDE.Support.Params.MldeParameters.csv, containing
        only those models for which Include is True.
    x_shape: tuple
        Shape of the design space (should be 3D)
        
    Returns
    -------
    mods_for_default: Iterable of MldeModel instances
        MldeModel instances to be passed into run_mlde()
    hyperopt_args: Iterable of tuples
        Arguments to pass into run_hyperopt_mlde()
    """
    # Make sure the shape is 3D
    assert len(x_shape) == 3, "Input shape should be 3D"
    
    # Create an empty list in which to store objects
    n_mods = len(parameter_df)
    mods_for_default = [None for _ in range(n_mods)]
    hyperopt_args = [None for _ in range(n_mods)]
    for i, (_, row) in enumerate(parameter_df.iterrows()):
        
        # Pull info needed to instantiate model
        major_model = row["ModelClass"]
        specific_model = row["SpecificModel"]
        
        # Define the model and training parameters
        if major_model == "Keras":
            
            # Pull the appropriate model parameters
            temp_params = default_model_params[major_model][specific_model].copy()
            model_params = {}
            
            # Add input_shape as a parameter
            if specific_model in {"OneConv", "TwoConv"}:
                final_shape = x_shape[1:]
                finalized_x_shape = x_shape
            else:
                final_shape = (np.prod(x_shape[1:]),)
                finalized_x_shape = (x_shape[0], final_shape[0])
            model_params["input_shape"] = final_shape
            
            # Loop over the model parameters and update appropriately
            for key, val in temp_params.items():
                
                # Append the new value to model_params
                model_params[key] = process_val(key, val, finalized_x_shape)
                            
        else:
            model_params = default_model_params[major_model][specific_model].copy()
        
        # Instantiate a model with default parameters
        mods_for_default[i] = MldeModel(major_model, specific_model,
                                        model_params = model_params,
                                        training_params = default_training_params[major_model],
                                        eval_metric = mse)
        
        # Package args for hyperopt
        hyperopt_args[i] = (major_model, specific_model, row["NHyperopt"])
    
    # Return the instantiated models and the hyperopt args
    return mods_for_default, hyperopt_args
    
# Write a function that handles both default training and hyperparameter optimization
def run_mlde_cl(training_data, normalized_design_space, combo_to_ind, 
                model_info_df, output, n_to_average = 3, n_cv = 5,
                hyperopt = True, shuffle = True, _debug = False):
    """
    Function called when performing MLDE from the command line. 
    
    Parameters
    ----------
    training_data: pd.DataFrame
        Dataframe containing the columns 'AACombo' and 'Fitness'. The combo given
        by 'AACombo' will be embedded using the space given by normalized_design_space.
        This embedding will be the features used for MLDE, while the associated
        fitness will be the labels.
    normalized_design_space: Numpy array
        This is the full design space output from GenerateEncodings.py. Before use,
        this design space should be reshaped to fit whatever model is being trained.
        For instance, the design space from GenerateEncodings.py is 3D, while
        sklearn models inbuilt in MLDE require a 2D input; the last two dimensions
        should be flattened before training an sklearn model.
    combo_to_ind: dict
        The dictionary output by GenerateEncodings.py that links a combination
        name to its index in the normalized_design_space
    model_info_df: pd.DataFrame
        Dataframe loaded from MLDE.Support.Params.MldeParameters.csv
    output: str
        Path to location where output will be saved
    n_to_average: int: default = 3
        Number of top models from which to average the predictions when generating
        predictions on the design space
    n_cv: int: default = 5
        Number of rounds of cross validation performed during training
    hyperopt: bool: default = True
        Whether or not to perform hyperparameter optimization
    shuffle: bool: default = True
        Whether or not to shuffle cross-validation indices
        
    Returns
    -------
    None. Saves the output of process_results().
    """
    # Generate the cross-validation indices that will be used throughput this run
    kfold_splitter = KFold(n_splits = n_cv, shuffle = shuffle)
    train_test_inds = list(kfold_splitter.split(training_data.Fitness.values))
    
    # Process training data
    (training_inds, embedded_combos, 
     training_fitness) = get_training_from_design(training_data, combo_to_ind,
                                                  normalized_design_space)
    
    # Limit the input dataframe to those which we want to include
    limited_df = model_info_df.loc[model_info_df.Include, :]
    
    # If we have requested averaging more models than will be tested, 
    # drop the value of n_to_average
    n_models = len(limited_df)
    if n_to_average > n_models:
        warnings.warn(f"Requested averaging {n_to_average}, but only {n_models} will be trained. Averaging all models.")
        n_to_average = n_models
    
    # Identify gpu and cpu models
    model_info = limited_df.copy()
    
    # Instantiate gpu and cpu models. Package info for hyperparameter optimization.
    default_mods, mod_hyperopts = prep_input_data(model_info, embedded_combos.shape)
    
    # Run all default models
    mlde_results, model_names = run_mlde(default_mods, training_data,
                                          normalized_design_space,
                                          combo_to_ind, n_to_average = n_to_average,
                                          train_test_inds = train_test_inds,
                                           progress_pos = 0, _return_processed = False,
                                            _reshape_x = True)
    
    # Run all hyperopt models if the flag is thrown, then process both default
    # and hyperoptimized models
    if hyperopt:
        hyperopt_results = run_hyperopt_mlde(mod_hyperopts,
                                             embedded_combos,
                                             training_fitness,
                                             normalized_design_space,
                                             train_test_inds,
                                             progress_pos = 0)
               
        # Combine hyperparameter results with default results
        mlde_results = combine_results(mlde_results, hyperopt_results, output)
    
    # If debugging, don't save results. Just return them.
    if _debug:
        return model_names, train_test_inds
    
    # Process and save results
    save_results(process_results(mlde_results, model_names, n_to_average,
                                 combo_to_ind, training_inds), output)
                
# Write a function that combines results from default and hyperopt training
def combine_results(default_results, hyperopt_data, output, _debug = False):
    """
    Combines the results from the default model training and hyperparameter 
    optimization, using the results from the best model identified between the
    two processes. 
    
    Parameters
    ----------
    default_results: Tuple
        The results from run_mlde()
    hyperopt_results: Tuple
        The results from run_hyperopt_mlde()
    output: str
        Path to location where output will be saved
        
    Returns
    -------
    compiled_results: Tuple
        Results formatted as if they were produced by 
        MLDE.Support.RunMlde.TrainAndPredict.train_and_predict(). The results for
        the model with the lowest test error between default hyperparameters and
        hyperopt is returned.
    """
    # Unpack hyperopt results
    trial_info, hyperopt_results = zip(*hyperopt_data)
    
    # Concatenate and save trial info
    concatenated_trials = pd.concat(trial_info)
    
    if not _debug:
        concatenated_trials.to_csv(os.path.join(output, "HyperoptInfo.csv"), 
                                index = False)
    
    # Create lists in which we will store combined data
    n_results = len(default_results)
    assert len(default_results) == len(hyperopt_results)
    compiled_results = [None for _ in range(n_results)]
    
    # Loop over the default and hyperopt results
    for i, (default_result, hyperopt_result) in enumerate(zip(default_results, hyperopt_results)):
        
        # Identify test errors
        default_test_err = default_result[1]
        opt_test_err = hyperopt_result[1]
        
        # Record the results of whichever has lower error
        if default_test_err <= opt_test_err:
            compiled_results[i] = default_result
        else:
            compiled_results[i] = hyperopt_result
            
    # Return the compiled results
    if _debug:
        return compiled_results, concatenated_trials
    else:
        return compiled_results

# Write a function that processes input arguments to run_mlde_cl
def process_args(args):
    """
    This function is called within the ExecuteMlde.py script. It's primary purpose
    is to load the relevant data input on the command line. It also generates the
    output location.
    
    Parameters
    ----------
    args: argparser instance
        The argparser instance after self.parse() has been called. This contains
        all input information from the command line.
    
    Returns
    -------
    training_data_df: pd.DataFrame
        Dataframe containing the columns 'AACombo' and 'Fitness'. The combo given
        by 'AACombo' will be embedded using the space given by normalized_design_space.
        This embedding will be the features used for MLDE, while the associated
        fitness will be the labels.
    design_space: Numpy array
        This is the full design space output from GenerateEncodings.py. Before use,
        this design space should be reshaped to fit whatever model is being trained.
        For instance, the design space from GenerateEncodings.py is 3D, while
        sklearn models inbuilt in MLDE require a 2D input; the last two dimensions
        should be flattened before training an sklearn model.
    combo_to_ind: dict
        The dictionary output by GenerateEncodings.py that links a combination
        name to its index in the normalized_design_space
    mlde_params: pd.DataFrame
        Dataframe loaded from MLDE.Support.Params.MldeParameters.csv
    output_loc: str
        Path to location where output will be saved
    """
    # Load training data
    training_data_df = pd.read_csv(args.training_data)
    
    # Load the normalized design space
    design_space = np.load(args.encoding_data)
    
    # Assert that it is 3D
    assert len(design_space.shape) == 3, "Input tensor not 3D"
    
    # Load the combo to ind dict
    with open(args.combo_to_ind_dict, "rb") as f:
        combo_to_ind = pickle.load(f)
        
    # Load the mlde parameters file
    mlde_params = pd.read_csv(args.model_params)
        
    # Construct the output location
    output_loc = os.path.join(args.output, strftime("%Y%m%d-%H%M%S"))
    os.mkdir(output_loc)
    
    # Return finalized args
    return (training_data_df, design_space, combo_to_ind,
            mlde_params, output_loc)