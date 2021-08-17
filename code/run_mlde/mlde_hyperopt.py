"""
This module contains all functions needed for performing hyperparameter optimization
on the inbuilt model classes. These functions are not intended for private use only.
"""
# Import third party modules
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from time import time
from functools import partial
import numpy as np
import pandas as pd

# Import custom modules
from .mlde_classes import MldeModel
from ..params.search_spaces import SEARCH_SPACES, SPACE_BY_MODEL, CATEGORICAL_PARAMS
from .loss_funcs import mse
from .train_and_predict import train_and_predict
from .complete_keras_params import process_val

# Write a function that handles some finicky details of hyperparameter optimization
# with LinearSVR and Linear
def handle_linear_exceptions(model_params, major_model, specific_model):
    """
    sklearn LinearSVR requires that 'squared_loss_insensitive' be used when
    'dual == False'. sklearn Linear takes no parameters, but for programmatic 
    sake of ease it is passed into hyperparameter optimization with a dummy
    parameter. This function forces there to be no model parameters for Linear.
    
    Parameters
    ----------
    model_params: dict
        The model params for the given major_model and specific_model
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
        
    Returns
    -------
    updated_model_params: dict
        If this is anything other than sklearn-regressor LinearSVR or Linear,
        the input model_params is returned. Otherwise, updated model params with 
        appropriate parameters is returned.
    """
    # Some changes for sklearn-regressor:
    if major_model == "sklearn-regressor":
        
        # If this is LinearSVR model, add appropriate parameters
        if specific_model == "LinearSVR":
        
            # If we have dual set to True, then we must have loss = "epsilon_insensitive"
            if not model_params["dual"]:
                model_params["loss"] = "squared_epsilon_insensitive"
                
        # If this is a Linear model, just return an empty dictionary
        if specific_model == "Linear":
            model_params = {}
            
    return model_params

# Define a function that completes model parameters
def space_to_model_params(space, space_names, major_model, 
                          specific_model, x_shape):
    """
    The outputs of search spaces are not always in the correct format for training
    a model. This function converts the output format to the appropriate format
    for training.
    
    Parameters
    ----------
    space: iterable of numeric
        The output parameter values from some hyperopt search space
    space_names: iterable of str
        The variable names of each parameter in space
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
    x_shape: tuple
        Gives the shape of the input x-values for a model. This is used to
        calculate the appropriate conversions from percentile to integer for
        keras parameters
        
    Returns
    -------
    updated_model_params: dict
        Search space output values converted to the appropriate format for training
    """
    # Package into a dictionary and convert datatypes as appropriate
    model_params = {var_name: process_val(var_name, var_val, x_shape)
                    for var_name, var_val in zip(space_names, space)}
                
    # Add the input shape on to model_params if this is a keras model
    if major_model == "Keras":
        model_params["input_shape"] = x_shape[1:]
                
    # Return model params
    return handle_linear_exceptions(model_params, major_model, specific_model)

# Define a function that processes the best parameters and converts them to 
# the appropriate data types
def process_best(best_dict, major_model, specific_model, x_shape):
    """
    The best parameters returned from hyperopt.fmin() are not necessarily in
    the correct format for training downstream models. This function converts
    output "best_params" to the correct format.
    
    Parameters
    ----------
    best_dict: dict
        The parameters output by hyperopt.fmin() post hyperparameter optimization
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
    x_shape: tuple
        Gives the shape of the input x-values for a model. This is used to
        calculate the appropriate conversions from percentile to integer for
        keras parameters
        
    Returns
    -------
    updated_model_params: dict
        best_dict values converted to the appropriate format for training
    """
    # Convert choice indices back to the correct values
    for var_name, var_val in best_dict.items():
        
        # If the variable name is a choice, convert var_val
        if var_name in CATEGORICAL_PARAMS:
            best_dict[var_name] = CATEGORICAL_PARAMS[var_name][var_val]
    
    # Redefine the dictionary
    formatted_dict = {key: process_val(key, val, x_shape) for key, val in best_dict.items()}
    
    # Add input_shape if we are working with Keras
    if major_model == "Keras":
        formatted_dict["input_shape"] = x_shape[1:]
    
    # Handle the problems with some of the linear models, then return
    return handle_linear_exceptions(formatted_dict, major_model, specific_model)

# Define a function which extracts information from a trials object
def process_trials(trials, major_model, specific_model):
    """
    MLDE stores the results of every round of hyperparameter optimization. This
    function takes a "hyperopt.trials" object as input post hyperparameter 
    optimization and returns a dataframe detailing each round of hyperparameter
    optimization.
    
    Parameters
    ----------
    trials: hyperopt.Trials object
        hyperopt.Trials object post training with hyperopt.fmin()
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
        
    Returns
    -------
    trials_df: pandas.DataFrame object
        A dataframe detailing each round of hyperparameter optimization
    """
    # Create a list to store all trial information in
    full_trial_info = []

    # Loop over each element in trials and pull the desired values
    for i, elem in enumerate(trials):

        # If this is the first iteration, get the names of the hyperparameters
        if i==0:
            hyper_names = elem["misc"]["vals"].keys()

        # Test to see if we were successful or not
        if elem["result"]["status"] == STATUS_OK:

            # For this iteration, get the training time, training error, and testing error
            train_err = elem["result"]["train_err"]
            test_err = elem["result"]["loss"]
            run_time = elem["result"]["train_time"]

            # Identify the hyperparameters and create new rows for the full_trial_info
            full_trial_info.extend([[major_model, specific_model, i, run_time,
                                     train_err, test_err,
                                     hyper, elem["misc"]["vals"][hyper][0]]
                                     for hyper in hyper_names])
            
        # If we weren't successful, report messages
        else:
            
            # Identify the failure message
            message = elem["result"]["message"]
            
            # Identify the hyperparameters and create new rows for the full_trial_info
            full_trial_info.extend([[major_model, specific_model, i, 0,
                                     message, message, hyper,
                                     elem["misc"]["vals"][hyper][0]]
                                    for hyper in hyper_names])

    # Convert to a dataframe and return
    columns = ["MajorModel", "SpecificModel", "HyperRound", "RunTime",
               "TrainErr", "TestErr", "Hyper", "HyperVal"]
    return pd.DataFrame(full_trial_info, columns = columns)

# Define a function for a single round of hyperparameter optimization
def optimize(space, space_names = None, x = None, y = None, major_model = None,
             specific_model = None, training_params = {}, eval_metric = None,
             train_test_inds = None):
    """
    The function used by hyperopt.fmin() for hyperparameter optimization.
    
    Parameters
    ----------
    space: iterable of hp search spaces
        List containing the bounds and priors for hyperparameter optimization
    space_names: iterable of str
        Variable names associated with the values in 'space'
    x: numpy array
        Training features
    y: numpy array, 1D
        Training labels
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.
    training_params: dict
        These are parameters required for training the models specified by 
        'major_model' and 'specific_model'. Details on the requirements for each
        submodel can be found in the online documentation.
    eval_metric: func
        The function used for evaluating cross-validation error. This metric will
        be used to rank model architectures from best to worst. The function must
        take the form 'function(real_values, predicted_values)'.
    train_test_inds: list of lists: 
        Cross validation indices to use in training.
        
    Returns
    -------
    optimization_results: dict
        The results for a single round of hyperparameter optimization, including
        the testing loss, training loss, time to complete training, and whether
        or not the round was successful
    """
    # Get the shape of x
    x_shape = x.shape
    
    # Complete the search space. This involves building a dictionary linking
    # variable name to search space, converting appropriate values to the right
    # data types, and adding a few parameters in the case of LinearSVR
    model_params = space_to_model_params(space, space_names, major_model, 
                                         specific_model, x_shape)

    # Build the MLDE model
    test_model = MldeModel(major_model, specific_model, 
                           model_params = model_params, 
                           training_params = training_params, 
                           eval_metric = eval_metric)
    
    # Start a timer
    start = time()

    # # Try completing the rest of the function
    try:
        
        # Train the model and record training/testing error
        training_loss, testing_loss = test_model.train_cv(x, y, train_test_inds)

        # Clear the model
        test_model.clear_submodels()

        # End the timer and calculate training time
        end = time()
        train_time = end - start

        # Report results
        return {"loss": testing_loss,
                "status": STATUS_OK,
                "train_err": training_loss,
                "train_time": train_time}

    # Different route if hyperopt failed
    except Exception as e:
        
        # Return a failure message
        return {"status": STATUS_FAIL,
                "message": e}
            
# Define a function that performs hyperparameter optimization for a single MLDE
# model class, then uses the optimal hyperparameters to make predictions from the
# now-optimized model
def run_hyperopt(major_model, specific_model, training_params, 
                 sampled_x, sampled_y, x_to_predict, 
                 eval_metric = mse, train_test_inds = None, hyperopt_rounds = 100):
    """
    Executes hyperparameter optimization for a given inbuilt model
    
    Parameters
    ----------
    major_model: str
        Choice of 'Keras', 'XGB', or 'sklearn-regressor'. This argument
        tells MldeModel from which package we will be pulling models. 
    specific_model: str
        This arguments tells MldeModel which regressor to use within the package
        defined by major_model.    
    training_params: dict
        These are parameters required for training the models specified by 
        'major_model' and 'specific_model'. Details on the requirements for each
        submodel can be found in the online documentation.
    sampled_x: numpy array
        Training features
    sampled_y: numpy array, 1D
        Training labels
    x_to_predict: numpy array
        Features for which we want to predict labels
    eval_metric: func
        The function used for evaluating cross-validation error. This metric will
        be used to rank model architectures from best to worst. The function must
        take the form 'function(real_values, predicted_values)'.
    train_test_inds: list of lists
        Cross validation indices to use in training.
    hyperopt_rounds: int
        Number of rounds of hyperparameter optimization to perform
        
    Returns
    -------
    all_trial_info: pd.DataFrame
        The results of "process_trials" post hyperparameter optimization
    train_pred_results: tuple
        The results of MLDE.Support.RunMlde.TrainAndPredict.train_and_predict()
        using the best parameters identified during hyperparameter optimization
    """
    # Get the shape of x
    x_shape = sampled_x.shape
    
    # Build the search space
    space_var_names = list(SPACE_BY_MODEL[major_model][specific_model])
    search_space = [SEARCH_SPACES[major_model][space_var] 
                    for space_var in space_var_names]
    
    # Construct a dictionary for passing in kwargs to Optimize
    optimizer_kwargs = {"space_names": space_var_names, 
                        "x": sampled_x,
                        "y": sampled_y,
                        "major_model": major_model,
                        "specific_model": specific_model,
                        "training_params": training_params,
                        "eval_metric": eval_metric,
                        "train_test_inds": train_test_inds}
    
    # Construct the optimizer function
    complete_optimizer = partial(optimize, **optimizer_kwargs)
    
    # Build the trials object
    trials = Trials()
    
    # Run hyperparameter optimization
    best_params = fmin(complete_optimizer,
                       space = search_space,
                       algo = tpe.suggest,
                       max_evals = hyperopt_rounds,
                       trials = trials,
                       show_progressbar = False)
    
    # Reformat best_params to have the correct datatypes
    best_params = process_best(best_params, major_model, specific_model, x_shape)
    
    # Process the trials
    all_trial_info = process_trials(trials, major_model, specific_model)
    
    # Now build the model using the best parameters
    best_model = MldeModel(major_model, specific_model, 
                           model_params = best_params,
                           training_params = training_params, 
                           eval_metric = eval_metric)
    
    # Train and predict using the best model
    train_pred_results = train_and_predict(best_model, sampled_x = sampled_x,
                                            sampled_y = sampled_y,
                                            x_to_predict = x_to_predict,
                                            train_test_inds = train_test_inds)       

    # Return all relevant information
    return all_trial_info, train_pred_results